import torch
import torch.nn as nn
from PIL import Image
from models.language_encoder import language_encoder
from models.visual_encoder import visual_encoder
from models.DViR.head import WeakREChead
from models.network_blocks import MultiScaleFusion
import timm
import open_clip
from torch.nn import ModuleList, Linear
import torch.nn.functional as F
from models.clip_encoder import CLIPVisionTower
from transformers import logging
from transformers import Dinov2Model
from EfficientSAM.efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
from EfficientSAM.efficient_sam.efficient_sam import EfficientSam
logging.set_verbosity_error()


class CrossAttentionRouter(nn.Module):
    def __init__(self, dim=1024, num_experts=4):
        super().__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj   = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, yolo_feat, expert_feats):
        # yolo_feat:    (B, 1024)
        # expert_feats: (B, 4, 1024)
        q = self.query_proj(yolo_feat).unsqueeze(1)        # (B, 1, 1024)
        k = self.key_proj(expert_feats)                     # (B, 4, 1024)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, 1, 4)
        router_logits = torch.softmax(attn.squeeze(1), dim=-1)     # (B, 4)
        return router_logits


class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size):
        super(Net, self).__init__()
        self.select_num = __C.SELECT_NUM
        self.visual_encoder = visual_encoder(__C).eval()
        self.lang_encoder = language_encoder(__C, pretrained_emb, token_size)

        self.linear_vs = nn.Linear(1024, __C.HIDDEN_SIZE)
        self.linear_ts = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.head = WeakREChead(__C)
        self.multi_scale_manner = MultiScaleFusion(v_planes=(256, 512, 1024), hiden_planes=1024, scaled=True)
        self.linear     = nn.Linear(768, 1024)
        self.linear_sam = nn.Linear(256, 1024)
        self.clip_model = CLIPVisionTower(vision_tower="openai/clip-vit-base-patch32", input_image_size=224, select_layer=12)
        self.clip_model.eval()
        self.dino_model = Dinov2Model.from_pretrained('weights/dinov2')
        self.dino_model.eval()
        self.efficientsam = build_efficient_sam_vitt()
        self.efficientsam.eval()
        self.convnext_model = timm.create_model('convnext_tiny', pretrained=True)
        self.convnext_model.eval()

        self.router = CrossAttentionRouter(dim=1024, num_experts=4)
        self.expert_pool_proj = nn.Linear(1024, 1024)
        self.lambda_sparse = 0.01
        self.num_experts   = 4

        self.linear_decoder = nn.Sequential(
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 512)
        )
        self.projector_expert = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 128)
        )
        self.projector_yolo = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 128)
        )

        self.class_num = __C.CLASS_NUM
        if __C.VIS_FREEZE:
            self.frozen(self.efficientsam)
        if __C.VIS_FREEZE:
            self.frozen(self.convnext_model)
        if __C.VIS_FREEZE:
            self.frozen(self.clip_model)
        if __C.VIS_FREEZE:
            self.frozen(self.visual_encoder)
        if __C.VIS_FREEZE:
            self.frozen(self.dino_model)

    def frozen(self, module):
        if getattr(module, 'module', False):
            for child in module.module():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = False

    def re_normalize(self, x):
        original_mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        original_std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        new_mean = torch.tensor([0.48145466, 0.4578275,  0.40821073], device=x.device).view(1, 3, 1, 1)
        new_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1, 3, 1, 1)
        x_unnormalized = x * original_std + original_mean
        return (x_unnormalized - new_mean) / new_std

    def load_balancing_loss(self, router_logits):
        # router_logits: (B, 4)
        mean_probs = router_logits.mean(dim=0)                         # (4,)
        target     = torch.full_like(mean_probs, 1.0 / self.num_experts)
        return F.mse_loss(mean_probs, target)

    def forward(self, x, y):

        # ── Vision & Language Encoding ──
        with torch.no_grad():
            boxes_all, x_, boxes_sml = self.visual_encoder(x)
        y_ = self.lang_encoder(y)

        x_new_pro = self.re_normalize(x)
        resized_image_feature_dino = F.interpolate(x, size=(364, 364), mode='bilinear', align_corners=False)
        target_size = self.efficientsam.image_encoder.img_size
        resized_image_feature_sam  = F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=False)

        # ── CLIP feature ──
        with torch.no_grad():
            clip_feature = self.clip_model(x_new_pro).to(x.device)
        # clip_feature: (B, seq=169, 768)
        clip_feature = self.linear(clip_feature)                       # (B, 169, 1024)
        # ✅ 修正：先 permute 再 view，確保 shape 正確為 (B, 1024, 13, 13)
        clip_feature = clip_feature.permute(0, 2, 1)                   # (B, 1024, 169)
        clip_feature = clip_feature.view(
            clip_feature.size(0), clip_feature.size(1), 13, 13)        # (B, 1024, 13, 13)

        # ── DINOv2 feature ──
        with torch.no_grad():
            dino_feature = self.dino_model(resized_image_feature_dino).last_hidden_state.to(x.device)
        dino_feature = dino_feature[:, 1:, :]                          # 去掉 CLS token: (B, 676, 768)
        dino_feature = dino_feature.transpose(1, 2).contiguous()       # (B, 768, 676)
        dino_feature = dino_feature.view(
            dino_feature.size(0), dino_feature.size(1), 26, 26)        # (B, 768, 26, 26)
        dino_feature = F.avg_pool2d(dino_feature, kernel_size=2, stride=2)  # (B, 768, 13, 13)
        dino_feature = self.linear(
            dino_feature.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)     # (B, 1024, 13, 13)

        # ── ConvNeXt feature ──
        with torch.no_grad():
            convnext_feature = self.convnext_model.forward_features(x).to(x.device)
            convnext_feature = convnext_feature.permute(0, 2, 3, 1)   # (B, H, W, C)
        # ✅ 修正：interpolate 到 13×13 再投影，確保 spatial size 一致
        convnext_feature = self.linear(convnext_feature)               # (B, H, W, 1024)
        convnext_feature = convnext_feature.permute(0, 3, 1, 2)        # (B, 1024, H, W)
        convnext_feature = F.interpolate(
            convnext_feature, size=(13, 13), mode='bilinear', align_corners=False)  # (B, 1024, 13, 13)

        # ── SAM feature ──
        with torch.no_grad():
            sam_feature = self.efficientsam.image_encoder(resized_image_feature_sam).to(x.device)
        sam_feature = self.linear_sam(sam_feature.permute(0, 2, 3, 1))  # (B, H, W, 1024)
        sam_feature = sam_feature.permute(0, 3, 1, 2)                   # (B, 1024, H, W)
        sam_feature = F.interpolate(
            sam_feature, size=(13, 13), mode='bilinear', align_corners=False)       # (B, 1024, 13, 13)

        # ── Vision Multi Scale Fusion ──
        s, m, l = x_
        l_new, m_new, s_new = self.multi_scale_manner([l, m, s])
        s_new_original = s_new.clone()

        # ✅ 確認所有 Expert 特徵 shape 一致後才 stack
        # dynamic_features 中每個元素都應是 (B, 1024, 13, 13)
        dynamic_features = [clip_feature, dino_feature, sam_feature, convnext_feature]
        assert all(f.shape == dynamic_features[0].shape for f in dynamic_features), \
            f"Expert feature shape mismatch: {[f.shape for f in dynamic_features]}"

        # ── Router ──
        yolo_feature = F.adaptive_avg_pool2d(s_new, (1, 1)).flatten(1)  # (B, 1024)

        # CrossAttentionRouter：YOLO 作為 Query，Expert 作為 Key
        expert_pooled = torch.stack([
            F.adaptive_avg_pool2d(f, (1, 1)).flatten(1)   # (B, 1024)
            for f in dynamic_features
        ], dim=1)                                           # (B, 4, 1024)
        expert_pooled = self.expert_pool_proj(expert_pooled)  # (B, 4, 1024)

        router_logits = self.router(
            yolo_feature.detach(),
            expert_pooled.detach()
        )                                                   # (B, 4)

        # Load Balancing Loss
        sparse_loss = self.load_balancing_loss(router_logits)

        # ── Soft Router：全部 Expert 加權融合 ──
        # ✅ 修正：stack 後是 (B, 4, 1024, 13, 13)，weights 需要是 (B, 4, 1, 1, 1)
        selected_features = torch.stack(dynamic_features, dim=1)   # (B, 4, 1024, 13, 13)
        weights = router_logits.view(
            router_logits.size(0), router_logits.size(1), 1, 1, 1) # (B, 4, 1, 1, 1)
        expert_fused = (selected_features * weights).sum(dim=1)    # (B, 1024, 13, 13)

        # YOLO 與 Expert 動態加權
        yolo_weight   = 1.0 - router_logits.max(dim=1).values.mean()
        expert_weight = 1.0 - yolo_weight
        s_new = s_new * yolo_weight + expert_fused * expert_weight  # (B, 1024, 13, 13)

        # ── Contrastive Loss ──
        z_expert = self.projector_expert(expert_fused)
        z_yolo   = self.projector_yolo(s_new_original)
        temperature = 0.5
        batch_size  = z_expert.size(0)
        z_expert_norm = F.normalize(z_expert, dim=1)
        z_yolo_norm   = F.normalize(z_yolo,   dim=1)
        similarity_matrix = torch.matmul(z_expert_norm, z_yolo_norm.T)
        labels    = torch.arange(batch_size).to(x.device)
        criterion = nn.CrossEntropyLoss()
        loss_contrastive = criterion(similarity_matrix / temperature, labels)

        x_ = [s_new, m_new, l_new]

        # ── Anchor Selection ──
        boxes_sml_new = []
        mean_i = torch.mean(boxes_sml[0], dim=2, keepdim=True)
        mean_i = mean_i.squeeze(2)[:, :, 4]
        vals, indices = mean_i.topk(k=int(self.select_num), dim=1, largest=True, sorted=True)
        bs, gridnum, anncornum, ch = boxes_sml[0].shape
        bs_, selnum = indices.shape
        box_sml_new = boxes_sml[0].masked_select(
            torch.zeros(bs, gridnum).to(boxes_sml[0].device)
                .scatter(1, indices, 1).bool()
                .unsqueeze(2).unsqueeze(3)
                .expand(bs, gridnum, anncornum, ch)
        ).contiguous().view(bs, selnum, anncornum, ch)
        boxes_sml_new.append(box_sml_new)

        batchsize, dim, h, w = x_[0].size()
        i_new = x_[0].view(batchsize, dim, h * w).permute(0, 2, 1)
        bs, gridnum, ch = i_new.shape
        i_new = i_new.masked_select(
            torch.zeros(bs, gridnum).to(i_new.device)
                .scatter(1, indices, 1).bool()
                .unsqueeze(2).expand(bs, gridnum, ch)
        ).contiguous().view(bs, selnum, ch)

        recon_text_feature = self.linear_decoder(i_new)
        recon_text_feature_pooled = recon_text_feature.mean(dim=1, keepdim=True)
        recon_loss = F.mse_loss(recon_text_feature_pooled, y_['flat_lang_feat'].unsqueeze(1))

        # ── Anchor-based Contrastive Learning ──
        x_new = self.linear_vs(i_new)
        y_new = self.linear_ts(y_['flat_lang_feat'].unsqueeze(1))
        if self.training:
            loss = self.head(x_new, y_new)
            total_loss = loss + recon_loss + self.lambda_sparse * sparse_loss + loss_contrastive
            return total_loss
        else:
            predictions_s   = self.head(x_new, y_new)
            box_pred = get_boxes(boxes_sml_new, [predictions_s], self.class_num)
            return box_pred


def get_boxes(boxes_sml, predictionslist, class_num):
    batchsize = predictionslist[0].size()[0]
    pred = []
    for i in range(len(predictionslist)):
        mask = predictionslist[i].squeeze(1)
        masked_pred  = boxes_sml[i][mask]
        refined_pred = masked_pred.view(batchsize, -1, class_num + 5)
        refined_pred[:, :, 0] = refined_pred[:, :, 0] - refined_pred[:, :, 2] / 2
        refined_pred[:, :, 1] = refined_pred[:, :, 1] - refined_pred[:, :, 3] / 2
        refined_pred[:, :, 2] = refined_pred[:, :, 0] + refined_pred[:, :, 2]
        refined_pred[:, :, 3] = refined_pred[:, :, 1] + refined_pred[:, :, 3]
        pred.append(refined_pred.data)
    boxes = torch.cat(pred, 1)
    score = boxes[:, :, 4]
    max_score, ind = torch.max(score, -1)
    ind_new = ind.unsqueeze(1).unsqueeze(1).repeat(1, 1, 5)
    box_new = torch.gather(boxes, 1, ind_new)
    return box_new