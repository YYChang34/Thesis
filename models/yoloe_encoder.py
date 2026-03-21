# coding=utf-8
"""
YOLOE Visual Encoder Wrapper

Wraps the Ultralytics YOLOE model to produce output compatible with
the existing DViN pipeline (same interface as YOLOv3 in visual_encoder.py).

YOLOE is anchor-free, but we adapt its outputs to anchor-based format
to maintain compatibility with WeakREChead's anchor selection logic.

Output format matches YOLOv3:
  - boxes_all: (B, total_anchors, n_classes+5) — concatenated predictions
  - feature_output: [large, medium, small] feature maps
      large:  (B, 1024, H/32, W/32)  — P5 (projected from YOLOE's channel)
      medium: (B, 512,  H/16, W/16)  — P4
      small:  (B, 256,  H/8,  W/8)   — P3
  - boxes_sml: list of per-scale box predictions
      each: (B, grid_h*grid_w, n_anchors, n_classes+5)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOEEncoder(nn.Module):
    """YOLOE visual encoder with anchor-based output adaptation.

    Extracts multi-scale features from YOLOE backbone+neck and converts
    anchor-free predictions to anchor-based format for WeakREChead compatibility.
    """

    # Virtual anchors per scale (to match YOLOv3's 4 anchors per cell)
    VIRTUAL_ANCHORS = {
        32: [(116, 90), (156, 198), (121, 240)],  # P5 large objects
        16: [(30, 61), (62, 45), (42, 119)],       # P4 medium objects
        8:  [(10, 13), (16, 30), (33, 23)],        # P3 small objects
    }
    N_ANCHORS = 4  # 3 virtual anchors + 1 anchor-free = 4 (matches YOLOv3)

    def __init__(self, __C):
        super(YOLOEEncoder, self).__init__()

        self.n_classes = __C.CLASS_NUM
        self.input_size = __C.INPUT_SHAPE[0]  # 416

        # Load YOLOE model from ultralytics
        from ultralytics import YOLO
        yoloe_variant = getattr(__C, 'YOLOE_VARIANT', 'weights/yoloe-v8l-seg.pt')
        self._yoloe = YOLO(yoloe_variant)
        self.backbone_neck = self._yoloe.model.model  # nn.Sequential of backbone+neck

        # Determine YOLOE neck output channels by probing
        self._p3_ch, self._p4_ch, self._p5_ch = self._probe_channels()

        # Channel projections to match expected [256, 512, 1024]
        self.p3_proj = nn.Conv2d(self._p3_ch, 256, 1) if self._p3_ch != 256 else nn.Identity()
        self.p4_proj = nn.Conv2d(self._p4_ch, 512, 1) if self._p4_ch != 512 else nn.Identity()
        self.p5_proj = nn.Conv2d(self._p5_ch, 1024, 1) if self._p5_ch != 1024 else nn.Identity()

        # Detection heads (anchor-based adaptation)
        # Each produces (n_anchors * (n_classes + 5)) channels
        n_ch_out = self.N_ANCHORS * (self.n_classes + 5)
        self.det_head_p5 = nn.Conv2d(1024, n_ch_out, 1)
        self.det_head_p4 = nn.Conv2d(512, n_ch_out, 1)
        self.det_head_p3 = nn.Conv2d(256, n_ch_out, 1)

        # Anchor guidance (matching YOLOv3Head interface)
        self.guide_wh_p5 = nn.Conv2d(1024, 2 * self.N_ANCHORS, 1)
        self.guide_wh_p4 = nn.Conv2d(512, 2 * self.N_ANCHORS, 1)
        self.guide_wh_p3 = nn.Conv2d(256, 2 * self.N_ANCHORS, 1)

    def _probe_channels(self):
        """Run a dummy forward pass to determine neck output channels."""
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.input_size, self.input_size)
            features = self._extract_neck_features(dummy)
        p3_ch = features[0].shape[1]
        p4_ch = features[1].shape[1]
        p5_ch = features[2].shape[1]
        return p3_ch, p4_ch, p5_ch

    def _extract_neck_features(self, x):
        """Extract P3, P4, P5 features from YOLOE backbone+neck.

        Returns list of [P3, P4, P5] feature maps.
        """
        # Ultralytics model stores intermediate features during forward
        outputs = {}
        n_modules = len(self.backbone_neck)
        for i, module in enumerate(self.backbone_neck):
            # Skip the detection head (last module) — it requires text embeddings
            # which are not available here; we only need neck features.
            if i == n_modules - 1:
                break

            if hasattr(module, 'f'):
                # Handle modules that take input from specific previous layers
                f = module.f
                if isinstance(f, int):
                    x = module(outputs.get(f, x) if f != -1 else x)
                elif isinstance(f, list):
                    x = module([outputs.get(j, x) for j in f])
                else:
                    x = module(x)
            else:
                x = module(x)
            outputs[i] = x

        # The last 3 feature maps from the neck are P3, P4, P5
        # Find the Detect/Segment head's input indices (filter to those actually
        # present in outputs — YOLOE head also has a text-embedding index which
        # will not be in outputs and must be excluded).
        detect_module = self.backbone_neck[-1]
        if hasattr(detect_module, 'f') and isinstance(detect_module.f, list):
            feat_indices = [idx for idx in detect_module.f if idx in outputs]
            if len(feat_indices) >= 3:
                features = [outputs[idx] for idx in feat_indices[:3]]
            else:
                keys = sorted(outputs.keys())
                features = [outputs[keys[-3]], outputs[keys[-2]], outputs[keys[-1]]]
        else:
            # Fallback: take the last 3 outputs
            keys = sorted(outputs.keys())
            features = [outputs[keys[-3]], outputs[keys[-2]], outputs[keys[-1]]]

        return features  # [P3 (stride 8), P4 (stride 16), P5 (stride 32)]

    def _make_anchor_predictions(self, feat, det_head, guide_wh, stride, anchors):
        """Convert feature map to anchor-based predictions matching YOLOv3Head output.

        Args:
            feat: (B, C, H, W) — projected feature map
            det_head: Conv2d producing (B, n_anchors*(n_classes+5), H, W)
            guide_wh: Conv2d for anchor width/height guidance
            stride: spatial stride (8, 16, or 32)
            anchors: list of (w, h) tuples for this scale

        Returns:
            refined_pred: (B, n_anchors*H*W, n_classes+5) — for boxes_all
            pred_new: (B, H*W, n_anchors, n_classes+5) — for boxes_sml
        """
        batchsize = feat.shape[0]
        fsize = feat.shape[2]
        n_ch = 5 + self.n_classes
        dtype = feat.dtype
        device = feat.device

        # Detection output
        output = det_head(feat)  # (B, n_anchors*n_ch, H, W)
        output = output.view(batchsize, self.N_ANCHORS, n_ch, fsize, fsize)
        output = output.permute(0, 1, 3, 4, 2).contiguous()

        # Anchor guidance
        wh_pred = guide_wh(feat)
        wh_pred = torch.exp(wh_pred)
        wh_pred = wh_pred.view(batchsize, self.N_ANCHORS, 2, fsize, fsize)
        wh_pred = wh_pred.permute(0, 1, 3, 4, 2).contiguous()

        # Grid shifts
        x_shift = torch.arange(fsize, dtype=dtype, device=device).view(1, 1, 1, fsize).expand(
            batchsize, self.N_ANCHORS, fsize, fsize)
        y_shift = torch.arange(fsize, dtype=dtype, device=device).view(1, 1, fsize, 1).expand(
            batchsize, self.N_ANCHORS, fsize, fsize)

        # Anchor dimensions
        masked_anchors = np.array(anchors)
        default_center = torch.zeros(batchsize, self.N_ANCHORS, fsize, fsize, 2,
                                     dtype=dtype, device=device)
        pred_anchors = torch.cat((default_center, wh_pred), dim=-1).contiguous()

        # Scale anchor-based predictions
        n_base = len(anchors)
        if n_base > 0:
            w_anchors = torch.tensor(
                masked_anchors[:, 0], dtype=dtype, device=device
            ).view(1, n_base, 1, 1).expand(batchsize, n_base, fsize, fsize)
            h_anchors = torch.tensor(
                masked_anchors[:, 1], dtype=dtype, device=device
            ).view(1, n_base, 1, 1).expand(batchsize, n_base, fsize, fsize)

            pred_anchors[:, :n_base, :, :, 2] *= w_anchors
            pred_anchors[:, :n_base, :, :, 3] *= h_anchors

        # Anchor-free slot
        if self.N_ANCHORS > n_base:
            pred_anchors[:, n_base:, :, :, 2] *= stride * 4
            pred_anchors[:, n_base:, :, :, 3] *= stride * 4

        pred_anchors[..., :2] = pred_anchors[..., :2].detach()

        # Final predictions
        pred = output.clone()
        pred[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(pred[..., np.r_[:2, 4:n_ch]])
        pred[..., 0] += x_shift
        pred[..., 1] += y_shift
        pred[..., :2] *= stride
        pred[..., 2] = torch.exp(pred[..., 2]) * pred_anchors[..., 2]
        pred[..., 3] = torch.exp(pred[..., 3]) * pred_anchors[..., 3]

        # Reshape to match YOLOv3 output format
        pred_new = pred.view(batchsize, -1, fsize * fsize, n_ch).permute(0, 2, 1, 3)
        refined_pred = pred.view(batchsize, -1, n_ch)

        return refined_pred.data, pred_new.data

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input image tensor

        Returns:
            boxes_all:      (B, total_preds, n_classes+5) concatenated from all scales
            feature_output: [large, medium, small] = [(B,1024,H/32,W/32), (B,512,H/16,W/16), (B,256,H/8,W/8)]
            boxes_sml:      [pred_p5, pred_p4, pred_p3] each (B, grid*grid, n_anchors, n_ch)
        """
        # Extract neck features
        raw_features = self._extract_neck_features(x)
        p3_raw, p4_raw, p5_raw = raw_features  # stride 8, 16, 32

        # Project to standard channel dimensions
        p3 = self.p3_proj(p3_raw)   # (B, 256, H/8, W/8)
        p4 = self.p4_proj(p4_raw)   # (B, 512, H/16, W/16)
        p5 = self.p5_proj(p5_raw)   # (B, 1024, H/32, W/32)

        # Feature output: [large, medium, small] matching YOLOv3 convention
        feature_output = [p5, p4, p3]

        # Generate anchor-based predictions per scale
        pred_p5, box_p5 = self._make_anchor_predictions(
            p5, self.det_head_p5, self.guide_wh_p5, 32, self.VIRTUAL_ANCHORS[32])
        pred_p4, box_p4 = self._make_anchor_predictions(
            p4, self.det_head_p4, self.guide_wh_p4, 16, self.VIRTUAL_ANCHORS[16])
        pred_p3, box_p3 = self._make_anchor_predictions(
            p3, self.det_head_p3, self.guide_wh_p3, 8, self.VIRTUAL_ANCHORS[8])

        boxes_all = torch.cat([pred_p5, pred_p4, pred_p3], dim=1)
        boxes_sml = [box_p5, box_p4, box_p3]

        return boxes_all, feature_output, boxes_sml
