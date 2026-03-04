import torch
import torch.nn.functional as F
import cv2
import numpy as np
import argparse
from PIL import Image
from torchvision import transforms
from importlib import import_module
from utils import config
from utils.utils import yolobox2label
from utils.ckpt import load_ckpt

# ─── 圖片前處理（與 dataloader 一致）───
def preprocess_image(img_path, input_size=416):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # letterbox resize
    scale = min(input_size / h, input_size / w)
    nh, nw = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (nw, nh))

    dx = (input_size - nw) // 2
    dy = (input_size - nh) // 2
    img_padded = np.full((input_size, input_size, 3), 128, dtype=np.uint8)
    img_padded[dy:dy+nh, dx:dx+nw] = img_resized

    info = (h, w, nh, nw, dx, dy, scale)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(Image.fromarray(cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)))
    return tensor.unsqueeze(0), img, info   # (1,3,H,W), original img, info


# ─── 文字前處理（簡易 tokenizer）───
def preprocess_text(text, token_to_ix, max_len=20):
    tokens = text.lower().split()
    ids = [token_to_ix.get(t, token_to_ix.get('<unk>', 0)) for t in tokens]
    ids = ids[:max_len] + [0] * (max_len - len(ids))
    return torch.tensor(ids).unsqueeze(0)   # (1, max_len)


# ─── 畫出 bounding box ───
def draw_box(img, box, text):
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, text, (x1, max(y1-10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',       required=True,  help='config yaml path')
    parser.add_argument('--eval-weights', required=True,  help='checkpoint path')
    parser.add_argument('--image',        required=True,  help='input image path')
    parser.add_argument('--text',         required=True,  help='referring expression')
    parser.add_argument('--output',       default='output.jpg', help='output image path')
    args = parser.parse_args()

    # ── 載入 config ──
    cfg = config.load_cfg_from_cfg_file(args.config)

    # ── 載入 token vocab ──
    import json
    token_to_ix = json.load(open(f'data/anns/{cfg.DATASET}_token_to_ix.json'))
    ix_to_token = {v: k for k, v in token_to_ix.items()}
    pretrained_emb = np.load(f'data/anns/{cfg.DATASET}_pretrained_emb.npy')
    pretrained_emb = torch.from_numpy(pretrained_emb).float()

    # ── 載入模型 ──
    model_module = import_module(f'models.{cfg.MODEL}.net')
    net = model_module.Net(cfg, pretrained_emb, len(token_to_ix))
    net.cuda().eval()
    load_ckpt(cfg, net, None, None, None, args.eval_weights)
    print(f'✅ Model loaded from {args.eval_weights}')

    # ── 前處理 ──
    image_tensor, orig_img, info = preprocess_image(args.image)
    text_tensor  = preprocess_text(args.text, token_to_ix)

    image_tensor = image_tensor.cuda()
    text_tensor  = text_tensor.cuda()

    # ── Inference ──
    with torch.no_grad():
        box = net(image_tensor, text_tensor)   # (1, 1, 5) → [x,y,w,h,score]

    box = box.squeeze(1).cpu().numpy()[0]      # (5,)

    # ── 座標還原到原圖 ──
    box_label = yolobox2label(box, info)       # [x1, y1, x2, y2, score]

    # ── 視覺化 ──
    result = draw_box(orig_img.copy(), box_label, args.text)
    cv2.imwrite(args.output, result)
    print(f'✅ Result saved to {args.output}')
    print(f'   Box: x1={box_label[0]:.1f}, y1={box_label[1]:.1f}, '
          f'x2={box_label[2]:.1f}, y2={box_label[3]:.1f}')

    # ── 顯示結果（可選）──
    cv2.imshow('DViN Detection', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()