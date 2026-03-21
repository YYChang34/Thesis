"""
Minimal smoke test to verify environment setup and model forward pass.
Usage:
    python smoke_test.py           # full test (imports + GPU + weights + forward pass)
    python smoke_test.py --quick   # quick test (imports + GPU + weights only)
"""

import sys
import os
import argparse
import traceback

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"

def section(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print('='*50)

def check(label, fn):
    try:
        result = fn()
        msg = f" ({result})" if result else ""
        print(f"  {PASS} {label}{msg}")
        return True
    except Exception as e:
        print(f"  {FAIL} {label}")
        print(f"         {e}")
        return False

# ─────────────────────────────────────────────
# Stage 1: Imports
# ─────────────────────────────────────────────
def stage_imports():
    section("Stage 1: Import Check")
    results = []

    results.append(check("torch", lambda: __import__('torch').__version__))
    results.append(check("torchvision", lambda: __import__('torchvision').__version__))
    results.append(check("numpy", lambda: __import__('numpy').__version__))
    results.append(check("PIL (Pillow)", lambda: __import__('PIL').__version__))
    results.append(check("cv2 (opencv)", lambda: __import__('cv2').__version__))
    results.append(check("timm", lambda: __import__('timm').__version__))
    results.append(check("transformers", lambda: __import__('transformers').__version__))
    results.append(check("open_clip", lambda: __import__('open_clip').__version__))
    results.append(check("spacy", lambda: __import__('spacy').__version__))
    results.append(check("yaml", lambda: __import__('yaml').__version__))
    results.append(check("omegaconf", lambda: __import__('omegaconf').__version__))

    # Project-internal imports
    sys.path.insert(0, os.path.dirname(__file__))
    results.append(check("models.visual_encoder", lambda: __import__('models.visual_encoder')))
    results.append(check("models.language_encoder", lambda: __import__('models.language_encoder')))
    results.append(check("models.clip_encoder", lambda: __import__('models.clip_encoder')))
    results.append(check("models.DViR.head", lambda: __import__('models.DViR.head')))
    results.append(check("EfficientSAM.efficient_sam.build_efficient_sam",
                         lambda: __import__('EfficientSAM.efficient_sam.build_efficient_sam')))
    results.append(check("utils.config", lambda: __import__('utils.config')))

    return all(results)

# ─────────────────────────────────────────────
# Stage 2: GPU / CUDA
# ─────────────────────────────────────────────
def stage_gpu():
    section("Stage 2: GPU / CUDA Check")
    import torch
    results = []

    results.append(check("CUDA available", lambda: f"torch.cuda.is_available() = {torch.cuda.is_available()}"
                         if torch.cuda.is_available() else (_ for _ in ()).throw(RuntimeError("CUDA not available"))))
    if not torch.cuda.is_available():
        print(f"  {FAIL} CUDA not found - cannot proceed with GPU tests")
        return False

    results.append(check("GPU count", lambda: f"{torch.cuda.device_count()} GPU(s) detected"))
    results.append(check("GPU name", lambda: torch.cuda.get_device_name(0)))
    results.append(check("Simple CUDA op", lambda: (
        torch.tensor([1.0, 2.0]).cuda().sum().item()
    ) and "OK"))

    return all(results)

# ─────────────────────────────────────────────
# Stage 3: Weight Files
# ─────────────────────────────────────────────
def stage_weights():
    section("Stage 3: Weight Files Check")
    results = []

    weight_files = [
        ("weights/efficient_sam_vitt.pt",   "EfficientSAM-ViT-T"),
        ("weights/dinov2",                  "DINOv2 (directory)"),
        ("yolov3_coco.pth",                 "YOLOv3 pretrained"),
    ]

    for path, name in weight_files:
        full_path = os.path.join(os.path.dirname(__file__), path)
        exists = os.path.exists(full_path)
        label = f"{name} -> {path}"
        if exists:
            size = ""
            if os.path.isfile(full_path):
                size_mb = os.path.getsize(full_path) / 1e6
                size = f"{size_mb:.1f} MB"
            print(f"  {PASS} {label} {size}")
            results.append(True)
        else:
            print(f"  {FAIL} {label}  (NOT FOUND)")
            results.append(False)

    # Check DINOv2 subdirectory has essential files
    dinov2_dir = os.path.join(os.path.dirname(__file__), "weights/dinov2")
    if os.path.isdir(dinov2_dir):
        check("DINOv2 config.json inside weights/dinov2/",
              lambda: os.path.isfile(os.path.join(dinov2_dir, 'config.json'))
              or (_ for _ in ()).throw(FileNotFoundError("config.json missing in weights/dinov2")))

    return all(results)

# ─────────────────────────────────────────────
# Stage 4: Model Forward Pass
# ─────────────────────────────────────────────
def stage_forward_pass():
    section("Stage 4: Model Forward Pass (dummy data, batch=1)")
    import torch
    import numpy as np
    from utils.config import CfgNode

    # Minimal config mirroring refcoco.yaml
    cfg = CfgNode({
        'MODEL':       'DViR',
        'USE_GLOVE':   True,
        'WORD_EMBED_SIZE': 300,
        'EMBED_FREEZE': True,
        'HIDDEN_SIZE': 512,
        'DROPOUT_R':   0.1,
        'MULTI_HEAD':  8,
        'FF_SIZE':     2048,
        'FLAT_GLIMPSES': 1,
        'LANG_ENC':    'lstm',
        'N_SA':        3,
        'VIS_ENC':     'yolov3',
        'VIS_FREEZE':  True,
        'SELECT_NUM':  17,
        'CLASS_NUM':   80,
    })

    TOKEN_SIZE = 1000
    dummy_emb  = np.random.randn(TOKEN_SIZE, 300).astype(np.float32)

    print("  Loading model (may take 30-60s on first run)...")

    ok = check("Net instantiation", lambda: (
        __import__('models.DViR.net_v2', fromlist=['Net']).Net(cfg, dummy_emb, TOKEN_SIZE)
        and "OK"
    ))
    if not ok:
        return False

    from models.DViR.net_v2 import Net
    net = Net(cfg, dummy_emb, TOKEN_SIZE)
    net.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)

    B = 1
    dummy_img   = torch.randn(B, 3, 416, 416).to(device)
    dummy_token = torch.randint(0, TOKEN_SIZE, (B, 15)).to(device)

    def run_forward():
        with torch.no_grad():
            out = net(dummy_img, dummy_token)
        return f"output shape {out.shape}"

    return check("Forward pass (eval mode)", run_forward)

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='Skip model forward pass (stages 1-3 only)')
    args = parser.parse_args()

    print("\n" + "="*50)
    print("  DViR Smoke Test")
    print("="*50)

    stages = [
        ("Imports",      stage_imports),
        ("GPU/CUDA",     stage_gpu),
        ("Weight Files", stage_weights),
    ]
    if not args.quick:
        stages.append(("Forward Pass", stage_forward_pass))

    passed = 0
    failed = 0
    for name, fn in stages:
        try:
            ok = fn()
        except Exception:
            traceback.print_exc()
            ok = False
        if ok:
            passed += 1
        else:
            failed += 1

    section("Summary")
    print(f"  Passed: {passed}/{len(stages)}")
    print(f"  Failed: {failed}/{len(stages)}")
    if failed == 0:
        print("\n  環境確認完成！可以開始訓練。\n")
    else:
        print("\n  有問題需要修正，請檢查上方 [FAIL] 項目。\n")
        sys.exit(1)

if __name__ == '__main__':
    main()
