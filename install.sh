set -e

# 0) enable conda in script
eval "$(conda shell.bash hook)"

# 1) conda env
conda create -n DViN python=3.9 -y
conda activate DViN

# 2) torch 1.11 + cu113 (conda)
conda install -y -c pytorch pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3

# 3) constraints
cat > constraints.txt << 'EOF'
numpy==1.23.5
opencv-python<4.11
opencv-python-headless<4.11
EOF

pip install -U pip setuptools wheel

# 4) repo requirements (guarded)
pip install -c constraints.txt -r requirements.txt

# 5) extra pinned deps needed by actual run
pip install -c constraints.txt \
  timm==0.6.13 \
  open_clip_torch==2.0.2 \
  transformers==4.33.1 \
  tokenizers==0.13.3 \
  einops ftfy regex sentencepiece tqdm ninja \
  ultralytics

# 6) spacy vectors
wget https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.1.0/en_vectors_web_lg-2.1.0.tar.gz -O en_vectors_web_lg-2.1.0.tar.gz
pip install en_vectors_web_lg-2.1.0.tar.gz

# 7) DCN
cd utils/DCN
./make.sh
cd ../..

# 7.5) apex (mixed precision training)
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
  --global-option="--cpp_ext" --global-option="--cuda_ext" .
cd ..

# 8) weights/
mkdir -p weights/clip weights/dinov2

python -c "
from transformers import CLIPModel, CLIPProcessor
model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
model.save_pretrained('weights/clip')
processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
processor.save_pretrained('weights/clip')
"

python -c "
from transformers import AutoModel
model = AutoModel.from_pretrained('facebook/dinov2-large')
model.save_pretrained('weights/dinov2')
"

wget -O weights/efficient_sam_vitt.pt \
  https://huggingface.co/yunyangx/efficient-sam/resolve/main/efficient_sam_vitt.pt

# 9) EfficientSAM weights & torchscripted_model
mkdir -p EfficientSAM/weights EfficientSAM/torchscripted_model

wget -O EfficientSAM/weights/efficient_sam_vitt.pt \
  https://huggingface.co/yunyangx/efficient-sam/resolve/main/efficient_sam_vitt.pt

wget -O EfficientSAM/torchscripted_model/efficient_sam_vitt_torchscript.pt \
  https://huggingface.co/yunyangx/efficient-sam/resolve/main/efficient_sam_vitt_torchscript.pt

# 9.5) YOLOE weights (for net_v3)
python -c "
from ultralytics import YOLO
model = YOLO('yoloe-s.pt')
"
mv yoloe-s.pt weights/ 2>/dev/null || true

pip install gdown
gdown "https://drive.google.com/uc?id=1nxVTx8Zv52VSO-ccHVFe2ggG0HbGnw9g" -O weights/yolov3_coco.pth

# 10) data/ — COCO 2014 train images + RefCOCO annotations
# NOTE: RefCOCO annotations require access from https://github.com/lichengunc/refer
mkdir -p data/images

wget -O data/images/train2014.zip \
  http://images.cocodataset.org/zips/train2014.zip
unzip data/images/train2014.zip -d data/images/
rm data/images/train2014.zip

echo "DONE"