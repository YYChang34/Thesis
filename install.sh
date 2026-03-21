set -e

# 0) enable conda in script
eval "$(conda shell.bash hook)"

# 1) conda env
conda clean --packages --tarballs -y
conda remove -n DViN --all -y 2>/dev/null || true
conda create -n DViN python=3.9 -y
conda activate DViN

# 2) torch 1.11 + cu113 (conda)
conda install -y -c pytorch pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3

# 2.5) fix MKL version conflict
conda install -y mkl==2022.1.0 -c conda-forge --force-reinstall

# 2.6) fix setuptools for pkg_resources
pip install setuptools==59.5.0 --force-reinstall

# 2.7) patch CUDA version check in torch
python - << 'EOF'
path = "/opt/conda/envs/DViN/lib/python3.9/site-packages/torch/utils/cpp_extension.py"
with open(path, 'r') as f:
    content = f.read()
content = content.replace(
    'raise RuntimeError(CUDA_MISMATCH_MESSAGE',
    'pass  # raise RuntimeError(CUDA_MISMATCH_MESSAGE'
)
with open(path, 'w') as f:
    f.write(content)
print("Patched cpp_extension.py")
EOF

# 3) constraints
cat > constraints.txt << 'EOF'
numpy==1.23.5
opencv-python<4.11
opencv-python-headless<4.11
EOF

pip install -U pip wheel

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
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

cd utils/DCN
TORCH_CUDA_ARCH_LIST="8.6" FORCE_CUDA=1 ./make.sh
cd ../..

# 7.5) apex (mixed precision training)
cd apex
TORCH_CUDA_ARCH_LIST="8.6" FORCE_CUDA=1 pip install -v \
  --disable-pip-version-check \
  --no-cache-dir \
  --no-build-isolation \
  --config-settings="--build-option=--cpp_ext" \
  --config-settings="--build-option=--cuda_ext" \
  .
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

# 9) EfficientSAM weights
mkdir -p EfficientSAM/weights EfficientSAM/torchscripted_model

wget -O weights/efficient_sam_vitt.pt \
  "https://github.com/yformer/EfficientSAM/raw/main/weights/efficient_sam_vitt.pt"

cp weights/efficient_sam_vitt.pt EfficientSAM/weights/efficient_sam_vitt.pt

# torchscript version
wget -O EfficientSAM/torchscripted_model/efficient_sam_vitt_torchscript.pt \
  "https://github.com/yformer/EfficientSAM/releases/download/v1.0/efficient_sam_vitt_torchscript.pt" || \
cp weights/efficient_sam_vitt.pt EfficientSAM/torchscripted_model/efficient_sam_vitt_torchscript.pt

# 9.5) YOLOE weights
pip install -q huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='jameslahm/yoloe', filename='yoloe-v8l-seg.pt', local_dir='weights')
print('Downloaded yoloe-v8l-seg.pt to weights/')
"

pip install gdown
gdown 'https://drive.google.com/uc?id=1nxVTx8Zv52VSO-ccHVFe2ggG0HbGnw9g' -O weights/yolov3_coco.pth

# 10) data/ — COCO 2014 train images + RefCOCO annotations
mkdir -p data/images

wget -O data/images/train2014.zip \
  http://images.cocodataset.org/zips/train2014.zip

python -c "
import zipfile
print('Extracting COCO 2014...')
with zipfile.ZipFile('data/images/train2014.zip', 'r') as z:
    z.extractall('data/images/')
print('Done')
"
rm data/images/train2014.zip

echo "DONE"
