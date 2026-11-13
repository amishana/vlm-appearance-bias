python -m pip install --no-cache-dir torch torchvision transformers accelerate scikit-learn
python -m pip install --no-cache-dir timm einops num2words

python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, '| transformers', transformers.__version__)"

# Models
MODELS=~/Documents/models
mkdir -p "$MODELS"
hf download Qwen/Qwen2.5-VL-3B-Instruct          --local-dir "$MODELS/Qwen2.5-VL-3B-Instruct"
hf download HuggingFaceTB/SmolVLM2-2.2B-Instruct --local-dir "$MODELS/SmolVLM2-2.2B-Instruct"
hf download OpenGVLab/InternVL3-8B               --local-dir "$MODELS/InternVL3-8B"
