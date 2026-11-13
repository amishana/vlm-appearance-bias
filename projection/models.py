"""Per-model adapters. Each one loads the model and preprocesses images exactly
like its baseline runner in baseline/, and names the connector module the
eraser hooks.

    backend      checkpoint                connector (vision -> LM)       d
    qwen2_5_vl   Qwen2.5-VL-3B-Instruct    visual.merger (patch merger)   2048
    smolvlm2     SmolVLM2-2.2B-Instruct    connector (pixel shuffle+proj) 2048
    internvl3    InternVL3-8B              mlp1 (2-layer MLP projector)   3584
"""

from pathlib import Path

import torch
from PIL import Image

from projection.common import PROMPT

DEVICE = "cuda:0"


def find_module(model, suffix):
    hits = [(n, m) for n, m in model.named_modules()
            if n == suffix or n.endswith("." + suffix)]
    if len(hits) != 1:
        raise RuntimeError(f"expected one module named '*{suffix}', found "
                           f"{[n for n, _ in hits]}")
    return hits[0]


class _HFChatBackend:
    """Qwen2.5-VL and SmolVLM2: AutoProcessor + chat template + generate."""
    connector = None

    def _load_model(self, path):
        raise NotImplementedError

    def load(self, path):
        from transformers import AutoProcessor
        self.path = path
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.model = self._load_model(path).to(DEVICE).eval()
        messages = [{"role": "user", "content": [{"type": "image"},
                                                 {"type": "text", "text": PROMPT}]}]
        self.prompt_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        return self

    def _inputs(self, image_path):
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        return self.processor(text=self.prompt_text, images=[image],
                              return_tensors="pt").to(DEVICE)

    def encode(self, image_path):
        """Run far enough to reach the connector (a capture hook stops it)."""
        self.model(**self._inputs(image_path))

    def generate(self, image_path):
        inputs = self._inputs(image_path)
        out = self.model.generate(**inputs, max_new_tokens=32, do_sample=False)
        return self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    def baseline_settings(self, split):
        return {
            "model": str(Path(self.path).resolve()), "split": split, "prompt": PROMPT,
            "max_new_tokens": 32, "do_sample": False, "dtype": "bfloat16",
            "torch_version": torch.__version__,
        }


class QwenBackend(_HFChatBackend):
    connector = "visual.merger"

    def _load_model(self, path):
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path, dtype=torch.bfloat16, local_files_only=True)


class SmolBackend(_HFChatBackend):
    connector = "connector"

    def _load_model(self, path):
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(
            path, dtype=torch.bfloat16, local_files_only=True)


class InternBackend:
    """Mirrors baseline/run_internvl3.py: one 448x448 tile, model.chat()."""
    connector = "mlp1"

    def load(self, path):
        from torchvision import transforms as T
        from torchvision.transforms.functional import InterpolationMode
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        self.path = path
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=True, local_files_only=True, fix_mistral_regex=True)
        config = AutoConfig.from_pretrained(path, trust_remote_code=True,
                                            local_files_only=True)
        model_class = get_class_from_dynamic_module(
            config.auto_map["AutoModel"], path, local_files_only=True)
        if not hasattr(model_class, "all_tied_weights_keys"):
            model_class.all_tied_weights_keys = {}
        self.model = AutoModel.from_pretrained(
            path, trust_remote_code=True, local_files_only=True,
            dtype=torch.bfloat16, low_cpu_mem_usage=True).eval().cuda()
        self.transform = T.Compose([
            T.Lambda(lambda image: image.convert("RGB")),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        return self

    def _pixels(self, image_path):
        with Image.open(image_path) as image:
            px = self.transform(image).unsqueeze(0)
        return px.to(device=DEVICE, dtype=torch.bfloat16)

    def encode(self, image_path):
        self.model.extract_feature(self._pixels(image_path))

    def generate(self, image_path):
        return self.model.chat(self.tokenizer, self._pixels(image_path),
                               "<image>\n" + PROMPT,
                               {"max_new_tokens": 32, "do_sample": False}).strip()

    def baseline_settings(self, split):
        return {
            "model": str(Path(self.path).resolve()), "split": split, "prompt": PROMPT,
            "max_new_tokens": 32, "do_sample": False, "dtype": "bfloat16",
            "torch_version": torch.__version__, "image_size": 448, "image_tiles": 1,
            "image_preprocessing": "resize_448_imagenet_normalization",
            "transformers_compatibility_patch": "all_tied_weights_keys",
        }


BACKENDS = {"qwen2_5_vl": QwenBackend, "smolvlm2": SmolBackend,
            "internvl3": InternBackend}


def load_backend(name, path):
    backend = BACKENDS[name]().load(path)
    module_name, module = find_module(backend.model, backend.connector)
    return backend, module_name, module
