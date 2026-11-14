"""Counterfactual consistency fine-tuning for the team's generative VLMs.

Port of the teammate's CLIP objective (counterfactual-team-code/cf_model.py,
group_objective) to Qwen2.5-VL-3B, SmolVLM2-2.2B and InternVL3-8B:

    loss(group) = mean_i CE(answer tokens of image i)
                  + lambda * mean_i KL(p_i || mean_j p_j)       (Jensen-Shannon)

where p_i are the model's next-token distributions over the answer positions
(teacher-forced on the gold occupation) for the counterfactual variants i of
one SocialCounterfactuals group. lambda = 0 is the CE-only control.

LoRA goes on the vision side only (vision attention + connector); the language
model stays frozen, mirroring the CLIP version (vision q/v LoRA).

Prompts and preprocessing come from projection/models.py, i.e. exactly the
baseline runners, so fine-tuned predictions are comparable to the baseline.
"""

import re
import sys

import torch
import torch.nn.functional as F
from PIL import Image

from projection.common import PROMPT

# vision attention + connector Linear layers, per backend (matched on module names)
LORA_PATTERNS = {
    "qwen2_5_vl": r"(^|\.)visual\.(blocks\.\d+\.attn\.(qkv|proj)|merger\.mlp\.\d+)$",
    "smolvlm2": r"(^|\.)(vision_model\.encoder\.layers\.\d+\.self_attn\.(q_proj|v_proj)"
                r"|connector\.modality_projection\.proj)$",
    "internvl3": r"(^|\.)(vision_model\.encoder\.layers\.\d+\.attn\.(qkv|proj)|mlp1\.\d+)$",
}


def lora_targets(model, backend):
    pattern = re.compile(LORA_PATTERNS[backend])
    names = [n for n, m in model.named_modules()
             if isinstance(m, torch.nn.Linear) and pattern.search(n)]
    if not names:
        raise RuntimeError(f"no vision-side Linear layers matched for {backend}")
    return names


def answer_text(occupation_raw):
    """Gold answer in the style the models use ('Accountant')."""
    s = occupation_raw.strip()
    return s[:1].upper() + s[1:]


def _common_prefix(a, b):
    n = min(len(a), len(b))
    neq = (a[:n] != b[:n]).nonzero()
    return int(neq[0]) if len(neq) else n


class HFExampleBuilder:
    """Qwen2.5-VL / SmolVLM2: baseline chat template + gold answer + end of turn."""

    def __init__(self, backend):
        self.b = backend
        proc = backend.processor
        msgs = [{"role": "user", "content": [{"type": "image"},
                                            {"type": "text", "text": PROMPT}]}]
        probe = "Xyzzyocc"
        full = proc.apply_chat_template(
            msgs + [{"role": "assistant", "content": [{"type": "text", "text": probe}]}],
            tokenize=False)
        if not full.startswith(backend.prompt_text):
            raise RuntimeError("chat template with an answer does not extend the baseline prompt")
        tail = full[len(backend.prompt_text):]
        i = tail.index(probe)
        self.pre, self.post = tail[:i], tail[i + len(probe):].rstrip()

    def __call__(self, image_path, answer, device):
        proc = self.b.processor
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        prompt = proc(text=self.b.prompt_text, images=[image], return_tensors="pt")
        full = proc(text=self.b.prompt_text + self.pre + answer + self.post,
                    images=[image], return_tensors="pt")
        ids = full["input_ids"]
        n_prompt = _common_prefix(ids[0], prompt["input_ids"][0])
        labels = ids.clone()
        labels[:, :n_prompt] = -100
        enc = {k: v.to(device) for k, v in full.items()}
        enc["use_cache"] = False
        return enc, labels.to(device)


class InternExampleBuilder:
    """InternVL3: rebuilds the prompt exactly like model.chat() (one 448 tile)."""

    def __init__(self, backend):
        self.b = backend
        m, tok = backend.model, backend.tokenizer
        conv = sys.modules[type(m).__module__].get_conv_template
        m.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
        t = conv(m.template)
        t.system_message = m.system_message
        t.append_message(t.roles[0], "<image>\n" + PROMPT)
        t.append_message(t.roles[1], None)
        image_tokens = "<img>" + "<IMG_CONTEXT>" * m.num_image_token + "</img>"
        self.prompt = t.get_prompt().replace("<image>", image_tokens, 1)
        self.end = t.sep.strip()          # model.chat stops generation on this token

    def __call__(self, image_path, answer, device):
        tok = self.b.tokenizer
        px = self.b._pixels(image_path)
        prompt_ids = tok(self.prompt, return_tensors="pt").input_ids
        full = tok(self.prompt + answer + self.end, return_tensors="pt")
        n_prompt = _common_prefix(full.input_ids[0], prompt_ids[0])
        labels = full.input_ids.clone()
        labels[:, :n_prompt] = -100
        enc = {"pixel_values": px, "input_ids": full.input_ids.to(device),
               "attention_mask": full.attention_mask.to(device),
               "image_flags": torch.ones(px.shape[0], 1, dtype=torch.long, device=device)}
        return enc, labels.to(device)


def example_builder(name, backend):
    return InternExampleBuilder(backend) if name == "internvl3" else HFExampleBuilder(backend)


def answer_logits(model, enc, labels):
    """Logits predicting each answer token, [L, V] float32, and the targets [L]."""
    logits = model(**enc).logits[0]
    pos = (labels[0] != -100).nonzero().squeeze(-1)
    if len(pos) == 0 or pos[0] == 0:
        raise RuntimeError("no answer tokens found after the prompt")
    return logits[pos - 1].float(), labels[0, pos]


def group_objective(answer_logits_list, targets_list, consistency_weight):
    """Mean CE over the group + lambda * JS between the variants' answer
    distributions (same definition as the CLIP version, per answer position)."""
    ce = torch.stack([F.cross_entropy(lg, tg) for lg, tg in zip(answer_logits_list, targets_list)]).mean()
    lengths = {lg.shape[0] for lg in answer_logits_list}
    if len(lengths) != 1:
        raise RuntimeError("variants of one group must share the answer tokenization")
    logp = torch.stack([F.log_softmax(lg, dim=-1) for lg in answer_logits_list])  # [G, L, V]
    p = logp.exp()
    log_mean = p.mean(0, keepdim=True).clamp_min(torch.finfo(p.dtype).tiny).log()
    js = (p * (logp - log_mean)).sum(-1).mean()
    return ce + consistency_weight * js, ce, js
