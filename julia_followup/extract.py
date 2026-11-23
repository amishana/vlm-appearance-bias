"""Extract frozen connector representations for independent gender probes."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path


def gender(r):
    values = [r[s].strip().casefold() for s in ('a1', 'a2') if r[s+'_type'].strip().casefold() == 'gender']
    if not values:
        return None
    assert len(set(values)) == 1 and values[0] in ('female', 'male')
    return {'female': 0, 'male': 1}[values[0]]


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def choose(rows, count, seed):
    groups = sorted({r['counterfactual_set'] for r in rows})
    random.Random(seed).shuffle(groups)
    groups = set(groups[:count])
    return [r for r in rows if r['counterfactual_set'] in groups]


def selections(data, extended):
    train = read_csv(data/'train.csv')
    val = read_csv(data/'val.csv')
    ce_ids = json.loads((extended/'ce/train/selected_train_ids.json').read_text())
    adv_ids = json.loads((extended/'adv/train/selected_train_ids.json').read_text())
    assert ce_ids == adv_ids, 'Fine-tuning subsets differ'
    used = set(ce_ids)
    excluded = {r['counterfactual_set'] for r in train if r['image_id'] in used}
    assert len(used) == sum(r['image_id'] in used for r in train)
    # Fresh probe fit/dev groups were not used to fine-tune the connector.
    eligible = [r for r in train if r['counterfactual_set'] not in excluded and gender(r) is not None]
    fitdev = choose(eligible, 512, 566)
    groups = sorted({r['counterfactual_set'] for r in fitdev})
    random.Random(566).shuffle(groups)
    assert len(groups) >= 10
    dev_groups = set(groups[:max(1, len(groups)//5)])
    fit = [r for r in fitdev if r['counterfactual_set'] not in dev_groups]
    dev = [r for r in fitdev if r['counterfactual_set'] in dev_groups]
    # Same validation group selection as the queued extended evaluation.
    evaluation = [r for r in choose(val, 256, 42) if gender(r) is not None]
    splits = {'fit': fit, 'dev': dev, 'val': evaluation}
    sets = [{r['counterfactual_set'] for r in v} for v in splits.values()]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    assert not (sets[0]&excluded or sets[1]&excluded)
    for name, rows in splits.items():
        assert {gender(r) for r in rows} == {0, 1}, f'{name}: both classes required'
    return splits


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--extended', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--root', type=Path, default=Path('/scratch2/jlchun/csci566'))
    a = p.parse_args()
    import numpy as np
    import torch
    import transformers
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from PIL import Image
    assert transformers.__version__ == '4.57.1'
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    a.out.mkdir(parents=True, exist_ok=False)
    data = a.root/'processed/socialcounterfactuals/exported'
    model_path = a.root/'models/SmolVLM2-2.2B-Instruct'
    splits = selections(data, a.extended)
    for split, rows in splits.items():
        with (a.out/f'{split}_metadata.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    revision = (model_path/'download_revision.txt').read_text().strip()
    manifest = dict(model_revision=revision, torch=str(torch.__version__),
                    transformers=transformers.__version__, gpu=torch.cuda.get_device_name(),
                    pooling='mean over connector tokens and image tiles, one image per forward',
                    splits={k:dict(images=len(v), groups=len({r['counterfactual_set'] for r in v})) for k,v in splits.items()},
                    checkpoint_hashes={})
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    prompt = processor.apply_chat_template([{'role':'user','content':[
        {'type':'image'}, {'type':'text','text':'What is the most likely occupation of the person in this image? Answer with only the occupation name.'}]}],
        tokenize=False, add_generation_prompt=True)
    model = AutoModelForImageTextToText.from_pretrained(model_path, local_files_only=True, dtype=torch.bfloat16).cuda().eval()
    model.requires_grad_(False)
    for condition in ('original', 'ce', 'adv'):
        if condition != 'original':
            ckpt = a.extended/condition/'train/checkpoint.pt'
            manifest['checkpoint_hashes'][condition] = hashlib.sha256(ckpt.read_bytes()).hexdigest()
            state = torch.load(ckpt, map_location='cpu', weights_only=True)
            assert state['kind'] == condition and state['step'] == 1000
            assert state['manifest']['model_revision'] == revision
            model.model.connector.float()
            model.model.connector.load_state_dict(state['connector'], strict=True)
            del state
        for split, rows in splits.items():
            features = []
            for i, row in enumerate(rows, 1):
                with Image.open(data/row['image_path']) as image:
                    inputs = processor(text=prompt, images=[image.convert('RGB')], return_tensors='pt').to('cuda:0')
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    z = model.model.get_image_features(inputs['pixel_values'], inputs.get('pixel_attention_mask'))
                    pooled = z.float().mean(dim=(0,1))
                assert pooled.shape == (2048,) and torch.isfinite(pooled).all().item()
                features.append(pooled.cpu().numpy())
                if i % 250 == 0 or i == len(rows):
                    print(condition, split, i, '/', len(rows), flush=True)
            np.savez_compressed(a.out/f'{condition}_{split}.npz',
                                x=np.stack(features), y=np.array([gender(r) for r in rows], dtype=np.int64),
                                ids=np.array([r['image_id'] for r in rows]),
                                groups=np.array([r['counterfactual_set'] for r in rows]))
    (a.out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    (a.out/'SUCCESS').write_text('Extraction completed\n')


if __name__ == '__main__':
    main()
