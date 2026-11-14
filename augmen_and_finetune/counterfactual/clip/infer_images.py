#!/usr/bin/env python3
"""Run the saved occupation LoRA on local images without benchmark manifests."""
import argparse
import json
from pathlib import Path
import torch
from peft import PeftModel
from PIL import Image
from transformers import CLIPProcessor
from cf_model import (PROMPT, choose_device, file_hash, load_base, occupation_logits,
                      seed_everything, text_features)


def infer(adapter, paths, output, device='auto'):
    adapter=Path(adapter).resolve()
    metadata=json.loads((adapter/'metadata.json').read_text())
    labels=json.loads((adapter/'labels.json').read_text())
    revision=metadata.get('resolved_base_commit') or metadata['requested_revision']
    cfg={'_project_root':str(adapter), 'model':{'backend':'clip','name':metadata['base_model'],
         'revision':revision,'local_files_only':False}}
    if output.exists():raise FileExistsError(f'Output already exists: {output}')
    seed_everything(42);device=choose_device(device)
    model,_=load_base(cfg)
    model=PeftModel.from_pretrained(model,adapter).to(device).eval()
    processor=CLIPProcessor.from_pretrained(adapter/'processor',local_files_only=True)
    classes=text_features(model,processor,labels,device)
    records=[]
    with torch.no_grad():
        for path in paths:
            path=Path(path).resolve()
            with Image.open(path) as image:
                pixels=processor(images=[image.convert('RGB')],return_tensors='pt')['pixel_values'].to(device)
            scores=occupation_logits(model,pixels,classes)[0]
            probabilities=scores.softmax(dim=-1).cpu().tolist()
            predicted=int(scores.argmax())
            records.append({'image':str(path),'image_sha256':file_hash(path),
                 'predicted_occupation':labels[predicted],'confidence':probabilities[predicted],
                 'probabilities':probabilities})
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as handle:
        json.dump({'base_model':metadata['base_model'],'revision':revision,'prompt':PROMPT,
                   'labels':labels,'predictions':records,
                   'scope':'Closed-vocabulary occupation predictions; scores are not calibrated probabilities.'},handle,indent=2)
    return records


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter',type=Path,required=True)
    parser.add_argument('--image',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='auto',choices=['auto','cpu','mps','cuda'])
    args=parser.parse_args()
    records=infer(args.adapter,args.image,args.output,args.device)
    print(json.dumps([{'image':r['image'],'predicted_occupation':r['predicted_occupation']} for r in records]))
