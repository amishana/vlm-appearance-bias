#!/usr/bin/env python3
"""CIFAR-10 zero-shot retention proxy, never used for training or selection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request
import numpy as np
from PIL import Image
import torch
from peft import PeftModel
from cf_model import load_base, read_config, choose_device, seed_everything, dump_json, file_hash

URL='https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz'
MD5='c32a1d4ab5d03f1284b67883e8d87530'


def parse_labels(raw):
    labels = [line.strip() for line in raw.decode().splitlines() if line.strip()]
    if len(labels) != 10 or len(set(labels)) != 10:
        raise ValueError('CIFAR-10 requires exactly ten unique nonempty labels')
    return labels


def test_data(root):
    root.mkdir(parents=True,exist_ok=True)
    path=root/'cifar-10-binary.tar.gz'
    if not path.exists():
        tmp=path.with_suffix('.tmp')
        with urllib.request.urlopen(URL,timeout=90) as r, tmp.open('wb') as f:
            while data:=r.read(1024*1024):f.write(data)
        tmp.replace(path)
    with path.open('rb') as f:actual=hashlib.file_digest(f,'md5').hexdigest()
    if actual!=MD5:raise ValueError('Official CIFAR-10 archive checksum mismatch')
    with tarfile.open(path) as archive:
        raw=archive.extractfile('cifar-10-batches-bin/test_batch.bin').read()
        labels=parse_labels(archive.extractfile('cifar-10-batches-bin/batches.meta.txt').read())
    records=np.frombuffer(raw,dtype=np.uint8).reshape(10000,3073)
    return records,labels,path


def evaluate(cfg,adapter,records,labels,output):
    seed_everything(42);device=choose_device('auto'); model,processor=load_base(cfg)
    if adapter:model=PeftModel.from_pretrained(model,adapter)
    model.to(device).eval()
    prompts=['a photo of a '+name+'.' for name in labels]
    tokens=processor(text=prompts,padding=True,return_tensors='pt').to(device)
    with torch.no_grad():
        texts=model.get_text_features(**tokens);texts=texts/texts.norm(dim=-1,keepdim=True)
    correct=0
    with output.open('x',newline='') as f:
        writer=csv.writer(f);writer.writerow(['test_index','label_id','predicted_label_id'])
        with torch.no_grad():
            for start in range(0,len(records),64):
                batch=records[start:start+64]
                images=[Image.fromarray(row[1:].reshape(3,32,32).transpose(1,2,0)) for row in batch]
                pixels=processor(images=images,return_tensors='pt')['pixel_values'].to(device)
                features=model.get_image_features(pixel_values=pixels);features=features/features.norm(dim=-1,keepdim=True)
                predicted=(features@texts.T).argmax(dim=-1).cpu().tolist()
                for idx,(row,pred) in enumerate(zip(batch,predicted),start):
                    writer.writerow([idx,int(row[0]),pred]);correct+=pred==int(row[0])
    return {'accuracy':correct/len(records),'correct':correct,'images':len(records),'prediction_sha256':file_hash(output)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--adapter',required=True,type=Path);p.add_argument('--output-dir',required=True,type=Path);args=p.parse_args()
    cfg=read_config(args.config);args.output_dir.mkdir(parents=True,exist_ok=True)
    records,labels,archive=test_data(Path(__file__).resolve().parent/'data/cifar10')
    baseline=evaluate(cfg,None,records,labels,args.output_dir/'baseline.csv')
    adapted=evaluate(cfg,args.adapter,records,labels,args.output_dir/'adapted.csv')
    dump_json(args.output_dir/'comparison.json',{'dataset':'CIFAR-10 official test split','source':URL,'archive_sha256':file_hash(archive),
       'labels':labels,'prompt_template':'a photo of a {class}.','baseline':baseline,'adapted':adapted,
       'accuracy_delta':adapted['accuracy']-baseline['accuracy'],'limitation':'One external image-classification proxy; not a comprehensive general-capability assessment.'})
