"""Fine-tune pretrained OSNet on MOT training identities, separately from YOLOX.

Example: python osnet-train.py --lr 1e-5 --epochs 5
GT person crops are training input; validation identities never update weights.
"""
import argparse
import csv
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from Utils import config
from Utils.data import load_ground_truth, load_image_frames
from Utils.detections import file_digest
from Utils.reid import load_reid_encoder
from Utils.splits import validate_video_splits


def people(sequences, stride):
    """Keep identity keys scoped to each video; MOT IDs repeat across videos."""
    groups, sources = {}, {}
    for sequence in sequences:
        path=Path(config.DATASET_ROOT)/config.DATASET_SPLIT/sequence
        truth=load_ground_truth(path,config.GT_FILE,box_origin=config.BOX_ORIGIN)
        frames=load_image_frames(path)
        sources[sequence]=dict(gt_sha256=file_digest(path/config.GT_FILE),
                               seqinfo_sha256=file_digest(path/'seqinfo.ini'))
        for frame_id in list(frames)[::stride]:
            gt=truth[frame_id]
            for row in np.flatnonzero(gt['valid_mask']):
                box=gt['boxes'][row].copy()
                box[0::2]=box[0::2].clip(0,frames[frame_id]['width'])
                box[1::2]=box[1::2].clip(0,frames[frame_id]['height'])
                if (box[2:]>box[:2]).all():
                    groups.setdefault((sequence,int(gt['ids'][row])),[]).append((frames[frame_id]['image_path'],frame_id,box))
    groups={key:rows for key,rows in groups.items() if len(rows)>=2}
    if len(groups)<2:
        raise ValueError('Need at least two identities with two frames each')
    return groups,sources


def pairs(groups,count,seed):
    """Sample identity-balanced triplets; prefer a different person in the same video."""
    rng=random.Random(seed)
    keys=list(groups)
    by_sequence={sequence:[key for key in keys if key[0]==sequence] for sequence in {k[0] for k in keys}}
    by_frame={}
    for identity,observations in groups.items():
        for item in observations:
            by_frame.setdefault((identity[0],item[1]),[]).append((identity,item))
    output=[]
    for _ in range(count):
        key=rng.choice(keys)
        anchors=groups[key]
        a=rng.choice(anchors)
        candidates=[item for item in anchors if abs(item[1]-a[1])>=10]
        p=rng.choice(candidates or [item for item in anchors if item is not a])
        negatives=[other for other in by_sequence[key[0]] if other!=key]
        same_frame=[item for identity,item in by_frame[(key[0],a[1])] if identity!=key]
        if same_frame:
            n=rng.choice(same_frame)
        else:
            other=rng.choice(negatives or [other for other in keys if other!=key])
            n=rng.choice(groups[other])
        output.append((a,p,n))
    return output


class Triplets(torch.utils.data.Dataset):
    def __init__(self,triplets,preprocess):
        self.triplets,self.preprocess=triplets,preprocess
    def __len__(self):return len(self.triplets)
    def __getitem__(self,index):
        crops=[]
        for path,_,box in self.triplets[index]:
            with Image.open(path) as image:
                image=image.convert('RGB')
                bounds=(int(np.floor(box[0])),int(np.floor(box[1])),
                        int(np.ceil(box[2])),int(np.ceil(box[3])))
                crops.append(self.preprocess(image.crop(bounds)))
        return torch.stack(crops)


def features(network,images,training):
    output=network(images)
    vector=output[1] if training else output
    return F.normalize(vector,dim=1)


def run_epoch(network,loader,device,optimizer=None):
    training=optimizer is not None
    network.train(training)
    total_loss=total_correct=total=0
    with tqdm(loader,desc='Train OSNet' if training else 'Validate OSNet',unit='batch',
              leave=False,disable=not config.SHOW_PROGRESS) as bar:
        for batch in bar:
            batch=batch.to(device,non_blocking=True)
            with torch.set_grad_enabled(training):
                vectors=features(network,batch.flatten(0,1),training).reshape(-1,3,512)
                a,p,n=vectors.unbind(1)
                loss=F.triplet_margin_loss(a,p,n,margin=.3)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(),5.,error_if_nonfinite=True)
                optimizer.step()
            total_loss+=float(loss)*len(batch)
            total_correct+=int(((a*p).sum(1)>(a*n).sum(1)).sum())
            total+=len(batch)
            bar.set_postfix(loss=f'{total_loss/total:.4f}',accuracy=f'{total_correct/total:.3f}',refresh=False)
    return dict(loss=total_loss/total,triplet_accuracy=total_correct/total,pairs=total)


def save(value,path):
    pending=path.with_name(path.name+'.pending')
    torch.save(value,pending)
    pending.replace(path)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lr',type=float,default=1e-5,help='OSNet learning rate; use a fresh output directory for each rate')
    parser.add_argument('--epochs',type=int,default=5,help='Total epoch target; rerun to resume')
    parser.add_argument('--batch-size',type=int,default=8,help='Triplets per optimizer step')
    parser.add_argument('--pairs-per-epoch',type=int,default=2000)
    parser.add_argument('--val-pairs',type=int,default=500)
    parser.add_argument('--frame-stride',type=int,default=5)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--device',default=config.DEVICE)
    args=parser.parse_args(argv)
    if args.lr<=0 or min(args.epochs,args.batch_size,args.pairs_per_epoch,args.val_pairs,args.frame_stride)<1:
        parser.error('Learning rate and counts must be positive')
    if torch.device(args.device).type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu for a small check')
    args.output_dir=args.output_dir or config.PROJECT_ROOT/'outputs'/'osnet_finetune'/f'lr_{args.lr:g}'
    validate_video_splits()
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(config.SEED)
    train_groups,train_sources=people(config.TRAIN_SEQUENCES,args.frame_stride)
    val_groups,val_sources=people(config.VAL_SEQUENCES,args.frame_stride)
    if set(train_groups)&set(val_groups):raise ValueError('Training and validation identities overlap')
    config.FREEZE_ENCODER=True
    config.USE_PRETRAINED_WEIGHTS=True
    config.FEATURE_DIM=512
    encoder=load_reid_encoder(args.device)
    network=encoder.network
    network.loss='triplet'
    network.requires_grad_(True)
    network.classifier.requires_grad_(False)
    optimizer=torch.optim.AdamW((p for p in network.parameters() if p.requires_grad),lr=args.lr,weight_decay=1e-5)
    recipe=dict(version='mot_osnet_triplet_v1',lr=args.lr,batch_size=args.batch_size,
        pairs_per_epoch=args.pairs_per_epoch,val_pairs=args.val_pairs,frame_stride=args.frame_stride,
        seed=config.SEED,train=train_sources,val=val_sources,base_features=encoder.feature_settings)
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    last=out/'last.pt'
    saved=torch.load(last,map_location='cpu',weights_only=True) if last.exists() else None
    if saved and saved['recipe']!=recipe:raise ValueError('Settings or data changed; choose a new output directory')
    if saved:
        best_path=out/'best.pt'
        selected=torch.load(best_path,map_location='cpu',weights_only=True) if best_path.is_file() else None
        current=saved['history'][-1]
        current_key=(current['val_triplet_accuracy'],-current['val_loss'])
        if current_key==tuple(saved['best_key']) and (selected is None or
                tuple(selected['best_key'])<tuple(saved['best_key'])):
            save(saved,best_path)  # Interrupted between latest and best writes.
        elif selected is None:
            raise FileNotFoundError('Missing best.pt for this resumed run')
    if not saved and any(out.iterdir()):raise FileExistsError('Output is not empty; choose another --output-dir')
    history=[];best_key=(-1.,float('-inf'));start=0
    if saved:
        network.load_state_dict(saved['encoder_state_dict'],strict=True)
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        history=saved['history'];best_key=tuple(saved['best_key']);start=saved['epoch']+1
    (out/'settings.json').write_text(json.dumps(recipe,indent=2,default=str)+'\n')
    val_triplets=pairs(val_groups,args.val_pairs,config.SEED+10000)
    val_loader=torch.utils.data.DataLoader(Triplets(val_triplets,encoder.preprocess),batch_size=args.batch_size)
    with tqdm(total=args.epochs+1-start,desc='OSNet epochs',unit='epoch',disable=not config.SHOW_PROGRESS) as overall:
        for epoch in range(start,args.epochs+1):
            train_result=None
            if epoch:
                train_triplets=pairs(train_groups,args.pairs_per_epoch,config.SEED+epoch)
                train_loader=torch.utils.data.DataLoader(Triplets(train_triplets,encoder.preprocess),
                    batch_size=args.batch_size,shuffle=True,
                    generator=torch.Generator().manual_seed(config.SEED+epoch))
                train_result=run_epoch(network,train_loader,args.device,optimizer)
            result=run_epoch(network,val_loader,args.device)
            row=dict(epoch=epoch,lr=args.lr,train_loss=None if train_result is None else train_result['loss'],
                     val_loss=result['loss'],val_triplet_accuracy=result['triplet_accuracy'])
            history.append(row)
            key=(result['triplet_accuracy'],-result['loss'])
            checkpoint=dict(epoch=epoch,recipe=recipe,encoder_state_dict=network.state_dict(),
                            optimizer_state_dict=optimizer.state_dict(),history=history,best_key=max(key,best_key))
            save(checkpoint,last)
            if key>best_key:
                save(checkpoint,out/'best.pt')
                best_key=key
            with (out/'history.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerows(history)
            tqdm.write(f"Epoch {epoch}: validation triplet accuracy {result['triplet_accuracy']:.3f}, loss {result['loss']:.4f}")
            overall.update(1)
    print(f'Best OSNet checkpoint: {out/"best.pt"}')
    return out


if __name__=='__main__':main()
