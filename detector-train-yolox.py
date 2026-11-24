"""Fine-tune the CrowdHuman + MOT20 YOLOX-X on configured MOT train videos.

Run python detector-train-yolox.py. Best checkpoint uses validation detection F1.
See Documentation/Implementation/31-yolox-training.md.
"""
import argparse
import csv
import fcntl
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from Utils import config
from Utils.detector_training import build_datasets
from Utils.detector_selection import select_operating_point, selection_key
from Utils.yolox import YOLOXDetector
from Utils.yolox_training import YOLOXCollator
from evaluate_detector import COUNT_FIELDS, metrics, score_frame
VERSION='mot_yolox_finetune_v1'

def save(value,path):
    pending=path.with_name(path.name+'.pending')
    torch.save(value,pending);pending.replace(path)

def write(value,path):
    pending=path.with_name(path.name+'.pending')
    pending.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');pending.replace(path)

def state():
    return dict(python=random.getstate(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def epoch(model,loader,optimizer,scaler,args,overall):
    model.train();optimizer.zero_grad(set_to_none=True);total=0.;seen=0
    with tqdm(loader,desc='Train YOLOX',unit='batch',position=1,leave=False,disable=not config.SHOW_PROGRESS) as bar:
        for i,(batch,records) in enumerate(bar):
            group_start=(i//args.accumulation)*args.accumulation
            group_images=min(args.accumulation*args.batch_size,len(loader.dataset)-group_start*args.batch_size)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=torch.device(args.device).type=='cuda'):
                losses=model(**model.data_preprocessor(batch,training=True),mode='loss')
                loss=sum(v.mean() for k,v in losses.items() if 'loss' in k)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite YOLOX loss')
            scaler.scale(loss*len(records)/group_images).backward()
            if (i+1)%args.accumulation==0 or i+1==len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
            total+=loss.item()*len(records);seen+=len(records)
            bar.set_postfix(loss=f'{total/seen:.3f}',refresh=False);overall.update(1)
    return total/seen

@torch.no_grad()
def validate(model,loader,args,overall):
    model.eval();counts={};scores=sorted(set(args.scores+[args.score]))
    with tqdm(loader,desc='Validate YOLOX',unit='frame',position=1,leave=False,disable=not config.SHOW_PROGRESS) as bar:
        for batch,records in bar:
            processed=model.data_preprocessor(batch,training=False)
            outputs=model(**processed,mode='predict')
            for result,(gt,frame) in zip(outputs,records,strict=True):
                boxes=result.pred_instances.bboxes.float().cpu().numpy().copy()
                confidence=result.pred_instances.scores.float().cpu().numpy()
                if not np.isfinite(boxes).all() or not np.isfinite(confidence).all():raise FloatingPointError('Nonfinite YOLOX boxes')
                boxes[:,0::2]=boxes[:,0::2].clip(0,frame['width']);boxes[:,1::2]=boxes[:,1::2].clip(0,frame['height'])
                valid=(boxes[:,2:]-boxes[:,:2]>1e-5).all(axis=1)
                for iou in (.5,.75):
                    for score in scores:
                        result_counts=score_frame(boxes[valid & (confidence>=score)],gt,iou)
                        total=counts.setdefault((frame['sequence'],iou,score),dict.fromkeys(COUNT_FIELDS,0))
                        for field,value in result_counts.items():total[field]+=value
            overall.update(1)
    rows=[dict(sequence=seq,iou=iou,score=score,**metrics(c)) for (seq,iou,score),c in counts.items()]
    for iou in (.5,.75):
        for score in scores:
            total={f:sum(c[f] for (_,i,s),c in counts.items() if (i,s)==(iou,score)) for f in COUNT_FIELDS}
            rows.append(dict(sequence='COMBINED',iou=iou,score=score,**metrics(total)))
    chosen=select_operating_point(rows,score=args.score,sweep=True,min_precision=args.min_precision,min_recall=args.min_recall)
    if not chosen['gt']:raise ValueError('Validation has no GT')
    return chosen,rows

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=config.PROJECT_ROOT/'outputs/yolox_x_mot17_finetune_v1')
    p.add_argument('--epochs',type=int,default=5,help='Total epoch target; rerun to resume')
    p.add_argument('--width',type=int,default=960);p.add_argument('--height',type=int,default=544)
    p.add_argument('--batch-size',type=int,default=1);p.add_argument('--accumulation',type=int,default=4)
    p.add_argument('--lr',type=float,default=5e-5);p.add_argument('--backbone-lr',type=float,default=5e-6)
    p.add_argument('--weight-decay',type=float,default=1e-5)
    p.add_argument('--frame-stride',type=int,default=1);p.add_argument('--train-fraction',type=float,default=1.)
    p.add_argument('--score',type=float,default=.25);p.add_argument('--scores',type=float,nargs='+',default=[.1,.15,.2,.25,.3,.4,.5,.6,.7])
    p.add_argument('--min-precision',type=float,default=.7);p.add_argument('--min-recall',type=float,default=.55)
    p.add_argument('--max-frames',type=int,help='Small check: cap each train/validation video')
    p.add_argument('--device',default=config.DEVICE)
    args=p.parse_args(argv)
    if min(args.epochs,args.width,args.height,args.batch_size,args.accumulation,args.frame_stride)<1 or not 0<args.train_fraction<=1 or any(not 0<s<1 for s in args.scores+[args.score]):p.error('Invalid training settings')
    if not 0<=args.min_precision<=1 or not 0<=args.min_recall<=1 or args.lr<=0 or args.backbone_lr<=0 or args.weight_decay<0:p.error('Invalid learning rate or score floor')
    if args.max_frames is not None and args.max_frames<1:p.error('max-frames must be positive')
    if torch.device(args.device).type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with (args.output_dir/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        random.seed(config.SEED);torch.manual_seed(config.SEED)
        train_data,val_data=build_datasets(args.frame_stride,args.max_frames,train_fraction=args.train_fraction)
        model_args=argparse.Namespace(width=args.width,height=args.height,min_score=.01,nms_iou=.7,
            checkpoint=None,weights_dir=config.PROJECT_ROOT/'outputs/yolox_weights',device=args.device)
        detector=YOLOXDetector(model_args,trainable=True);model=detector.model
        recipe=dict(version=VERSION,initial_checkpoint=detector.recipe['checkpoint_sha256'],
            settings={k:v for k,v in vars(args).items() if k not in ('output_dir','epochs','device')},
            seed=config.SEED,precision='cuda_amp_bfloat16' if torch.device(args.device).type=='cuda' else 'float32',
            train=train_data.sources,val=val_data.sources,box_origin=config.BOX_ORIGIN)
        out=args.output_dir;last=out/'last.pt';best=out/'best.pt'
        if any(out.iterdir()) and not last.exists() and {x.name for x in out.iterdir()}!={'.lock'}:
            raise FileExistsError('Nonempty output without last.pt; choose another --output-dir')
        saved=torch.load(last,map_location='cpu',weights_only=True) if last.exists() else None
        if saved and saved['recipe']!=recipe:raise ValueError('Settings/data changed; choose a new output directory')
        if saved and not best.exists():raise FileNotFoundError('Best checkpoint missing')
        groups=[dict(params=[p for n,p in model.named_parameters() if n.startswith('backbone.') and p.requires_grad],lr=args.backbone_lr),
                dict(params=[p for n,p in model.named_parameters() if not n.startswith('backbone.') and p.requires_grad],lr=args.lr)]
        optimizer=torch.optim.AdamW(groups,weight_decay=args.weight_decay)
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        history=[];best_selection=dict(eligible=False,f1=-1.);start=0
        if saved:
            model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
            history=saved['history'];best_selection=saved['best_selection'];start=saved['epoch']+1
            random.setstate(saved['rng']['python']);torch.set_rng_state(saved['rng']['torch'])
            if torch.cuda.is_available() and saved['rng']['cuda']:torch.cuda.set_rng_state_all(saved['rng']['cuda'])
        write(recipe,out/'settings.json')
        train_loader=DataLoader(train_data,batch_size=args.batch_size,shuffle=True,num_workers=0,collate_fn=YOLOXCollator(args.width,args.height,augment=True))
        val_loader=DataLoader(val_data,batch_size=1,shuffle=False,num_workers=0,collate_fn=YOLOXCollator(args.width,args.height))
        steps=(len(val_loader) if start==0 else 0)+max(0,args.epochs-max(1,start)+1)*(len(train_loader)+len(val_loader))
        with tqdm(total=steps,desc='YOLOX training overall',unit='batch',disable=not config.SHOW_PROGRESS) as overall:
            for number in range(start,args.epochs+1):
                started=time.perf_counter()
                loss=epoch(model,train_loader,optimizer,scaler,args,overall) if number else None
                chosen,rows=validate(model,val_loader,args,overall)
                improved=selection_key(chosen)>selection_key(best_selection)
                if improved:best_selection={k:chosen[k] for k in ('score','precision','recall','f1','eligible')}
                history.append(dict(epoch=number,train_loss=loss,selected_score=chosen['score'],
                    precision=chosen['precision'],recall=chosen['recall'],f1=chosen['f1'],eligible=chosen['eligible'],seconds=time.perf_counter()-started))
                checkpoint=dict(recipe=recipe,epoch=number,model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                    best_selection=best_selection,history=history,rng=state())
                if improved:
                    save(checkpoint,best)
                    save(dict(state_dict=checkpoint['model'],meta=dict(dataset_meta=dict(classes=('person',)),
                        yolox_training=dict(version=VERSION,epoch=number,recipe=recipe,selection=best_selection))),out/'best.pth')
                save(checkpoint,last)
                write(dict(epoch=number,selection_score=chosen['score'],results=rows),out/f'validation_epoch_{number:03d}.json')
                with (out/'history.csv.pending').open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
                (out/'history.csv.pending').replace(out/'history.csv')
                p_text='n/a' if chosen['precision'] is None else f"{chosen['precision']:.4f}"
                tqdm.write(f"Epoch {number}: F1={chosen['f1']:.4f} P={p_text} R={chosen['recall']:.4f} score={chosen['score']}")
        return out

if __name__=='__main__':main()
