"""Compare Tracktor/YOLOX failure frames and GT-assisted association on fixed boxes.

This is an analysis oracle: it uses validation ground truth to replace output
identity numbers. It cannot be deployed and is not a held-out test score.
"""
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
import csv
from datetime import datetime
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
import trackeval
from tqdm.auto import tqdm

from Utils import config
from Utils.data import load_image_frames, read_image

DEFAULT_TRACKTOR=config.PROJECT_ROOT/'outputs/tracktor_osnet_cosine/matched_yolox_050_070'
DEFAULT_YOLOX=config.PROJECT_ROOT/'outputs/yolox_osnet/20261124_095002_667396'
SEQUENCES={'MOT17-02-SDP':600,'MOT17-09-SDP':525}


def dataset(run):
    root=run/'evaluation/val/inputs/MOT17'
    for seq in SEQUENCES:
        for path in (root/'gt'/seq/'gt/gt.txt',root/'trackers/DragonTrack/data'/f'{seq}.txt'):
            if not path.is_file():raise FileNotFoundError(f'Complete TrackEval input missing: {path}')
    return trackeval.datasets.MotChallenge2DBox(dict(
        GT_FOLDER=str(root/'gt'),TRACKERS_FOLDER=str(root/'trackers'),
        OUTPUT_FOLDER=str(run/'evaluation/val/MOT17'),TRACKERS_TO_EVAL=['DragonTrack'],
        BENCHMARK='MOT17',SPLIT_TO_EVAL='train',SKIP_SPLIT_FOL=True,
        SEQ_INFO=SEQUENCES,DO_PREPROC=True,PRINT_CONFIG=False,
        CLASSES_TO_EVAL=['pedestrian']))


def load(run):
    source=dataset(run)
    return {seq:source.get_preprocessed_seq_data(
        source.get_raw_seq_data('DragonTrack',seq),'pedestrian') for seq in SEQUENCES}


def assign(scores,threshold=.5,continuation=None):
    """One-to-one IoU matching; optional CLEAR continuity preference."""
    score=np.asarray(scores,dtype=np.float64).copy()
    if not score.size:return np.array([],int),np.array([],int)
    valid=score>=threshold-np.finfo(float).eps
    if continuation is None:
        score=np.where(valid,1000+score,0.)
    else:
        score=np.where(valid,score+1000*continuation,0.)
    rows,cols=linear_sum_assignment(-score)
    keep=score[rows,cols]>np.finfo(float).eps
    return rows[keep],cols[keep]


def frame_errors(data):
    """Reproduce TrackEval CLEAR's per-frame matching and ID-switch definition."""
    prior=np.full(data['num_gt_ids'],np.nan)
    previous_frame=np.full(data['num_gt_ids'],np.nan)
    box_error={}
    rows=[];jitters=[]
    for index in range(data['num_timesteps']):
        gt=data['gt_ids'][index];ids=data['tracker_ids'][index]
        scores=data['similarity_scores'][index]
        continuity=(ids[None,:]==previous_frame[gt[:,None]]) if len(gt) and len(ids) else None
        match_gt,match_tr=assign(scores,continuation=continuity)
        matched_gt=gt[match_gt];matched_ids=ids[match_tr]
        switches=int(np.count_nonzero(~np.isnan(prior[matched_gt]) & (prior[matched_gt]!=matched_ids)))
        for g,t in zip(match_gt,match_tr):
            gt_id=int(gt[g]);gt_box=data['gt_dets'][index][g];tr_box=data['tracker_dets'][index][t]
            error=(tr_box[:2]+tr_box[2:]/2)-(gt_box[:2]+gt_box[2:]/2)
            if gt_id in box_error and box_error[gt_id][0]==index-1:
                jitters.append(float(np.linalg.norm(error-box_error[gt_id][1])))
            box_error[gt_id]=(index,error)
        prior[matched_gt]=matched_ids
        previous_frame[:]=np.nan
        previous_frame[matched_gt]=matched_ids
        rows.append(dict(frame=index+1,gt=len(gt),detections=len(ids),tp=len(match_gt),
                         fp=len(ids)-len(match_tr),fn=len(gt)-len(match_gt),
                         idsw=switches,mean_iou=float(scores[match_gt,match_tr].mean()) if len(match_gt) else None))
    return rows,jitters


def oracle_ids(data):
    """Use GT IDs for matched boxes and fresh IDs for unmatched boxes."""
    changed=deepcopy(data)
    next_id=data['num_gt_ids']
    for index in range(data['num_timesteps']):
        ids=np.empty(len(data['tracker_ids'][index]),dtype=np.int64)
        gt_rows,tr_rows=assign(data['similarity_scores'][index])
        used=set(map(int,tr_rows))
        for gt_row,tr_row in zip(gt_rows,tr_rows):
            ids[tr_row]=data['gt_ids'][index][gt_row]
        for row in range(len(ids)):
            if row not in used:
                ids[row]=next_id;next_id+=1
        changed['tracker_ids'][index]=ids
    changed['num_tracker_ids']=next_id
    return changed


def score(data_by_sequence):
    metrics=(trackeval.metrics.HOTA(),trackeval.metrics.CLEAR({'PRINT_CONFIG':False}),
             trackeval.metrics.Identity({'PRINT_CONFIG':False}))
    results={}
    for metric in metrics:
        per_seq={seq:metric.eval_sequence(data) for seq,data in data_by_sequence.items()}
        combined=metric.combine_sequences(per_seq)
        for key in ('HOTA','DetA','AssA','MOTA','IDF1','CLR_TP','CLR_FP','CLR_FN','IDSW'):
            if key in combined:
                value=combined[key]
                results[key]=round(float(np.mean(value))*100,3) if key in ('HOTA','DetA','AssA') else (
                    round(float(value)*100,3) if key in ('MOTA','IDF1') else int(value))
    return results


def summarize(rows,jitters):
    matched=sum(x['tp'] for x in rows)
    return dict(frames=len(rows),fp=sum(x['fp'] for x in rows),fn=sum(x['fn'] for x in rows),
                idsw=sum(x['idsw'] for x in rows),mean_match_iou=round(
                    sum(x['mean_iou']*x['tp'] for x in rows if x['mean_iou'] is not None)/matched,4)
                    if matched else None,
                jitter_median_px=round(float(np.median(jitters)),2) if jitters else None,
                jitter_p90_px=round(float(np.percentile(jitters,90)),2) if jitters else None,
                jitter_definition='change in detection-center error relative to GT between consecutive matched frames')


def association_only(data_by_sequence):
    """Score identity consistency only on one-to-one GT-matched output boxes."""
    per_sequence={}
    for sequence,data in data_by_sequence.items():
        pairs=Counter();labels=defaultdict(set);last={}
        switches=after_gap=0
        for frame in range(data['num_timesteps']):
            gt_rows,tr_rows=assign(data['similarity_scores'][frame])
            for gt_row,tr_row in zip(gt_rows,tr_rows):
                gt_id=int(data['gt_ids'][frame][gt_row])
                track_id=int(data['tracker_ids'][frame][tr_row])
                pairs[(gt_id,track_id)]+=1
                labels[gt_id].add(track_id)
                if gt_id in last and last[gt_id][1]!=track_id:
                    switches+=1
                    after_gap+=frame-last[gt_id][0]>1
                last[gt_id]=(frame,track_id)
        gt_ids=sorted(labels)
        track_ids=sorted({track for _,track in pairs})
        if gt_ids and track_ids:
            counts=np.zeros((len(gt_ids),len(track_ids)),dtype=np.int64)
            gt_index={v:i for i,v in enumerate(gt_ids)}
            track_index={v:i for i,v in enumerate(track_ids)}
            for (gt_id,track_id),n in pairs.items():
                counts[gt_index[gt_id],track_index[track_id]]=n
            gt_rows,tr_rows=linear_sum_assignment(-counts)
            consistent=int(counts[gt_rows,tr_rows].sum())
        else:
            consistent=0
        per_sequence[sequence]=dict(matched_boxes=sum(pairs.values()),consistent_boxes=consistent,
                                    people=len(gt_ids),single_id_people=sum(len(v)==1 for v in labels.values()),
                                    identity_changes=switches,identity_changes_after_gap=after_gap)
    total={key:sum(item[key] for item in per_sequence.values()) for key in next(iter(per_sequence.values()))}
    total['consistent_pct']=round(100*total['consistent_boxes']/total['matched_boxes'],2) if total['matched_boxes'] else None
    total['single_id_pct']=round(100*total['single_id_people']/total['people'],2) if total['people'] else None
    return dict(overall=total,per_sequence=per_sequence)


def annotate(image,boxes,gt_boxes,color,label):
    canvas=Image.fromarray(image).resize((960,540))
    draw=ImageDraw.Draw(canvas)
    scale_x,scale_y=960/image.shape[1],540/image.shape[0]
    for x,y,w,h in gt_boxes:
        draw.rectangle((x*scale_x,y*scale_y,(x+w)*scale_x,(y+h)*scale_y),outline='lime',width=1)
    for x,y,w,h in boxes:
        draw.rectangle((x*scale_x,y*scale_y,(x+w)*scale_x,(y+h)*scale_y),outline=color,width=2)
    draw.rectangle((0,0,500,24),fill='black')
    draw.text((5,5),label,fill='white')
    return canvas


def previews(out,tracktor,yolox,ranked,count):
    folder=out/'frames';folder.mkdir()
    sequence=config.DATASET_ROOT/config.DATASET_SPLIT/'MOT17-02-SDP'
    frames=load_image_frames(sequence)
    for frame in tqdm(ranked[:count],desc='Diagnostic previews',unit='frame'):
        i=frame-1;image=read_image(frames[frame])
        left=annotate(image,tracktor['tracker_dets'][i],tracktor['gt_dets'][i],
                      'cyan',f'Tracktor frame {frame} | green = GT')
        right=annotate(image,yolox['tracker_dets'][i],yolox['gt_dets'][i],
                       'orange',f'YOLOX frame {frame} | green = GT')
        pair=Image.new('RGB',(1920,540));pair.paste(left,(0,0));pair.paste(right,(960,0))
        pair.save(folder/f'frame_{frame:04d}.jpg',quality=85)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracktor-run',type=Path,default=DEFAULT_TRACKTOR)
    parser.add_argument('--yolox-run',type=Path,default=DEFAULT_YOLOX)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--preview-count',type=int,default=6)
    args=parser.parse_args(argv)
    if args.preview_count<0:parser.error('--preview-count must be nonnegative')
    out=args.output_dir or config.PROJECT_ROOT/'outputs/tracking_diagnostics'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if out.exists():raise FileExistsError(f'Choose a new output directory: {out}')
    runs={'Tracktor':load(args.tracktor_run),'YOLOX':load(args.yolox_run)}
    baseline={name:score(data) for name,data in runs.items()}
    oracle={name:score({seq:oracle_ids(d) for seq,d in data.items()}) for name,data in runs.items()}
    association={name:association_only(data) for name,data in runs.items()}
    rows={};detail={}
    for name,data in runs.items():
        rows[name],jitter=frame_errors(data['MOT17-02-SDP'])
        detail[name]=summarize(rows[name],jitter)
        # Catch changes to TrackEval's CLEAR matching before publishing diagnostics.
        expected=trackeval.metrics.CLEAR({'PRINT_CONFIG':False}).eval_sequence(data['MOT17-02-SDP'])
        if (detail[name]['fp'],detail[name]['fn'],detail[name]['idsw']) != tuple(
                int(expected[key]) for key in ('CLR_FP','CLR_FN','IDSW')):
            raise AssertionError(f'{name}: frame counts disagree with TrackEval')
    ranked=sorted(range(1,601),key=lambda frame:(
        rows['YOLOX'][frame-1]['idsw']*5+rows['YOLOX'][frame-1]['fp']-
        rows['Tracktor'][frame-1]['idsw']*5-rows['Tracktor'][frame-1]['fp']),reverse=True)
    out.mkdir(parents=True)
    with (out/'frames.csv').open('w',newline='') as stream:
        fields=['frame']+[f'{name.lower()}_{key}' for name in runs for key in ('gt','detections','tp','fp','fn','idsw','mean_iou')]
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for i in range(600):
            writer.writerow(dict(frame=i+1,**{f'{name.lower()}_{key}':rows[name][i][key]
                for name in runs for key in ('gt','detections','tp','fp','fn','idsw','mean_iou')}))
    if args.preview_count:previews(out,runs['Tracktor']['MOT17-02-SDP'],runs['YOLOX']['MOT17-02-SDP'],ranked,args.preview_count)
    report=dict(protocol='TrackEval MOT17 preprocessing; IoU 0.5 CLEAR frame matching',
                runs={name:str(path.resolve()) for name,path in [('Tracktor',args.tracktor_run),('YOLOX',args.yolox_run)]},
                baseline=baseline,gt_assisted_fixed_boxes=oracle,association_on_matched_boxes=association,
                mot17_02=detail,worst_frames=ranked[:20])
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Tracking error and fixed-box association diagnostic','',
           'Overall scores combine MOT17-02-SDP and MOT17-09-SDP; frame diagnostics below use MOT17-02-SDP.',
           '',
           'Ground-truth-assisted IDs are an analysis oracle, not a deployable tracker or mathematical maximum.',
           'Tracker-emitted boxes, frame coverage and TrackEval distractor preprocessing are unchanged. Unmatched boxes receive fresh IDs.',
           'CLEAR favors identity continuity when matching boxes to GT, so reassigned IDs can slightly change its TP/FP/FN counts.',
           '', '| Source | Mode | HOTA | DetA | AssA | IDF1 | MOTA | FP | FN | IDSW |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name in runs:
        for label,scores in [('Tracked',baseline[name]),('GT-assisted IDs',oracle[name])]:
            lines.append('| '+' | '.join([name,label]+[str(scores[k]) for k in ('HOTA','DetA','AssA','IDF1','MOTA','CLR_FP','CLR_FN','IDSW')])+' |')
    lines.extend(['','## Association on correctly detected boxes','',
                  'Match output boxes to ground truth independently at IoU 0.5, then find the best one-to-one identity renaming over the full sequence. This removes missed and extra boxes from the identity-consistency denominator.',
                  'Identity changes here use independent per-frame matching, so they can differ slightly from CLEAR IDSW above.',
                  '',
                  '| Source | Matched boxes | Consistent boxes | Consistent | People with one ID | Identity changes | Changes after gap |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |'])
    for name,values in association.items():
        s=values['overall']
        lines.append(f"| {name} | {s['matched_boxes']} | {s['consistent_boxes']} | {s['consistent_pct']}% | {s['single_id_people']}/{s['people']} ({s['single_id_pct']}%) | {s['identity_changes']} | {s['identity_changes_after_gap']} |")
    lines.extend(['','## MOT17-02-SDP frame errors','',
                  '| Source | FP | FN | IDSW | Mean match IoU | Center-error jitter median / p90 (px) |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |'])
    for name,s in detail.items():
        lines.append(f"| {name} | {s['fp']} | {s['fn']} | {s['idsw']} | {s['mean_match_iou']} | {s['jitter_median_px']} / {s['jitter_p90_px']} |")
    lines.extend(['','Jitter is the change in box-center error relative to GT between consecutive matched frames.','',
                  '## Frames with largest YOLOX excess errors','',
                  '| Frame | Tracktor FP | YOLOX FP | Tracktor IDSW | YOLOX IDSW | Preview |',
                  '| ---: | ---: | ---: | ---: | ---: | --- |'])
    for frame in ranked[:10]:
        a,b=rows['Tracktor'][frame-1],rows['YOLOX'][frame-1]
        preview=f'[image](frames/frame_{frame:04d}.jpg)' if frame in ranked[:args.preview_count] else ''
        lines.append(f"| {frame} | {a['fp']} | {b['fp']} | {a['idsw']} | {b['idsw']} | {preview} |")
    lines.extend(['','[All 600 frame counts](frames.csv); figures show ground truth (green), Tracktor (cyan) and YOLOX (orange) boxes on the same original image.',''])
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(f'Diagnostic report: {out/"report.md"}')
    return out


if __name__=='__main__':main()
