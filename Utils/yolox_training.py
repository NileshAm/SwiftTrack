"""MOT GT batches for YOLOX; preserve the original GT for validation metrics."""
import cv2
import numpy as np
import torch
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
from Utils.detector_training import clip_boxes

class YOLOXCollator:
    def __init__(self, width, height, augment=False):
        self.width,self.height,self.augment=width,height,augment

    def __call__(self, samples):
        images, data_samples, records=[],[],[]
        for rgb, gt, frame in samples:
            h,w=rgb.shape[:2]
            valid=clip_boxes(gt['boxes'][gt['valid_mask']],w,h)
            ignored=clip_boxes(gt['boxes'][gt['ignored_mask']],w,h)
            if self.augment and torch.rand(()).item()<.5:
                rgb=rgb[:,::-1].copy()
                for boxes in (valid,ignored):
                    boxes[:,[0,2]]=w-boxes[:,[2,0]]
            ratio=min(self.width/w,self.height/h)
            new_w,new_h=round(w*ratio),round(h*ratio)
            resized=cv2.resize(rgb,(new_w,new_h),interpolation=cv2.INTER_LINEAR)
            # The published detector consumes BGR and pads with 114.
            padded=np.full((int(np.ceil(new_h/32)*32),int(np.ceil(new_w/32)*32),3),114,dtype=np.uint8)
            padded[:new_h,:new_w]=resized[...,::-1]
            images.append(torch.from_numpy(padded.transpose(2,0,1).copy()))
            scale=np.array([new_w/w,new_h/h,new_w/w,new_h/h],dtype=np.float32)
            sample=DetDataSample(metainfo=dict(img_id=frame['frame_id'],img_path=str(frame['image_path']),
                ori_shape=(h,w),img_shape=(new_h,new_w),pad_shape=padded.shape[:2],scale_factor=(new_w/w,new_h/h)))
            sample.gt_instances=InstanceData(bboxes=torch.as_tensor(valid*scale,dtype=torch.float32),labels=torch.zeros(len(valid),dtype=torch.long))
            sample.ignored_instances=InstanceData(bboxes=torch.as_tensor(ignored*scale,dtype=torch.float32),labels=torch.zeros(len(ignored),dtype=torch.long))
            data_samples.append(sample);records.append((gt,frame))
        return dict(inputs=images,data_samples=data_samples),records
