import numpy as np
import torch

def count_hit(bbox,sal):
    y, x = get_maximum_point(sal)
    x1, y1, x2, y2 = bbox
    if x >= x1 and x <= x2 and y >= y1 and y <= y2:
        return 1
    else:
        return 0

def get_maximum_point(sal):
    """Get the pixel of the saliency map with the highest score"""
    return np.unravel_index(torch.argmax(sal.cpu()), sal.shape)

def average_in_weight(bbox,sal):
    """Absolute average weight inside the bounding box"""
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    sal_in = sal[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
    return sal_in.sum() / (width * height)

def evaluate_single_img(sal,img,metric='PointingGameAcc'):
    """Evaluate a single image"""
    bboxes = img[1]['boxes']
    val = 0
    for bbox in bboxes:
        if metric == 'PointingGameAcc':
            val += count_hit(bbox,sal)
        if metric == 'avg_weight': 
            val += average_in_weight(bbox,sal)

    return val / len(bboxes)
    
def evaluate(sal_maps,dataset,metric):
    """Evaluate a batch of images"""
    val = 0
    for (sal, data_sample) in zip(sal_maps,dataset):
        val += evaluate_single_img(sal,data_sample,metric)
    return val / len(sal_maps)