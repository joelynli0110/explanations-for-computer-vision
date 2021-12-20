import numpy as np
import torch

def compute_localization_acc(sal_maps,dataset):
    """Compute the localization accuracy for all classes
    For now, we assume only one class "pedestrian". For one image with multiple pedestrians, we count a hit if the top relevant pixel 
    inside either one bbox.
    TODO: Consider multiple bboxes within one image (multiple classes)

    Args:
        sal_maps (numpy.array): Saliency maps with shape (num_images, height, width)
        dataset (image, annotation): Dataset contains images and human annotations

    Returns:
        local_acc (numpy.array): Localization accuracy of all classes (num_classes)
    """
    num_hits, num_misses = 0, 0

    for (sal,data_sample) in zip(sal_maps,dataset):
        img, annot = data_sample
        y,x = get_maximum_point(sal) # maximum point (the pixel with highest relevance) in the saliency map
        
        bboxes = annot['boxes'] # (num_bboxes, 4)
        hit = False
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            # verify the pixel with highest relevance lies inside one bounding box
            if x >= x_min and x <= x_max and y >= y_min and y <= y_max:
                hit = True
                break
            else:
                continue
        if hit:
            num_hits += 1
        else:
            num_misses += 1
            
    local_acc = float(num_hits / (num_hits + num_misses)) # localization acurracy
    return local_acc

def evaluate(sal_maps,dataset,num_classes=1,metric='PointingGameAcc'):

    local_accs = []
    for i in range(num_classes):
      local_acc = compute_localization_acc(sal_maps,dataset)
      local_accs.append(local_acc)
    
    return sum(local_accs) / num_classes

def get_maximum_point(sal):
  return np.unravel_index(torch.argmax(sal.cpu()), sal.shape)



    