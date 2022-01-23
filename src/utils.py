# https://github.com/amdegroot/ssd.pytorch/blob/master/layers/box_utils.py#L48
#https://github.com/eclique/RISE/blob/d91ea006d4bb9b7990347fe97086bdc0f5c1fe10/utils.py#L1

import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data.sampler import Sampler
from torchvision import transforms, datasets
from PIL import Image
import torch

# Dummy class to store arguments
class Dummy():
    pass


# Function that opens image from disk, normalizes it and converts to tensor
# read_tensor = transforms.Compose([
#     lambda x: Image.open(x),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                           std=[0.229, 0.224, 0.225]),
#     lambda x: torch.unsqueeze(x, 0)
# ])


# Plots image from tensor
def tensor_imshow(inp, title=None, **kwargs):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # Mean and std for ImageNet
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, **kwargs)
    if title is not None:
        plt.title(title)


# Given label number returns class name
# def get_class_name(c):
#     labels = np.loadtxt('synset_words.txt', str, delimiter='\t')
#     return ' '.join(labels[c].split(',')[0].split()[1:])


# Image preprocessing function
# preprocess = transforms.Compose([
#                 transforms.Resize((224, 224)),
#                 transforms.ToTensor(),
#                 # Normalization for ImageNet
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225]),
#             ])


# Sampler for pytorch loader. Given range r loader will only
# return dataset[r] instead of whole dataset.
# class RangeSampler(Sampler):
#     def __init__(self, r):
#         self.r = r

#     def __iter__(self):
#         return iter(self.r)

#     def __len__(self):
#         return len(self.r)




def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A âˆ© B / A âˆª B = A âˆ© B / (area(A) + area(B) - A âˆ© B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def get_class_probability(model,data_obj,img_as_arrays):
        """ The Method takes one image from the dataset and returns the class score as the probability 
            indicating how likely we detect the most confident object in this image.
        Args:
            data_obj (img, target): A sample from the PennFudanDataset
        Returns:
            get_probabilities (function): A function takes a numpy array and outputs prediction probabilities
        """
        _, target = data_obj # ground truth
        
        
        img_probs = []
        with torch.no_grad():
            for img in img_as_arrays:
                # img = torch.Tensor(img_as_array)
                image = img.cpu().permute(1,2,0).detach().numpy()
                image = torch.Tensor(image)
                boxes = model(image.unsqueeze(0).permute(0,3,1,2).cuda())[0]['boxes']
                scores = model(image.unsqueeze(0).permute(0,3,1,2).cuda())[0]['scores']

                if boxes.size() == 0: 
                    # if there's no object detected
                    prob = 0
                    print("No object detected!")
                else:
                    ious = jaccard(target['boxes'].cuda(),boxes) # ious with shape (num_objs, num_boxes)
                    ious = ious[ious > 0.1].unsqueeze(1) 
                    if ious.size(0) == 0: # No score above the threshold
                        prob = 0
                        print("No score above the threshold!")
                    else:
                        # print(ious)
                        obj_idx, box_idx = np.unravel_index(torch.argmax(ious.cpu()), ious.shape) # retrieve argmax-indices in 2d
                        prob = scores[box_idx] 
                probabilities = [prob, 1 - prob]
                img_probs.append(np.array(probabilities))   
        return np.array(img_probs)

     