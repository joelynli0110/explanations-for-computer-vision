import numpy as np
import torch


import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter

from src.utils import *

 # image area

def gkern(klen, nsig):
    """Returns a Gaussian kernel array.
    Convolution with it results in image blurring."""
    # create nxn zeros
    inp = np.zeros((klen, klen))
    # set element at the middle to one, a dirac delta
    inp[klen//2, klen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern.astype('float32'))

def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

class CausalMetric():

    def __init__(self, model, mode, step, substrate_fn):
        r"""Create deletion/insertion metric instance.
        Args:
            model (nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        assert mode in ['del', 'ins']
        self.model = model
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn

    def single_run(self, img_tensor,data_obj, explanation,hw, verbose=0, save_to=None):
        r"""Run metric on one image-saliency pair.
        Args:
            img_tensor (Tensor): normalized image tensor.
            explanation (np.ndarray): saliency map.
            verbose (int): in [0, 1, 2].
                0 - return list of scores.
                1 - also plot final step.
                2 - also plot every step and print 2 top classes.
            save_to (str): directory to save every step plots to.
        Return:
            scores (nd.array): Array containing scores at every step.
        """
        # pred = self.model(img_tensor.cuda())
    
        n_steps = (hw + self.step - 1) // self.step

        if self.mode == 'del':
            title = 'Deletion game'
            ylabel = 'Pixels deleted'
            start = img_tensor.clone()
            finish = self.substrate_fn(img_tensor)
        elif self.mode == 'ins':
            title = 'Insertion game'
            ylabel = 'Pixels inserted'
            start = self.substrate_fn(img_tensor)
            finish = img_tensor.clone()

        scores = np.empty(n_steps + 1)
        # Coordinates of pixels in order of decreasing saliency
        salient_order = np.flip(np.argsort(explanation.cpu().numpy().reshape(-1, hw), axis=1), axis=-1)
        for i in range(n_steps+1):
            probability = get_class_probability(self.model,data_obj,start.cuda())
            # print(probability)
            scores[i] = probability[0][0]
              
            # if verbose == 2:
                # print('{}: {:.3f}'.format(get_class_name(cl[0][0]), float(pr[0][0])))
                # print('{}: {:.3f}'.format(get_class_name(cl[0][1]), float(pr[0][1])))
            
            
            # Render image if verbose, if it's the last step or if save is required.
            if verbose == 2 or (verbose == 1 and i == n_steps) or save_to:
                plt.figure(figsize=(10, 5))
                plt.subplot(121)
                plt.title('{} {:.1f}%, P={:.4f}'.format(ylabel, 100 * i / n_steps, scores[i]))
                plt.axis('off')
                tensor_imshow(start[0])

                plt.subplot(122)
                plt.plot(np.arange(i+1) / n_steps, scores[:i+1])
                plt.xlim(-0.1, 1.1)
                plt.ylim(0, 1.05)
                plt.fill_between(np.arange(i+1) / n_steps, 0, scores[:i+1], alpha=0.4)
                plt.title(title)
                plt.xlabel(ylabel)
                if save_to:
                    plt.savefig(save_to + '/{:03d}.png'.format(i))
                    plt.close()
                else:
                    plt.show()
            if i < n_steps:
                coords = salient_order[:, self.step * i:self.step * (i + 1)]
                start.cpu().numpy().reshape(1, 3, hw)[0, :, coords] = finish.cpu().numpy().reshape(1, 3, hw)[0, :, coords]
        return scores

    def evaluate(self, img_batch, exp_batch, batch_size):
        r"""Efficiently evaluate big batch of images.
        Args:
            img_batch (Tensor): batch of images.
            exp_batch (np.ndarray): batch of explanations.
            batch_size (int): number of images for one small batch.
        Returns:
            scores (nd.array): Array containing scores at every step for every image.
        """
        n_samples = img_batch.shape[0]
        predictions = torch.FloatTensor(n_samples, n_classes)
        assert n_samples % batch_size == 0
        for i in tqdm(range(n_samples // batch_size), desc='Predicting labels'):
            preds = self.model(img_batch[i*batch_size:(i+1)*batch_size].cuda()).cpu()
            predictions[i*batch_size:(i+1)*batch_size] = preds
        top = np.argmax(predictions, -1)
        n_steps = (HW + self.step - 1) // self.step
        scores = np.empty((n_steps + 1, n_samples))
        salient_order = np.flip(np.argsort(exp_batch.reshape(-1, HW), axis=1), axis=-1)
        r = np.arange(n_samples).reshape(n_samples, 1)

        substrate = torch.zeros_like(img_batch)
        for j in tqdm(range(n_samples // batch_size), desc='Substrate'):
            substrate[j*batch_size:(j+1)*batch_size] = self.substrate_fn(img_batch[j*batch_size:(j+1)*batch_size])

        if self.mode == 'del':
            caption = 'Deleting  '
            start = img_batch.clone()
            finish = substrate
        elif self.mode == 'ins':
            caption = 'Inserting '
            start = substrate
            finish = img_batch.clone()

        # While not all pixels are changed
        for i in tqdm(range(n_steps+1), desc=caption + 'pixels'):
            # Iterate over batches
            for j in range(n_samples // batch_size):
                # Compute new scores
                preds = self.model(start[j*batch_size:(j+1)*batch_size].cuda())
                preds = preds.cpu().numpy()[range(batch_size), top[j*batch_size:(j+1)*batch_size]]
                scores[i, j*batch_size:(j+1)*batch_size] = preds
            # Change specified number of most salient pixels to substrate pixels
            coords = salient_order[:, self.step * i:self.step * (i + 1)]
            start.cpu().numpy().reshape(n_samples, 3, HW)[r, :, coords] = finish.cpu().numpy().reshape(n_samples, 3, HW)[r, :, coords]
        print('AUC: {}'.format(auc(scores.mean(1))))
        return scores


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
