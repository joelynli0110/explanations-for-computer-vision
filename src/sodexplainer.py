import os
import torch
import torchvision
import numpy as np

from lime.lime_image import LimeImageExplainer
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.utils import jaccard

from skimage.transform import resize
from tqdm import tqdm

class SODExplainer:
    def __init__(
        self, 
        model,
    ):
        """[summary]
        Args:
            model (torch.nn): Object detector.

        """
        self.model = model
        self.model.eval()
          
    def get_class_probability(self, data_obj):
        """ The Method takes one image from the dataset and returns the class score as the probability 
            indicating how likely we detect the most confident object in this image.
        Args:
            data_obj (img, target): A sample from the PennFudanDataset
        Returns:
            get_probabilities (function): A function takes a numpy array and outputs prediction probabilities
        """
        _, target = data_obj # ground truth
        
        def get_probabilities(img_as_arrays):
            """
            Args:
                img_as_arrays (numpy.array): A numpy array of images with shape (num_images, channel, height, width)
                                             where each image is ndarray of shape (channel, height, width)
            Returns:
                img_probs (numpy.array): Probabilities of binary classification for each image, with shape (num_images, 2)
            """
            # self.model.eval() # set the module in evaluation mode
            
            img_probs = []
            with torch.no_grad():
                for img_as_array in img_as_arrays:
                    img = torch.Tensor(img_as_array)
                    boxes = self.model(img.unsqueeze(0).permute(0,3,1,2))[0]['boxes']
                    scores = self.model(img.unsqueeze(0).permute(0,3,1,2))[0]['scores']
    
                    if boxes.size() == 0: 
                        # if there's no object detected
                        prob = 0
                        print("No object detected!")
                    else:
                        ious = jaccard(target['boxes'],boxes) # ious with shape (num_objs, num_boxes)
                        ious = ious[ious > 0.4].unsqueeze(1) 
                        if ious.size() == 0: # No score above the threshold
                            prob = 0
                            print("No score above the threshold!")
                        else:
                            print(ious)
                            obj_idx, box_idx = np.unravel_index(torch.argmax(ious), ious.shape) # retrieve argmax-indices in 2d
                            prob = scores[box_idx] 
                    probabilities = [prob, 1 - prob]
                    img_probs.append(np.array(probabilities))   
            return np.array(img_probs)

        return get_probabilities
    
    #get_explnation for the image
    def get_lime_explanation(self,image_test,data_obj, num_samples, num_features=100000):
        """Args: 
        image_test: image in numpy array form with dtype= double and image shape(_,_,3)
        data_obj : data_obj (img, target): A sample from the PennFudanDataset 
            note : image_test and data_obj should be of same image in different form.
        num_samples (int): number of perturbations in LIME explainer
        num_features (int): number of features in LIME explainer, default = 100000

        Returns : An ImageExplanation object (see lime documentation: https://lime-ml.readthedocs.io/en/latest/lime.html#module-lime.explanation) with 		the corresponding explanations
        """
        explainer = LimeImageExplainer(verbose=True)
        explanation = explainer.explain_instance(
            image= image_test,
            classifier_fn=self.get_class_probability(data_obj),
            num_samples=num_samples,
            num_features=num_features)
        return explanation

    def get_rise_explanation(self,image_test, N, s, p1):
        """Args: 
        image_test: image in numpy array form with dtype= double and image shape(_,_,3)
        N: Number of masks that are generated
        s: Width and height of binary mask
        p1: Const. factor for saliency values
        
        Returns :
        sal: Resulting saliency image after RISE explanation
        """
        input_size = image_test.shape[0:2]
        cell_size = np.ceil(np.array(input_size) / s)
        up_size = (s + 1) * cell_size
        
        # Binary masks as grid
        # One grid per mask with size s x s
        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')

        # Masks have the same size as the input image
        masks = np.empty((N, *input_size))
        
        for i in tqdm(range(N), desc='Generating masks'):
            # Random shifts for later cropping
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            # Resize and crop binary grid masks
            # Resulting masks are smooth
            masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                    anti_aliasing=False)[x:x + input_size[0], y:y + input_size[1]]

        masks = masks.reshape(-1, *input_size, 1)
        preds = []
        masked = image_test * masks
        sal = np.zeros_like(masked[0])

        for i in tqdm(range(0, N), desc='Explaining'):
            pred = self.model(masked[i].permute(2, 0, 1).view(1, 3, input_size[0], input_size[1]).float())
      
            labels = pred[0]["labels"].detach().numpy()
            scores = pred[0]["scores"].detach().numpy()
            
            for j, label in enumerate(labels):
                # Label pedestrian
                # TODO (sherif): add label as input to function for case with mutiple labels (E.g. coco)
                if label == 1:
                    preds.append(scores[j])
                    # Weighted sum of masks and all scores
                    sal += scores[j] * masked[i].numpy()
                    
        sal = sal / len(preds) / p1
        
        # sum saliency over 3 RGB channels
        return sal.sum(axis=2), preds, masked
