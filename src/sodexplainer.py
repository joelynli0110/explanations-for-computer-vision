import torch
import torchvision
import numpy as np

from lime.lime_image import LimeImageExplainer
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.utils import jaccard

class SODExplainer:
    def __init__(
        self, model, num_samples
    ):
        """[summary]
        Args:
            model (torch.nn): Object detector.
            num_samples (int): number of perturbations in LIME explainer
        """
        self.model = model
        self.num_samples = num_samples
          
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
            self.model.eval() # set the module in evaluation mode
            img_probs = []
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
    def get_explanation(self,image_test,data_obj):
        """Args: 
        image_test: image in numpy array form with dtype= double and image shape(_,_,3)
        data_obj : data_obj (img, target): A sample from the PennFudanDataset 
            note : image_test and data_obj should be of same image in different form.

        Returns : An ImageExplanation object (see lime documentation: https://lime-ml.readthedocs.io/en/latest/lime.html#module-lime.explanation) with 		the corresponding explanations
        """
        explainer = LimeImageExplainer(verbose=True)
        explanation = explainer.explain_instance(
            image= image_test,
            classifier_fn=self.get_class_probability(data_obj),
            num_samples=self.num_samples)
        return explanation
 