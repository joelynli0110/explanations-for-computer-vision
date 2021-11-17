from numpy.lib.npyio import load
import torch
import torchvision
import numpy as np

from torchvision.models.detection.faster_rcnn import FasterRCNN,FastRCNNPredictor

from src.utils import jaccard

class SODExplainer:
    def __init__(
        self, detector='FasterRCNN',load_from=None
    ):
        """[summary]
        Args:
            detector (string): Object detector. Defaults to "FasterRCNN".
            load_from: Model path. Defaults to None.
        """
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True) # pretrained model         
        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
        
        if load_from is not None:
            # load checkpoint 
            self.model.load_state_dict(torch.load(load_from, map_location=torch.device('cpu'))['model_state_dict'])
                   
            
    def get_class_probability(self, data_obj):
        """ The Method takes one image from the dataset and returns the class score as the probability 
            indicating how likely we detect the most confident object in this image.
        Args:
            data_obj (img, target): A sample from the PennFudanDataset
        Returns:
            probabilities (Tensor): Probability distribution for detecting the most confident object
        """
        img, target = data_obj # ground truth
        
        self.model.eval() # set the module in evaluation mode
        boxes = self.model(img.unsqueeze(0))[0]['boxes']
        scores = self.model(img.unsqueeze(0))[0]['scores']
        
        if len(boxes) == 0: 
            # if there's no object detected
            prob = 0
            print("No object detected!")
        else:
            ious = jaccard(target['boxes'],boxes) # ious with shape (num_objs, num_boxes)
            ious = ious[ious > 0.4].unsqueeze(1) 
            if len(ious) == 0: # No score above the threshold
                prob = 0
            else:
                obj_idx, box_idx = np.unravel_index(torch.argmax(ious), ious.shape) # retrieve argmax-indices in 2d
                prob = scores[box_idx] 
        probabilities = [prob, 1 - prob]      
        return torch.Tensor(probabilities)
    
    # def explain(self, data_obj):
        
        # Do explanation