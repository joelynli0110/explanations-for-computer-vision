import torch
import torchvision
import numpy as np

from lime.lime_image import LimeImageExplainer
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

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
                   
            
    def get_class_probability(self,data_obj):
      _, target = data_obj # ground truth
      def get_probabilities(img_as_arrays):
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
                  obj_idx, box_idx = np.unravel_index(torch.argmax(ious), ious.shape) # retrieve argmax-indices in 2d
                  prob = scores[box_idx] 
          probabilities = [prob, 1 - prob]
          img_probs.append(np.array(probabilities))   
        return np.array(img_probs)

      return get_probabilities
    
    #get_explnation
    def get_explanation(self,image_test,data_obj):
      explainer = LimeImageExplainer(verbose=True)
      # self.logger.info("Explaining object: ")
      explanation = explainer.explain_instance(
          image= image_test,
          classifier_fn=self.get_class_probability(data_obj),
          num_samples=10)
      return explanation