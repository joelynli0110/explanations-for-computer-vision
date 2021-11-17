class SODExplainer:
    def __init__(
        self, detector='FasterRCNN'
    ):
        """[summary]
        Args:
            detector (string): Object detector. Defaults to "FasterRCNN".
        """
        
        # self.model = detector .... 
        
    def get_class_probability(self, data_obj):
        """[summary]
        Args:
            data_obj (img, target): a sample from the PennFudanDataset
        """
        img, target = data_obj
        probabilitites = model(data_obj)
        # ....
    
    def explain(self, data_obj):
        
        # Do explanation