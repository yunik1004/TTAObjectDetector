import os
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


class ObjectDetector:
    def __init__(self, cuda=True):
        # Create config
        cfg_path = os.path.join("COCO-Detection", "faster_rcnn_R_101_FPN_3x.yaml")

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(cfg_path))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_path)
        if not cuda:
            cfg.MODEL.DEVICE = "cpu"

        # Create predictor
        self.__predictor = DefaultPredictor(cfg)

    @property
    def predictor(self):
        return self.__predictor
