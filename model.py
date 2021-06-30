import os
from typing import Any, Dict, List
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model
import numpy as np
import torch
from tqdm import tqdm


class ObjectDetector:
    """
    Detect the objects using bounding boxes
    """

    def __init__(self, cuda: bool = True) -> None:
        """
        Constructor of the object detector

        Parameters
        ----------
        cuda : bool, optional (by default True)
            True if the detector model uses CUDA
        """
        # Create config
        cfg_path = os.path.join("COCO-Detection", "faster_rcnn_R_101_FPN_3x.yaml")

        self._cfg = get_cfg()
        self._cfg.merge_from_file(model_zoo.get_config_file(cfg_path))
        self._cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
            0.5  # set threshold for this model
        )
        self._cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_path)
        if not cuda:
            self._cfg.MODEL.DEVICE = "cpu"

        # Create predictor
        self._model = build_model(self._cfg)
        self._model.eval()

        checkpointer = DetectionCheckpointer(self._model)
        checkpointer.load(self._cfg.MODEL.WEIGHTS)

        self._pbar = None

    def __call__(self, img_infos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Callable function of the instance

        Parameters
        ----------
        img_infos : List[Dict[str, Any]]
            Information of the image: e.g. {"image": image, "height": height, "width": width}

        Returns
        -------
        List[Dict[str, Any]]
            Informations of the detected objects
        """
        pred_list = list()

        with torch.no_grad():
            for img_info in img_infos:
                pred = self._model([img_info])[0]
                pred_list.append(pred)
                if self._pbar:
                    self._pbar.update(1)

        return pred_list

    def predict(self, img: np.ndarray) -> Dict[str, Any]:
        """
        Predict the objects in the image

        Parameters
        ----------
        img : np.ndarray
            Image tensor. It should be of size [height, width, channel]

        Returns
        -------
        Dict[str, Any]
            Information of the detected objects
        """
        height, width = img.shape[:2]
        image = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        img_info = {"image": image, "height": height, "width": width}

        pred = self.__call__([img_info])[0]

        return pred

    def eval(self, dataset: str, output_dir: str) -> Dict[str, Any]:
        """
        Evaluate the object detector on the given dataset

        Parameters
        ----------
        dataset : str
            Dataset of the COCO evaluator
        output_dir : str
            Output directory where the result will be saved

        Returns
        -------
        Dict[str, Any]
            Inference output on the dataset
        """
        evaluator = COCOEvaluator(dataset, ("bbox"), False, output_dir=output_dir)
        val_loader = build_detection_test_loader(self._cfg, dataset)

        self._pbar = tqdm(total=len(val_loader))
        res = inference_on_dataset(self, val_loader, evaluator)
        self._pbar.close()
        self._pbar = None

        return res


if __name__ == "__main__":
    detector = ObjectDetector(cuda=False)

    dataset_val = "coco_2017_val"
    output_dir = "./output/"

    res = detector.eval(dataset_val, output_dir)
    print(res)
