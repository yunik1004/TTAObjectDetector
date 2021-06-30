import os
from typing import Any, Dict, List, Tuple
import cv2
from detectron2.data import MetadataCatalog
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from model import ObjectDetector


class COCODataLoader:
    """
    Data loader for the COCO dataset
    """

    def __init__(self, dataset: str, ann_path: str, img_dir_path: str) -> None:
        """
        Constructor of the COCODataLoader

        Parameters
        ----------
        dataset : str
            Name of the COCO dataset
        ann_path : str
            Path of the annotation file
        img_dir_path : str
            Path of the image directory
        """
        self._ann_path = ann_path
        self._coco = COCO(self._ann_path)
        self._img_dir_path = img_dir_path  # Path of the image directory
        self._imgIds = self._coco.getImgIds()  # Image ids
        self._metadata = MetadataCatalog.get(dataset)  # Metadata of the dataset

    @property
    def imgIds(self) -> List[int]:
        return self._imgIds

    @property
    def ann_path(self) -> str:
        return self._ann_path

    @property
    def metadata(self) -> List[Any]:
        return self._metadata

    def getCategoryInfo(self, catId: int) -> Dict[str, Any]:
        """
        Return the category information

        Parameters
        ----------
        catId : int
            Id of the category

        Returns
        -------
        Dict[str, Any]
            Category information
        """
        cont_id = self._metadata.thing_dataset_id_to_contiguous_id[catId]

        res = {
            "name": self._metadata.thing_classes[cont_id],
            "color": self._metadata.thing_colors[cont_id],
        }

        return res

    def loadImg(self, imgId: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Return the image

        Parameters
        ----------
        imgId : int
            Id of the image

        Returns
        -------
        Tuple[np.ndarray, List[Dict[str, Any]]]
            (Image tensor, annotation of the image)
        """
        img_info = self._coco.loadImgs(imgId)[0]
        img_annIds = self._coco.getAnnIds(imgIds=imgId)
        img_path = os.path.join(self._img_dir_path, img_info["file_name"])
        img_anns = self._coco.loadAnns(img_annIds)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        return img, img_anns

    def showImg(self, imgId: int) -> None:
        """
        Visualize the image

        Parameters
        ----------
        imgId : int
            Id of the image
        """
        img_info = self._coco.loadImgs(imgId)[0]
        img_annIds = self._coco.getAnnIds(imgIds=imgId)
        img_name = img_info["file_name"]
        img_path = os.path.join(self._img_dir_path, img_name)
        img_anns = self._coco.loadAnns(img_annIds)

        img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        plt.figure(img_name)
        plt.imshow(img)
        self._coco.showAnns(img_anns, draw_bbox=True)

        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    detector = ObjectDetector(cuda=False)

    loader = COCODataLoader(
        dataset="coco_2017_val",
        ann_path="datasets/coco/annotations/instances_val2017.json",
        img_dir_path="datasets/coco/val2017",
    )

    img_ids = loader.imgIds
    img, img_anns = loader.loadImg(img_ids[0])
    out = detector.predict(img)
    print(out)
