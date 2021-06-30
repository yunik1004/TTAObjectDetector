from model import ObjectDetector


if __name__ == "__main__":
    detector = ObjectDetector(cuda=False)

    dataset_val = "coco_2017_val"
    dataset_test = "coco_2017_test-dev"
    output_dir = "./output/"

    res_val = detector.eval(dataset_val, output_dir)
    res_test = detector.eval(dataset_test, output_dir)
