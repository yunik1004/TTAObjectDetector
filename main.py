from model import ObjectDetector


if __name__ == "__main__":
    mod = ObjectDetector(cuda=False)
    print(mod.predictor)
