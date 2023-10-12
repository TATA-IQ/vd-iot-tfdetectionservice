import os
import numpy as np
import tensorflow as tf
from PIL import Image
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")


class InferenceModel:
    def __init__(self, model_path, gpu=False):
        """
        Initialize Yolov5 inference

        Args:
            model_path (str): path of the downloaded and unzipped model
            gpu=True, if the system have NVIDIA GPU compatibility
        """
        self.model_path = model_path
        self.model = None
        self.detect_fn = None
        self.classes = None
        if gpu:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

    def loadmodel(self):
        """
        This will load Yolov5 model
        """
        print("model_path====>", self.model_path)
        self.model = tf.saved_model.load(self.model_path)
        self.detect_fn = self.model.signatures["serving_default"]

    def read_label_map(self, label_map_path):
        """
        This function read the .pbtext file for classes of the model
        Args:
            label_map_path (str): path of the .pbtext file.
        """
        item_id = None
        item_name = None
        items = {}

        with open(label_map_path, "r") as file:
            for line in file:
                line.replace(" ", "")
                if line == "item{":
                    pass
                elif line == "}":
                    pass
                elif "id" in line:
                    item_id = int(line.split(":", 1)[1].strip())
                elif "name" in line:
                    item_name = line.split(":", 1)[1].replace("'", "").strip()

                if item_id is not None and item_name is not None:
                    items[item_name] = item_id
                    item_id = None
                    item_name = None
        self.classes = items

    def getClasses(self):
        """
        Get the classes of the model

        """
        return self.classes

    def load_image_into_numpy_array(self, image):
        """
        load images into the numpy array
        """
        return np.array(image)

    def object_detection(self, image, detect_fn):
        """
        This will return the object detection result
        Args:
            image (np.array): image array
            detect_fn (function): detction function initialized by tensorflow
        """
        image_np = self.load_image_into_numpy_array(image)
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = self.detect_fn(input_tensor)
        num_detections = int(detections.pop("num_detections"))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}  # noqa: E501
        detections["num_detections"] = num_detections
        detections["detection_classes"] = detections["detection_classes"].astype(np.int64)  # noqa: E501
        return detections

    def infer(self, img, model_config=None):
        """
        This will do the detection on the image
        Args:
            img (array): image in numpy array
            model_config (dict): configuration specific to camera group for detection   # noqa: E501
        Returns:
            list: list of dictionary. It will have all the detection result.
        """
        print("=====inference starting=====")
        image = Image.fromarray(img)
        image_np_cv = img.copy()
        print("=====detection starting=====")
        det = self.object_detection(image, self.detect_fn)
        print("=====detection done=====")
        print(det)
        listresult = []
        for i in range(0, len(det["detection_classes"])):
            Xmin = int(image_np_cv.shape[1] * det["detection_boxes"][i][1].item())  # noqa: E501
            Ymin = int(image_np_cv.shape[0] * det["detection_boxes"][i][0].item())  # noqa: E501
            Xmax = int(image_np_cv.shape[1] * det["detection_boxes"][i][3].item())  # noqa: E501
            Ymax = int(image_np_cv.shape[0] * det["detection_boxes"][i][2].item())  # noqa: E501
            clas_nm = det["detection_classes"][i].item()
            score = det["detection_scores"][i].item()
            # print(")))))",type(Xmin))
            try:
                listresult.append(
                    {
                        "xmin": Xmin,
                        "ymin": Ymin,
                        "xmax": Xmax,
                        "ymax": Ymax,
                        "class": clas_nm,
                        "score": score,
                    }
                )
            except UnboundLocalError:
                listresult.append(
                    {
                        "xmin": Xmin,
                        "ymin": Ymin,
                        "xmax": Xmax,
                        "ymax": Ymax,
                        "class": self.classes[clas_nm],
                        "score": score,
                    }
                )
        return listresult
