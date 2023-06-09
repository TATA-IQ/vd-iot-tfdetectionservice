import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    
import pathlib
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings, argparse
warnings.filterwarnings('ignore')
import pandas as pd
import cv2
import glob2 as glob
import datetime
import threading
import json
import requests
from imutils.video import VideoStream
import imutils

class InferenceModel():
    def __init__(self,model_path,gpu=False):
        self.model_path=model_path
        self.model=None
        self.detect_fn=None
        self.classes=None
        if gpu:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    def loadmodel(self):
        print("model_path====>",self.model_path)
        self.model = tf.saved_model.load(self.model_path)
        self.detect_fn = self.model.signatures['serving_default']
    
    def read_label_map(self,label_map_path):
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
        self.classes=items

        
    def getClasses(self):
        return self.classes
    def load_image_into_numpy_array(self,image):

        return np.array(image)
    

    def object_detection(self,image_path, detect_fn):
        image_np = self.load_image_into_numpy_array(image_path)
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = self.detect_fn(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        return(detections)
    def infer(self,img):
        print("=====inference starting=====")
        image = Image.fromarray(img)
        image_np_cv = img.copy()
        print("=====detection starting=====")
        det = self.object_detection(image, self.detect_fn)
        print("=====detection done=====")
        print(det)
        listresult=[]
        for i in range(0, len(det["detection_classes"])):
        #if(det['detection_classes'][i]==1 and det['detection_scores'][i]>=0.1):
            Xmin=int(image_np_cv.shape[1]*det['detection_boxes'][i][1].item())
            Ymin=int(image_np_cv.shape[0]*det['detection_boxes'][i][0].item())
            Xmax=int(image_np_cv.shape[1]*det['detection_boxes'][i][3].item())
            Ymax=int(image_np_cv.shape[0]*det['detection_boxes'][i][2].item())
            clas_nm=det['detection_classes'][i].item()
            score=det['detection_scores'][i].item()
            #print(")))))",type(Xmin))
            try:
                listresult.append({"xmin":Xmin,"ymin":Ymin,"xmax":Xmax,"ymax":Ymax,"class":clas_nm,"score":score})
            except:
                listresult.append({"xmin":Xmin,"ymin":Ymin,"xmax":Xmax,"ymax":Ymax,"class":self.classes[clas_nm],"score":score})
        return listresult


