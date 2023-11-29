# test not required here yet
import cv2
import requests
import random
import base64
import numpy as np
from datetime import datetime
from PIL import Image
from io import BytesIO
import io

# from src.inference import InferenceModel

# model_path = "DeployedModel/Person/saved_model"
# image_path = ""
# ifm = InferenceModel(model_path)
# ifm.loadmodel()
# img = cv2.imread("test.jpg")
# res = ifm.infer(img)
# print(res)

url = "http://0.0.0.0:7000/detect/"

img = cv2.imread("2023_08_22_16_07_09_0.jpg")
img_str = cv2.imencode(".jpg", img)[1].tobytes().decode("ISO-8859-1")

stream = BytesIO(img_str.encode("ISO-8859-1"))
image = Image.open(stream).convert("RGB")
open_cv_image = np.array(image) 

query = {
    'image': img_str,
    'image_name': 'image3.jpg',
    'camera_id': '123',
    'image_time': '2023-09-04 22:13:23.123456',
    'model_type': 'object detection',
    'model_framework': 'yolov5',
    'model_config': {
    'is_track':True,
    'conf_thres': 0.8,
    'iou_thres': 0.5,
    'max_det': 300,
    'agnostic_nms': True,
    'augment': False,
    
   },
    "split_columns": 1,
    "split_rows": 1}


r = requests.post(url, json=query)
data = r.json()
print(r.json())

detections = r.json()['data']
print(len(detections))
if len(detections)>0:
    for a,i in enumerate(detections):
        img1 = cv2.rectangle(img,(i['xmin'],i['ymin']),(i["xmax"],i["ymax"]),(255,255,0),2)
        if i["class"]==5:
            cv2.putText(img, i["class_name"], (i['xmin'],i['ymin']), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
    cv2.imwrite("sub_"+str(i["class_name"])+"_"+str(a)+".jpg",img1)
else:
    print("no detections")
