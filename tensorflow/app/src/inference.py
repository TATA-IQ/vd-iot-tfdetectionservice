import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import pathlib
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings, argparse

warnings.filterwarnings("ignore")
import pandas as pd
import cv2
import glob2 as glob
import datetime
import threading
import json
import requests
from imutils.video import VideoStream
import imutils


class InferenceModel:
    def __init__(self, model_path, gpu=False):
        self.model_path = model_path
        self.model = None
        self.detect_fn = None
        self.classname= {}
        self.classes = None
        if gpu:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

    def loadmodel(self):
        print("model_path====>", self.model_path)
        
        self.model = tf.saved_model.load(self.model_path)
        self.detect_fn = self.model.signatures["serving_default"]

    def read_label_map(self, label_map_path):
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
        for k, v in self.classes.items():
            self.classname[v]=k

    def getClasses(self):   
        return self.classes

    def load_image_into_numpy_array(self, image):
        return np.array(image)

    def object_detection(self, image_path, detect_fn):
        image_np = self.load_image_into_numpy_array(image_path)
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = self.detect_fn(input_tensor)
        num_detections = int(detections.pop("num_detections"))
        detections = {
            key: value[0, :num_detections].numpy() for key, value in detections.items()
        }
        detections["num_detections"] = num_detections
        detections["detection_classes"] = detections["detection_classes"].astype(
            np.int64
        )
        return detections

    def infer(self, img,model_config):
        print("=====inference starting=====")
        image = Image.fromarray(img)
        image_np_cv = img.copy()
        print("=====detection starting=====")
        det = self.object_detection(image, self.detect_fn)
        print("=====detection done=====")
        print(self.classes)
        print(self.classname)
        # print(det)
        listresult = []
        for i in range(0, len(det["detection_classes"])):
            # if(det['detection_classes'][i]==1 and det['detection_scores'][i]>=0.1):
            Xmin = int(image_np_cv.shape[1] * det["detection_boxes"][i][1].item())
            Ymin = int(image_np_cv.shape[0] * det["detection_boxes"][i][0].item())
            Xmax = int(image_np_cv.shape[1] * det["detection_boxes"][i][3].item())
            Ymax = int(image_np_cv.shape[0] * det["detection_boxes"][i][2].item())
            
            clas_nm = det["detection_classes"][i].item()
            score = det["detection_scores"][i].item()
            # print(")))))",type(Xmin))
            try:
                listresult.append(
                    {
                        "class": clas_nm,
                        "class_name": self.classname[clas_nm],
                        "score": score,
                        "xmin": Xmin,
                        "ymin": Ymin,
                        "xmax": Xmax,
                        "ymax": Ymax,
                        "xmin_c": round(det['detection_boxes'][i][1].item(),4),
                        "ymin_c": round(det['detection_boxes'][i][0].item(),4),
                        "xmax_c": round(det['detection_boxes'][i][3].item(),4),
                        "ymax_c": round(det['detection_boxes'][i][2].item(),4),
                        
                    }
                )
            except UnboundLocalError:
                listresult.append(
                    {
                        "class": clas_nm,
                        "class_name": self.classname[clas_nm],
                        "score": score,
                        "xmin": Xmin,
                        "ymin": Ymin,
                        "xmax": Xmax,
                        "ymax": Ymax,
                        "xmin_c": round(det['detection_boxes'][i][1].item(),4),
                        "ymin_c": round(det['detection_boxes'][i][0].item(),4),
                        "xmax_c": round(det['detection_boxes'][i][3].item(),4),
                        "ymax_c": round(det['detection_boxes'][i][2].item(),4),
                    }
                )
        return listresult
    
    def mark_res(self, image_np_cv, det, origin_y, origin_x, H, W):
        listresult = []
        for i in range(0, len(det["detection_classes"])):
            # if(det['detection_classes'][i]==1 and det['detection_scores'][i]>=0.1):
            Xmin = int(image_np_cv.shape[1] * det["detection_boxes"][i][1].item())
            Ymin = int(image_np_cv.shape[0] * det["detection_boxes"][i][0].item())
            Xmax = int(image_np_cv.shape[1] * det["detection_boxes"][i][3].item())
            Ymax = int(image_np_cv.shape[0] * det["detection_boxes"][i][2].item())
            
            clas_nm = det["detection_classes"][i].item()
            score = det["detection_scores"][i].item()
            # print(")))))",type(Xmin))
            try:
                listresult.append(
                    {
                        "class": clas_nm,
                        "class_name": self.classname[clas_nm],
                        "score": score,
                        "xmin": Xmin + origin_x,
                        "ymin": Ymin + origin_y,
                        "xmax": Xmax + origin_x,
                        "ymax": Ymax + origin_y,
                        "xmin_c": round((Xmin + origin_x)/W,5),
                        "ymin_c": round((Ymin + origin_y)/H,5),
                        "xmax_c": round((Xmax + origin_x)/W,5),
                        "ymax_c": round((Ymax + origin_y)/H,5),
                        # "xmin_c": round(det['detection_boxes'][i][1].item(),4),
                        # "ymin_c": round(det['detection_boxes'][i][0].item(),4),
                        # "xmax_c": round(det['detection_boxes'][i][3].item(),4),
                        # "ymax_c": round(det['detection_boxes'][i][2].item(),4),
                        
                    }
                )
            except UnboundLocalError:
                listresult.append(
                    {
                        "class": clas_nm,
                        "class_name": self.classname[clas_nm],
                        "score": score,
                        "xmin": Xmin + origin_x,
                        "ymin": Ymin + origin_y,
                        "xmax": Xmax + origin_x,
                        "ymax": Ymax + origin_y,
                        "xmin_c": round((Xmin + origin_x)/W,5),
                        "ymin_c": round((Ymin + origin_y)/H,5),
                        "xmax_c": round((Xmax + origin_x)/W,5),
                        "ymax_c": round((Ymax + origin_y)/H,5),
                    }
                )
        return listresult
    
    def deduplication(self,det_list, h,w):
        H = h
        W = w
        final_det = []
        clist_i = []
        for i in range(0, len(det_list)):
            for j in range(i+1, len(det_list)):
                if(i not in clist_i and (det_list[i][0]['class'] == det_list[j][0]['class']) and 
                   (((abs(det_list[i][0]['xmin'] - det_list[j][0]['xmax']) <= 5 or abs(det_list[i][0]['xmax'] - det_list[j][0]['xmin']) <= 5) and 
                   ((det_list[i][0]['ymax']-det_list[j][0]['ymin'])/(det_list[j][0]['ymax']-det_list[j][0]['ymin']) > 0.5 and
                    (det_list[j][0]['ymax']-det_list[i][0]['ymin'])/(det_list[i][0]['ymax']-det_list[i][0]['ymin']) > 0.5)) or 
                   (((abs(det_list[i][0]['ymin'] - det_list[j][0]['ymax']) <= 5 or abs(det_list[i][0]['ymax'] - det_list[j][0]['ymin']) <= 5)) and 
                   ((det_list[i][0]['xmax']-det_list[j][0]['xmin'])/(det_list[j][0]['xmax']-det_list[j][0]['xmin']) > 0.5 and
                    (det_list[j][0]['xmax']-det_list[i][0]['xmin'])/(det_list[i][0]['xmax']-det_list[i][0]['xmin']) > 0.5)))):
                    #print(str(i)+"_"+str(j))
                    #print(det_list[i][0])
                    #print(det_list[j][0])
                    cord =   [{'class': det_list[i][0]['class_id'],
                               "id":None,
                              'class_name': det_list[i][0]['class'],
                              'score': det_list[i][0]['score'],
                              'xmin': min(det_list[i][0]['xmin'], det_list[j][0]['xmin']),
                              'ymin': min(det_list[i][0]['ymin'], det_list[j][0]['ymin']),
                              'xmax': max(det_list[i][0]['xmax'], det_list[j][0]['xmax']),
                              'ymax': max(det_list[i][0]['ymax'], det_list[j][0]['ymax']),
                              'xmin_c': min(det_list[i][0]['xmin'], det_list[j][0]['xmin']/W),
                              'ymin_c': min(det_list[i][0]['ymin'], det_list[j][0]['ymin']/H),
                              'xmax_c': max(det_list[i][0]['xmax'], det_list[j][0]['xmax'])/W,
                              'ymax_c': max(det_list[i][0]['ymax'], det_list[j][0]['ymin'])/H}]
                    #print(cord)
                    final_det.append(cord)
                    clist_i.append(i)
                    clist_i.append(j)
        for i in range(0, len(det_list)):
            if(i not in clist_i):
                final_det.append(det_list[i])
        return final_det
    def split(self,frame,split_col,split_row,model):
        swidth_col =  int(frame.shape[1]/split_col)
        sheight_row =  int(frame.shape[0]/split_row)
        det_list = []
        h,w,_=frame.shape
        print(f"frame height {h}, width {w}")
        for i in range(0, split_row):
            for j in range(0, split_col):
                sub_img = frame[i*sheight_row:(i+1)*sheight_row, j*swidth_col:(j+1)*swidth_col]                
                # res=model.predict(sub_img)      
                print(f"sub frame height {sub_img.shape[0]}, width {sub_img.shape[1]}")          
                # cv2.imwrite("config/"+str(i)+"_"+str(j)+".jpg",sub_img)

                image = Image.fromarray(sub_img)
                image_np_cv = sub_img.copy()
                print("=====detection starting=====")
                det = self.object_detection(image, self.detect_fn)
                print("=====detection done=====")
                print(self.classes)
                print(self.classname)
                # listresult = self.mark_res(image_np_cv,det, i*sheight_row, j*swidth_col, frame.shape[0], frame.shape[1])
                # # print(det)
                origin_y,origin_x,H,W=i*sheight_row, j*swidth_col, frame.shape[0], frame.shape[1]
                listresult = []
                for i1 in range(0, len(det["detection_classes"])):
                    # if(det['detection_classes'][i]==1 and det['detection_scores'][i]>=0.1):
                    Xmin = int(image_np_cv.shape[1] * det["detection_boxes"][i1][1].item())
                    Ymin = int(image_np_cv.shape[0] * det["detection_boxes"][i1][0].item())
                    Xmax = int(image_np_cv.shape[1] * det["detection_boxes"][i1][3].item())
                    Ymax = int(image_np_cv.shape[0] * det["detection_boxes"][i1][2].item())
                    
                    clas_nm = det["detection_classes"][i1].item()
                    score = det["detection_scores"][i1].item()
                    # print(")))))",type(Xmin))
                    try:
                        listresult.append(
                            {
                                "class": clas_nm,
                                "class_name": self.classname[clas_nm],
                                "score": score,
                                "xmin": Xmin + origin_x,
                                "ymin": Ymin + origin_y,
                                "xmax": Xmax + origin_x,
                                "ymax": Ymax + origin_y,
                                "xmin_c": round((Xmin + origin_x)/W,5),
                                "ymin_c": round((Ymin + origin_y)/H,5),
                                "xmax_c": round((Xmax + origin_x)/W,5),
                                "ymax_c": round((Ymax + origin_y)/H,5),
                                # "xmin_c": round(det['detection_boxes'][i][1].item(),4),
                                # "ymin_c": round(det['detection_boxes'][i][0].item(),4),
                                # "xmax_c": round(det['detection_boxes'][i][3].item(),4),
                                # "ymax_c": round(det['detection_boxes'][i][2].item(),4),
                                
                            }
                        )
                    except UnboundLocalError:
                        listresult.append(
                            {
                                "class": clas_nm,
                                "class_name": self.classname[clas_nm],
                                "score": score,
                                "xmin": Xmin + origin_x,
                                "ymin": Ymin + origin_y,
                                "xmax": Xmax + origin_x,
                                "ymax": Ymax + origin_y,
                                "xmin_c": round((Xmin + origin_x)/W,5),
                                "ymin_c": round((Ymin + origin_y)/H,5),
                                "xmax_c": round((Xmax + origin_x)/W,5),
                                "ymax_c": round((Ymax + origin_y)/H,5),
                            }
                        )
                # return listresult
                if len(listresult)>0:
                    det_list.append(listresult)
        return sum(self.deduplication(det_list,h,w),[])
                
    def infer_v2(self, img, model_config, split_columns=1, split_rows=1):
        image = img.copy()
        print(split_columns, split_rows)
        listresult = self.split(image, split_columns, split_rows, self.model)
        if len(listresult)==0:
            print("no detections")
        else:
            print(listresult)
        return listresult
        
        # print("=====inference starting=====")
        # image = Image.fromarray(img)
        # image_np_cv = img.copy()
        # print("=====detection starting=====")
        # det = self.object_detection(image, self.detect_fn)
        # print("=====detection done=====")
        # print(self.classes)
        # print(self.classname)
        # # print(det)
        # listresult = []
        # for i in range(0, len(det["detection_classes"])):
        #     # if(det['detection_classes'][i]==1 and det['detection_scores'][i]>=0.1):
        #     Xmin = int(image_np_cv.shape[1] * det["detection_boxes"][i][1].item())
        #     Ymin = int(image_np_cv.shape[0] * det["detection_boxes"][i][0].item())
        #     Xmax = int(image_np_cv.shape[1] * det["detection_boxes"][i][3].item())
        #     Ymax = int(image_np_cv.shape[0] * det["detection_boxes"][i][2].item())
            
        #     clas_nm = det["detection_classes"][i].item()
        #     score = det["detection_scores"][i].item()
        #     # print(")))))",type(Xmin))
        #     try:
        #         listresult.append(
        #             {
        #                 "class": clas_nm,
        #                 "class_name": self.classname[clas_nm],
        #                 "score": score,
        #                 "xmin": Xmin,
        #                 "ymin": Ymin,
        #                 "xmax": Xmax,
        #                 "ymax": Ymax,
        #                 "xmin_c": round(det['detection_boxes'][i][1].item(),4),
        #                 "ymin_c": round(det['detection_boxes'][i][0].item(),4),
        #                 "xmax_c": round(det['detection_boxes'][i][3].item(),4),
        #                 "ymax_c": round(det['detection_boxes'][i][2].item(),4),
                        
        #             }
        #         )
        #     except UnboundLocalError:
        #         listresult.append(
        #             {
        #                 "class": clas_nm,
        #                 "class_name": self.classname[clas_nm],
        #                 "score": score,
        #                 "xmin": Xmin,
        #                 "ymin": Ymin,
        #                 "xmax": Xmax,
        #                 "ymax": Ymax,
        #                 "xmin_c": round(det['detection_boxes'][i][1].item(),4),
        #                 "ymin_c": round(det['detection_boxes'][i][0].item(),4),
        #                 "xmax_c": round(det['detection_boxes'][i][3].item(),4),
        #                 "ymax_c": round(det['detection_boxes'][i][2].item(),4),
        #             }
        #         )
        # return listresult
