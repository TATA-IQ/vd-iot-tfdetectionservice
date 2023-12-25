from minio import Minio
from minio.error import S3Error
import urllib3
import cv2
from zipfile import ZipFile
import os
import numpy as np
from datetime import datetime
import shutil
from console_logging.console import Console
console=Console()


class DownloadModel:
    """
    Class to download model from miniodb and remove it after validation is done.
    """

    def __init__(self, bucket_name, minioconf, logger):
        """
        args: bucket_name-> Bucket name of miniodb
        """
        self.bucket_name = bucket_name
        self.log = logger
        self.client = Minio(
            endpoint=minioconf["endpoint"],
            access_key=minioconf["access_key"],
            secret_key=minioconf["secret_key"],
            secure=minioconf["secure"],
        )

    def save_data(self, object_name, local_path):
        """
        args:object_name-> full path of file from miniodb
        args: local_path-> path to save model locally
        """
        obj_name = object_name.split("/")[-1]
        self.log.info(f"obj_name {obj_name}")
        console.info(f"obj_name {obj_name}")
        save_path = os.path.join(local_path, obj_name)
        try:
            self.client.fget_object("models", object_name, save_path)
            print(f"{object_name} is saved into {save_path}")
            self.log.info(f"{object_name} is saved into {save_path}")
            console.info(f"{object_name} is saved into {save_path}")
        except S3Error as exp:
            print(f"{object_name} {exp.message} ")
            self.log.info(f"expection raised, no buckets {exp} i.e., for path {object_name}")
            console.info(f"expection raised, no buckets {exp} i.e., for path {object_name}")

    def save_model_files(self, object_path, local_path):
        """
        args:object_name-> full path of file from miniodb
        args:local_path-> path to save model  locally
        """
        obj_name = object_path.split("/")[-1]
        # print(obj_name)
        save_path = os.path.join(local_path, obj_name)
        try:
            self.client.fget_object("models", object_path, save_path)
            print(f"{object_path} is saved into {save_path}")
        except S3Error as e:
            print(e)
            print(f"{object_name} {e.message} ")

    def unzip(self, path, unzippath, modelname):
        """
        args:path-> path of the downloaded model in zip file
        unzippath:
        modelname:
        """
        self.log.info(f"zip path===>{path}")
        console.info(f"zip path===>{path}")
        with ZipFile(path, "r") as zObject:
            zObject.extractall(path=unzippath)
        os.remove(path)

    def removeData(self, path):
        shutil.rmtree(path)


# bucket_name = "yolov5"
# # object_name = "minio_images.zip"
# # object_name = "test(2).mp4"
# # object_name = "cam4/cam4_2023_06_13_18_17_01_739876.jpg"
# object_name = "/object_detection/usecase1/model_id_2/ppe.zip"  #object_path
# local_path = "/home/sridhar.bondla/api_database/app/minIO_db/minio_data1/"

# a = download_files(bucket_name)
# # a.save_data(object_name,local_path)
# a.save_model_files(object_name,local_path)
