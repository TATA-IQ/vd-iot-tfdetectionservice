import cv2
from src.inference import InferenceModel
model_path="DeployedModel/Person/saved_model"
image_path=""
ifm=InferenceModel(model_path)
ifm.loadmodel()
img=cv2.imread("test.jpg")
res=ifm.infer(img)
print(res)