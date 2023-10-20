FROM python:3.9.2
RUN pip install pandas
RUN pip install kafka-python
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install opencv-python
RUN pip install numpy
RUN pip install matplotlib
RUN pip install pandas
RUN pip install requests
RUN pip install sqlalchemy
#RUN pip3 install torch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install imutils
RUN pip install PyYAML
RUN pip install tqdm
RUN pip install seaborn
RUN pip install scipy
Run pip install glob2
Run pip install paramiko
Run pip install fastapi
Run pip install "uvicorn[standard]"
Run pip install tensorflow==2.8.3
Run pip install protobuf==3.20.*
Run pip install fastapi==0.99.1
Run pip install "uvicorn[standard]"
Run pip install protobuf==3.20.*
Run pip install minio
copy tensorflow/ /app
WORKDIR /app
cmd ["python3","app.py"]