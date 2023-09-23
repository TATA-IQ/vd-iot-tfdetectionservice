# Introduction 
This is a camera streaming repo. This contains the code of streaming multiple camera(203 has been tested till yet)

# How It Works

1. Check the config for customer id, subsite id, location id or camera group id
2. Aggregate the camera group and get all the camera config and cache the data
3. Read camera configuration from cache
4. Start streaming for camera
5. Check for the update in cache and update the camera configurations

# Architecture
![Architectural Flow](../../CameraStreaming/images/CamModule.png)

# Dependency
1. This Module is dependent on the https://tatacommiot@dev.azure.com/tatacommiot/Video%20Based%20IoT/_git/vd-iot-dataapiservice
2. This module also needs kafka broker

# Installation
1. Install Python3.9 
2. Install redis-server
3. poetry install
4. python3 cache.py
4. python3 streaming.py

# Docker 
To-do: Docker Implementation