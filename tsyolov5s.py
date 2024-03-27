# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 10:45:24 2022

@author: rtx20708g
"""
import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')  # or yolov5n - yolov5x6, custom

# Images
img = 'catDog.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
results.save()