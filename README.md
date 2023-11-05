# Computer Vision For All - Object Tracking With YOLO and Python

## Introduction
In the ever-evolving realm of computer vision and artificial intelligence, 
object tracking is a pivotal concept with diverse applications, from 
autonomous vehicles to surveillance systems. YOLOv8, a 
state-of-the-art real-time object detection framework, has gained 
significant attention. In this blog post, we explore the world of 
YOLOv8 object tracking, showcasing its capabilities and adding intelligence 
by analyzing tracked object statistics.

![Intro](./images/thumnbail.png)

Our Python-based project harnesses YOLOv8's power for highly accurate 
real-time object tracking. But we go a step further by examining 
tracked objects' movements, measuring distances traveled in pixels, 
and calculating average speeds. This approach offers a comprehensive 
understanding of how these objects behave in their environment.

Whether you're a computer vision enthusiast, a developer looking to add 
object tracking to your applications, or someone intrigued by AI, this 
post aims to inspire and educate. We dive into YOLOv8's potential, technical 
intricacies of object tracking, and how to gain insights into tracked object 
motion. By the end, you'll have the knowledge and tools to implement your 
object tracking solutions and a deeper understanding of the dynamic world 
within your videos and images.

## Libraries and References
For our object tracking project, we rely on two main libraries:

* OpenCV: Used for opening video streams, frame drawing, and more. It's a 
versatile open-source software library for computer vision and image processing, 
making it valuable for object detection, facial recognition, image stitching, and 
motion tracking. OpenCV's popularity stems from its efficiency, ease of use, and 
extensive community support, making it a preferred choice in various fields, 
including robotics, machine learning, and computer vision.

* Ultralytics: An open-source software framework focused on computer vision 
and deep learning. It streamlines the development of object detection, image 
classification, and other machine learning tasks. Ultralytics is popular for 
its user-friendly nature, comprehensive documentation, and seamless integration 
with PyTorch, a leading deep learning framework. It simplifies complex tasks 
like training and deploying neural networks for applications like autonomous 
vehicles, surveillance systems, and medical image analysis, earning recognition 
as an essential resource in the deep learning and computer vision community.

We also utilize supporting libraries like numpy, and you can find a complete 
list of requirements [here]().

## Building
Without further adieu, let's begin building our system. Our system will consist
of two main building blocks:

1. An object detection model which will accept a frame and perform 
    object detection and tracking on the given frame
2. A main loop which will fetch frames from a video stream, feed them into our 
    detection system defined above, annotate the frames, and show them to our user

Let's start with our object detection model, which is defined in [detector.py]().
As is customary with any python project, let's import our required libraries:

```python
# For machine learning
import torch
# For array computations
import numpy as np
# For image decoding / editing
import cv2
# For environment variables
import os
# For detecting which ML Devices we can use
import platform
# For actually using the YOLO models
from ultralytics import YOLO
```

As discussed above, the two most important libraries we will be using are
openCV and ultralytics. Numpy will also be used pretty heavily for array type
operations (as images are really just arrays of pixel values).

We can then jump into our class definition, `__init__`, and some supporting functions:

```python
class YoloV8ImageObjectDetection:

    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.50):
        """Initializes a yolov8 detector

        Arguments:
            model_path (str): A path to a pretrained model file or one on torchub
            conf_threshold (float): Confidence threshold for detections

        Default Model Supports The Following:

        {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 
            4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 
            8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 
            11: 'stop sign', 12: 'parking meter', 13: 'bench', 
            14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 
            18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 
            22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 
            26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 
            30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 
            34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 
            37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 
            41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 
            47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 
            52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 
            57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 
            61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 
            66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 
            70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 
            75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 
            79: 'toothbrush'
        }      
        """
        self.conf_threshold = conf_threshold
        self.model = self._load_model(model_path)
        self.device = self._get_device()
        self.classes = self.model.names

    def _load_model(self, model_path):
        """Loads Yolo8 model from pytorch hub or a path on disk

        Arguments:
            model_path (str):  A path to a pretrained model file or one on torchub
        Returns:
            model (Model) - Trained Pytorch model
        """
        model = YOLO(model_path)
        return model
    
    def _get_device(self):
        """Gets best device for your system

        Returns:
            device (str): The device to use for YOLO for your system
        """
        if platform.system().lower() == "darwin":
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
```

Our `__init__` function takes two parameters:

1. `model_path` - This will be the path to a pretrained model (checkpoint file). 
            If you use the default, we will just load the pretrained one from
            torchhub
2. `conf_threshold` - This will be our confidence threshold for detections. For example,
                   if our confidence is 0.5, it means that our model will only show 
                   and annotate detections in an image that have a 50% or higher
                   confidence. Anything lower will be ignored.

Our `__init__` function then loads our model by instantiating a new `YOLO` object
with the `model_path` parameter. It then uses platform detection to see if either `mps`
or `cuda` are available on your system. Either of those will be much faster than the default
`cpu`.

## Running