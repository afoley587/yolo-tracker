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

## Running