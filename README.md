# Computer Vision For All - Object Tracking With YOLO and Python

## Introduction
In the ever-evolving realm of computer vision and artificial intelligence, 
object tracking stands as a pivotal concept, serving as the linchpin to 
countless applications, from autonomous vehicles to surveillance systems and 
beyond. One remarkable tool that has garnered significant attention and acclaim 
in recent years is YOLOv8, a state-of-the-art real-time object detection framework. 
In this blog post, we embark on a journey through the fascinating world of YOLOv8 
object tracking, unveiling a project that not only showcases the prowess of this 
cutting-edge technology but also adds a layer of intelligence by dissecting key 
statistics of tracked objects.

We're about to dive into the intricacies of a Python-based project where we 
harness the capabilities of YOLOv8 to achieve highly accurate and real-time object 
tracking. But we're not stopping there. Our journey extends to the next level by 
dissecting the tracked objects' movements, measuring distances traveled in pixels, 
and calculating average speeds. The outcome? A comprehensive understanding of not 
just what's in the frame, but how these objects behave and interact with their surroundings.

Whether you're a computer vision enthusiast, a developer seeking to add object 
tracking to your applications, or simply someone intrigued by the magic of AI, 
this blog post is designed to inspire and educate. Join us as we explore YOLOv8's 
potential, delve into the technical intricacies of object tracking, and reveal how to 
extract valuable insights about the motion of tracked objects. By the end, you'll be 
equipped with the knowledge and tools to implement your very own object tracking solutions,
while also unlocking a deeper understanding of the dynamic world within your videos and images.

## Libraries and References
We are going to be using two main libraries to build our object tracker:

1. [OpenCV](https://opencv.org/) - For opening video streams, drawing on our frames, etc.
2. [Ultralytics](https://www.ultralytics.com/) - For performing the object detection and tracking with YOLO

If you're new to computer vision, OpenCV is definitely a great resource to get aquanted with.
It is an open-source computer vision and image processing software library designed to facilitate computer vision tasks. It provides a vast array of tools and functions for image and video analysis, manipulation, and processing, making it invaluable for a wide range of applications, including object detection, facial recognition, image stitching, and motion tracking. OpenCV is highly popular due to its versatility, efficiency, and ease of use, making it a go-to choice for researchers, developers, and engineers in fields such as robotics, machine learning, augmented reality, and computer vision, enabling them to harness the power of computer vision for various real-world applications. Its extensive community support, cross-platform compatibility, and continuous development have contributed to its widespread adoption in both academia and industry.

I've recently also stumbled upon Ultralytics, and have fallen a bit in love with it.
Ultralytics is an open-source software framework primarily focused on computer vision and deep learning. It is designed to streamline the development of object detection, image classification, and other machine learning tasks by providing easy-to-use APIs and pre-configured models. Ultralytics is especially popular among researchers and developers in the computer vision and machine learning communities due to its user-friendly nature, extensive documentation, and strong integration with PyTorch, one of the leading deep learning frameworks. With its comprehensive toolkit, it simplifies complex tasks like training and deploying neural networks for various applications, including autonomous vehicles, surveillance systems, and medical image analysis, contributing to its rising popularity as an essential resource in the field of deep learning and computer vision.

We use some other supporting libraries, such as numpy, and a full set of requirements
can be found [here]().

## Building

## Running