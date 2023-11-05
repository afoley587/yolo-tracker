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


class YoloV8ImageObjectDetection:
    PATH        = os.environ.get("YOLO_WEIGHTS_PATH", "yolov8n.pt")    # Path to a model. yolov8n.pt means download from PyTorch Hub
    CONF_THRESH = float(os.environ.get("YOLO_CONF_THRESHOLD", "0.50")) # Confidence threshold

    def __init__(self, looking_for):
        """Initializes a yolov8 detector

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
        self.model = self._load_model()
        self.device = self._get_device()
        self.classes = self.model.names

    def _load_model(self):
        """Loads Yolo8 model from pytorch hub or a path on disk

        Returns:
            model (Model) - Trained Pytorch model
        """
        model = YOLO(YoloV8ImageObjectDetection.PATH)
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

    def is_detectable(self, classname):
        looking_for = list(self.classes.keys())[list(self.classes.values()).index(classname)]

        if (looking_for < 0):
            return False
        
        return True

    def detect(self, frame, looking_for):
        """Analyze a frame using a YOLOv8 model to find any of the classes
        in question

        Arguments:
            frame (numpy.ndarray): The frame to analyze (from cv2)
        
        Returns:
            plotted (numpy.ndarray): Frame with bounding boxes and labels ploted on it.
            boxes (torch.Tensor): A set of bounding boxes
            tracks (list): A list of box IDs
        """
        results = self.model.track(frame, persist=True, conf=YoloV8ImageObjectDetection.CONF_THRESH, classes = [looking_for])

        plotted = results[0].plot()
        boxes   = results[0].boxes.xywh.cpu()
        tracks  = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id else []
        return plotted, boxes, tracks 