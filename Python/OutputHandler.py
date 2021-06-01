import DomainModels
import cv2
import numpy as np
from DomainModels import *


class Output:
    def __init__(self, image):
        self.original = image
        self.gray = np.empty((0, 0))
        self.red_mask = np.empty((0, 0))
        self.blue_mask = np.empty((0, 0))
        self.yellow_mask = np.empty((0, 0))
        self.red_mask_filter = np.empty((0, 0))
        self.blue_mask_filter = np.empty((0, 0))
        self.yellow_mask_filter = np.empty((0, 0))
        self.red_circles = np.empty((0, 0))
        self.blue_circles = np.empty((0, 0))
        self.red_contours = np.empty((0, 0))
        self.blue_contours = np.empty((0, 0))
        self.yellow_contours = np.empty((0, 0))
        self.detected = np.empty((0, 0))
        self.objects = []
        
    def get_not_noise_objects(self):
        objects = []
        for o in self.objects:
            if o.shape != Shapes.noise:
                objects.append(o)
        return objects

    def get_color_objects(self, color):
        objects=[]
        for o in self.objects:
            if o.color == color:
                objects.append(o)
        return objects

    def draw_tracking_obj_on_detected(self, bbox):
        bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        top_left = (bbox[0], bbox[1])
        bottom_right = (bbox[0]+bbox[2], bbox[1]+bbox[3])
        cv2.rectangle(self.detected, top_left, bottom_right, (0, 0, 255), 1)

    def show_output_frames(self, original=False, gray=False, red_mask=False,
                           blue_mask=False, yellow_mask=False, red_circles=False,
                           red_mask_filter=False, blue_mask_filter=False, yellow_mask_filter=False,
                           blue_circles=False, red_contours=False, blue_contours=False,
                           yellow_contours=False, detected=False, objects=False):
        if original:
            cv2.imshow("frame", self.original)
        if gray:
            cv2.imshow("gray", self.gray)
        if red_mask:
            cv2.imshow("red mask", self.red_mask)
        if blue_mask:
            cv2.imshow("blue mask", self.blue_mask)
        if yellow_mask:
            cv2.imshow("yellow mask", self.yellow_mask)
        if red_mask_filter:
            cv2.imshow("red mask filtered", self.red_mask_filter)
        if blue_mask_filter:
            cv2.imshow("blue mask filtered", self.blue_mask_filter)
        if yellow_mask_filter:
            cv2.imshow("yellow mask filtered", self.yellow_mask_filter)
        if red_circles:
            cv2.imshow("red circles", self.red_circles)
        if blue_circles:
            cv2.imshow("blue circles", self.blue_circles)
        if red_contours:
            cv2.imshow("red contours", self.red_contours)
        if blue_contours:
            cv2.imshow("blue contours", self.blue_contours)
        if yellow_contours:
            cv2.imshow("yellow contours", self.yellow_contours)
        if detected:
            cv2.imshow("detected objects", self.detected)
        if objects:
            print("Red objects: {0}, Blue objects: {1}, Yellow objects {2}, All sign object: {3}"
                  .format(len(self.get_color_objects(Colors.red)), len(self.get_color_objects(Colors.blue)),
                          len(self.get_color_objects(Colors.yellow)), len(self.get_not_noise_objects())))


