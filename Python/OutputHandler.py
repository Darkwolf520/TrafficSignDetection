import DomainModels
import cv2
import numpy as np
from DomainModels import *
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

class Output:
    def __init__(self, image, edited_image):
        self.frame = image
        self.edited_frame = edited_image
        self.gray_img = np.empty((0, 0))
        self.red_mask_img = np.empty((0, 0))
        self.blue_mask_img = np.empty((0, 0))
        self.yellow_mask_img = np.empty((0, 0))
        self.red_mask_filter_img = np.empty((0, 0))
        self.blue_mask_filter_img = np.empty((0, 0))
        self.yellow_mask_filter_img = np.empty((0, 0))
        self.red_circles_objects = np.empty((0, 0))
        self.blue_circles_objects = np.empty((0, 0))
        self.red_contours_img = np.empty((0, 0))
        self.blue_contours_img = np.empty((0, 0))
        self.yellow_contours_img = np.empty((0, 0))
        self.detected_img = np.empty((0, 0))
        self.all_objects = []
        self.trackable_objects = []
        
    def get_not_noise_objects(self):
        objects = []
        for o in self.all_objects:
            if o.shape != Shapes.noise:
                objects.append(o)
        return objects

    def get_color_objects(self, color):
        objects=[]
        for o in self.all_objects:
            if o.color == color:
                objects.append(o)
        return objects

    def draw_tracking_obj_on_detected(self, bbox):
        bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        top_left = (bbox[0], bbox[1])
        bottom_right = (bbox[0]+bbox[2], bbox[1]+bbox[3])
        cv2.rectangle(self.detected_img, top_left, bottom_right, (0, 0, 255), 1)

    def draw_tracking_obj_on_detected_V2(self, top_left, bottom_right,sign_name):
        cv2.rectangle(self.detected_img, top_left, bottom_right, (0, 255, 0), 1)
        x = top_left[0]
        y = top_left[1]
        y -= 2
        cv2.putText(self.detected_img, sign_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), thickness=1)


    def show_output_frames(self, original=False, gray=False, red_mask=False,
                           blue_mask=False, yellow_mask=False, red_circles=False,
                           red_mask_filter=False, blue_mask_filter=False, yellow_mask_filter=False,
                           blue_circles=False, red_contours=False, blue_contours=False,
                           yellow_contours=False, detected=False, objects=False, recognised_objects=False):
        if original:
            cv2.imshow("frame", self.frame)
        if gray:
            cv2.imshow("gray", self.gray_img)
        if red_mask:
            cv2.imshow("red mask", self.red_mask_img)
        if blue_mask:
            cv2.imshow("blue mask", self.blue_mask_img)
        if yellow_mask:
            cv2.imshow("yellow mask", self.yellow_mask_img)
        if red_mask_filter:
            cv2.imshow("red mask filtered", self.red_mask_filter_img)
        if blue_mask_filter:
            cv2.imshow("blue mask filtered", self.blue_mask_filter_img)
        if yellow_mask_filter:
            cv2.imshow("yellow mask filtered", self.yellow_mask_filter_img)
        if red_circles:
            cv2.imshow("red circles", self.red_circles_objects)
        if blue_circles:
            cv2.imshow("blue circles", self.blue_circles_objects)
        if red_contours:
            cv2.imshow("red contours", self.red_contours_img)
        if blue_contours:
            cv2.imshow("blue contours", self.blue_contours_img)
        if yellow_contours:
            cv2.imshow("yellow contours", self.yellow_contours_img)
        if detected:
            cv2.imshow("detected objects", self.detected_img)
        if objects:
            print("Red objects: {0}, Blue objects: {1}, Yellow object(s) {2}, All sign object: {3}"
                  .format(len(self.get_color_objects(Colors.red)), len(self.get_color_objects(Colors.blue)),
                          len(self.get_color_objects(Colors.yellow)), len(self.get_not_noise_objects())))
        if recognised_objects:
            print("There are {0} recognised objects on the frame".format(len(self.trackable_objects)))


