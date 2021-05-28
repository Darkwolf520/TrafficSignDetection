import DomainModels
import cv2
import numpy as np


class Output:
    def __init__(self, image):
        self.original = image
        self.gray = np.empty((0, 0))
        self.red_mask = np.empty((0, 0))
        self.blue_mask = np.empty((0, 0))
        self.yellow_mask = np.empty((0, 0))
        self.red_circles = np.empty((0, 0))
        self.blue_circles = np.empty((0, 0))
        self.red_contours = np.empty((0, 0))
        self.blue_contours = np.empty((0, 0))
        self.yellow_contours = np.empty((0, 0))
        self.detected = np.empty((0, 0))
        self.objects = []

    def show_output_frames(self, original=False, gray=False, red_mask=False,
                           blue_mask=False, yellow_mask=False, red_circles=False,
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
            pass #Todo normális logolás helye


