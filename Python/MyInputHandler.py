import cv2
import numpy as np


class InputHandler:
    def __init__(self):
        self.test_img_path="Test/maps_istvan.png"
        self.image = cv2.imread(self.test_img_path)


    def load_test_image(self):
        self.image = cv2.imread(self.test_img_path)
        cv2.imshow("test", self.image)
        cv2.waitKey()