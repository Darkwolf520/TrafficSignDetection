import enum
import cv2
import numpy as np


class Colors(enum.Enum):
    undefined = 0
    red = 1
    blue = 2
    yellow = 3


class Shapes(enum.Enum):
    undefined = 0
    noise = 1
    circle = 2
    triangle = 3
    square = 4
    octagon = 5


class Sign:
    def __init__(self, area, shape, color):
        self.area = area #coords
        self.image = None #croped img
        self.shape = shape
        self.color = color

    def imshow(self):
        if self.image != None:
            cv2.imshow("Sign object {0}, {1{".format(self.shape, self.color))
            cv2.waitKey()
            cv2.destroyAllWindows()
        else:
            print("Image not found")