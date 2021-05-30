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
        self.image = np.empty(0)
        self.shape = shape
        self.color = color
        self.coord_top_left= (0, 0)
        self.coord_bottom_right = (0, 0)
        self.sign_class_name = ""

    def get_bbox(self):
        #bbox = (x, y, width_from_coord, height_from_coord)
        x = self.coord_top_left[0]
        y = self.coord_top_left[1]

        w = self.coord_bottom_right[0] - x
        h = self.coord_bottom_right[1] - y
        bbox = (x, y, w, h)
        return bbox


    def imshow(self):
        if len(self.image.shape) != 1:
            width, height, c = self.image.shape
            print("Height: {0}, Width: {1}".format(height, width ))
            cv2.imshow("Sign object {0}, {1} ".format(self.shape, self.color), self.image)
            cv2.waitKey()
            cv2.destroyAllWindows()
        else:
            print("Image not found")