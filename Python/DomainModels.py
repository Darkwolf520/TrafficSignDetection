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
        self.sign_class_name = None
        self.real_coord_top_left = (0, 0)
        self.real_coord_bottom_right = (0, 0)

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

class TrackableSign:
    def __init__(self, id, sign):
        self.id = id
        self.sign = sign
        self.state = TrackingStates.init
        self.original_coords = (sign.real_coord_top_left[0], sign.real_coord_top_left[1], sign.real_coord_bottom_right[0] - sign.real_coord_top_left[0], sign.real_coord_bottom_right[1] - sign.real_coord_top_left[1])
        self.actual_coords = (sign.real_coord_top_left[0], sign.real_coord_top_left[1], sign.real_coord_bottom_right[0] - sign.real_coord_top_left[0], sign.real_coord_bottom_right[1] - sign.real_coord_top_left[1])
        self.tracker_algo = ""

    def getActualCoordsInTLBRFormat(self):
        top_left, bottom_right = self.convertCoordsToTLBRFromBbox(self.actual_coords)
        return top_left, bottom_right

    def convertCoordsToBboxFromTLBR(self, top_left, bottom_right):
        result = (top_left[0], top_left[1], bottom_right[0] - top_left[0], bottom_right[1] - top_left[1])
        return result

    def convertCoordsToTLBRFromBbox(self, bbox):
        top_left = (bbox[0] , bbox[1])
        bottom_right = (bbox[0] + bbox[2], bbox[1]+ bbox[3])
        return top_left, bottom_right

    def tracking(self, frame):
        if self.state == TrackingStates.init:
            self.tracker_algo = cv2.TrackerCSRT_create()
            self.tracker_algo.init(frame, self.original_coords)
            self.state = TrackingStates.tracking

        elif self.state == TrackingStates.tracking:
            (success, bbox) = self.tracker_algo.update(frame)
            if success:
                self.actual_coords = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            else:
                self.state = TrackingStates.lost
            return success

class TrackingStates(enum.Enum):
    init = 0,
    tracking = 1,
    lost = 2
