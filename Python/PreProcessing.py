import imutils
import numpy as np
import cv2
import enum
from DomainModels import *


class PreProcessing:
    def __init__(self):
        pass

    def detect(self, image, is_show=False):
        self.imshow("image", image)
        red_image, blue_image, yellow_image = self.segment_colors(image.copy(), is_show=is_show)
        self.detect_circle(red_image, is_show=is_show, test_image=image.copy())

        red_cnts = self.detect_contour(red_image)
        blue_cnts = self.detect_contour(blue_image)
        yellow_cnts = self.detect_contour(yellow_image)

        red_shapes = self.classify_and_recognise_contours(red_cnts, Colors.red)
        blue_shapes = self.classify_and_recognise_contours(blue_cnts, Colors.blue)
        blue_circles = self.detect_circle(blue_image, is_show=is_show, test_image=image.copy())
        blue_shapes += blue_circles
        yellow_shapes = self.classify_and_recognise_contours(yellow_cnts, Colors.yellow)
        test_image = image.copy()
        for o in blue_circles:
            self.crop_image(o, test_image)
            o.imshow()

    def segment_colors(self, image, is_show=False):
        red_image = self.segment_red_color(image)
        blue_image = self.segment_blue_color(image)
        yellow_image = self.segment_yellow_color(image)
        if is_show:
            self.imshow("red mask", red_image)
            self.imshow("blue mask", blue_image)
            self.imshow("yellow mask", yellow_image)

        return red_image, blue_image, yellow_image

    def detect_circle(self, image, is_show=False, test_image=np.empty((1, 1)), color=Colors.blue, shape=Shapes.circle):
        circles_objects = []
        minDist = 100
        param1 = 500  # 500
        param2 = 20  # 200 #smaller value-> more false circles
        minRadius = 5
        maxRadius = 100  # 10

        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                                   minRadius=minRadius, maxRadius=maxRadius)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                o = Sign(i, shape, color)
                circles_objects.append(o)
                if is_show:
                    cv2.circle(test_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            if is_show:
                self.imshow("detected circles", test_image)
        return circles_objects

    def recognise_shape_from_contour(self, c, is_show=False, test_image=np.empty((1, 1))):
        shape = Shapes.undefined
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) == 3:
            shape = Shapes.triangle
        elif len(approx) == 4:
            shape = Shapes.square
        elif len(approx) == 8:
            shape = Shapes.octagon
        else:
            shape = Shapes.noise
        if is_show and shape == Shapes.triangle:
            self.draw_and_show_contours(c, test_image)
        return shape

    def draw_and_show_contours(self, c, image):
        image = self.draw_contours(c, image)
        self.imshow("contours", image)

    def draw_contours(self, c, image):
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        return image

    def detect_contour(self, image):
        cnts= cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        return cnts

    def classify_and_recognise_contours(self, cnts, color):
        result_array = []

        for c in cnts:
            shape = self.recognise_shape_from_contour(c)
            if color == Colors.red:
                if shape == Shapes.triangle or shape == Shapes.circle or shape == Shapes.octagon:
                    o = Sign(c, shape, Colors.red)
                    result_array.append(o)
            if color == Colors.blue:
                if shape == Shapes.square or shape == Shapes.circle:
                    o = Sign(c, shape, Colors.blue)
                    result_array.append(o)
            if color == Colors.yellow:
                if shape == Shapes.square:
                    o = Sign(c, shape, Colors.yellow)
                    result_array.append(o)

        return result_array

    def crop_image(self, o, image):
        if o.shape == Shapes.square or o.shape == Shapes.triangle:
            tmp_c = o.area
            x, y, w, h = cv2.boundingRect(tmp_c)
            o.image = image[y:y + h, x: x+w]
        if o.shape == Shapes.circle:
            h_up = int(o.area[0] - o.area[2])
            h_bottom = int(o.area[0] + o.area[2])
            w_left = int(o.area[1] - o.area[2])
            w_right = int(o.area[1] + o.area[2])
            o.image = image[w_left: w_right, h_up: h_bottom]

    def segment_red_color(self, image):
        color_lower = np.array([160, 50, 50])
        color_upper = np.array([180, 255, 255])
        color_lower2 = np.array([0, 100, 20])
        color_upper2 = np.array([10, 255, 255])
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, color_lower, color_upper)
        mask2 = cv2.inRange(hsv_image, color_lower2, color_upper2)
        mask = cv2.bitwise_or(mask, mask2)
        return mask

    def segment_blue_color(self, image):
        color_lower = np.array([95, 100, 100])
        color_upper = np.array([110, 255, 255])
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, color_lower, color_upper)
        return mask

    def segment_yellow_color(self, image):
        color_lower = np.array([20,100,100])
        color_upper = np.array([30, 255, 255])
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, color_lower, color_upper)
        return mask

    def imshow(self, title, image):
        cv2.imshow(title, image)
        cv2.waitKey()
        cv2.destroyAllWindows()