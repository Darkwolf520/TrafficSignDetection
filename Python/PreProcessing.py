import numpy as np
import cv2

class PreProcessing:
    def __init__(self, image):
        self.image = image

    def segment_colors(self, is_show=False):
        red_image = self.segment_red_color()
        blue_image= self.segment_blue_color()
        yellow_image = self.segment_yellow_color()
        if is_show:
            self.imshow("red mask", red_image)
            self.imshow("blue mask", blue_image)
            self.imshow("yellow mask", yellow_image)



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