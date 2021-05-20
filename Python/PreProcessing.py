import numpy as np
import cv2

class PreProcessing:
    def __init__(self):
        pass

    def detect(self, image, is_show=False):
        red_image, blue_image, yellow_image = self.segment_colors(image, is_show=is_show)
        self.detecct_circle(red_image, is_show=is_show, test_image=image)

    def segment_colors(self, image, is_show=False):
        red_image = self.segment_red_color(image)
        blue_image= self.segment_blue_color(image)
        yellow_image = self.segment_yellow_color(image)
        if is_show:
            self.imshow("red mask", red_image)
            self.imshow("blue mask", blue_image)
            self.imshow("yellow mask", yellow_image)

        return red_image, blue_image, yellow_image

    def detecct_circle(self, image, is_show=False, test_image=np.empty((1,1))):
        minDist = 100
        param1 = 500  # 500
        param2 = 20  # 200 #smaller value-> more false circles
        minRadius = 5
        maxRadius = 100  # 10

        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                                   minRadius=minRadius, maxRadius=maxRadius)

        if is_show:
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    cv2.circle(test_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
                self.imshow("detected circles", test_image)


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