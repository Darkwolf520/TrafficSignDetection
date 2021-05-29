import imutils
import numpy as np
import cv2
import enum
from DomainModels import *
import Models
import time


# Todo refaktorlálás értelmetlen metódusok törlése
# Todo contours metódusok shape detection refaktorálás
# Todo obj simítással/szűréssel kísérletezés szín szegmentálás után
# Todo párhuzamosítás, 3 szín külön párhuzamosítható (bottleneck a neurális hálónál?)
# Todo ha párhuzamosítás színenként, akkor színenként külön metódusok
# Todo színek szegmentálásának finomhangolása
# Todo fals obj tovább jut a zajszűrőn, külön SVM vagy új keras model egy háttér classal
# Todo új modellek létrehozása, kísérletezés, tanítás validálással és teszteléssel
# Todo új modellnél GPU kompatibilis architektra kutatása, készítése.


class PreProcessing:
    def __init__(self):
        self.modelHandler = Models.ModelHandler()

    def detect(self, output_obj, is_show=False):

        image = output_obj.original.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        output_obj.gray = gray.copy()

        red_image, blue_image, yellow_image = self.segment_colors(image.copy(), is_show=is_show)
        output_obj.red_mask = red_image.copy()
        output_obj.blue_mask = blue_image.copy()
        output_obj.yellow_mask = yellow_image.copy()

        red_circles = self.detect_circle(red_image)
        blue_circles = self.detect_circle(blue_image)
        output_obj.red_circles = self.create_circle_image(image.copy(), red_circles)
        output_obj.blue_circles = self.create_circle_image(image.copy(), blue_circles)


        red_cnts = self.detect_contour(red_image)
        blue_cnts = self.detect_contour(blue_image)
        yellow_cnts = self.detect_contour(yellow_image)

        output_obj.red_contours = self.draw_contours(red_cnts, image.copy())
        output_obj.blue_contours = self.draw_contours(blue_cnts, image.copy())
        output_obj.yellow_contours = self.draw_contours(yellow_cnts, image.copy())


        red_shapes = self.classify_and_recognise_contours(red_cnts, Colors.red)
        blue_shapes = self.classify_and_recognise_contours(blue_cnts, Colors.blue)
        yellow_shapes = self.classify_and_recognise_contours(yellow_cnts, Colors.yellow)


        blue_shapes += blue_circles
        test_image = image.copy()



        for o in blue_shapes:
            self.crop_image(o, test_image)
        for o in yellow_shapes:
            self.crop_image(o, test_image)
        for o in red_shapes:
            self.crop_image(o, test_image)
        shapes = []
        shapes += blue_shapes
        shapes += red_shapes
        shapes += yellow_shapes

        output_obj.objects = shapes

        start = time.time()
        for o in shapes:
            self.recognise_object(o)

        print("Recognision time: {0} for {1} objects".format(time.time()-start, len(shapes)))
        output_obj.detected = self.show_results(shapes, image.copy())
        return output_obj


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
        return circles_objects

    def create_circle_image(self, image, circles):

        if circles is not None:
            for circle in circles:
                cv2.circle(image, (circle.area[0], circle.area[1]), circle.area[2], (0, 255, 0), 2)
        return image

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

        return shape

    def draw_and_show_contours(self, c, image):
        image = self.draw_contours(c, image)
        self.imshow("contours", image)

    def draw_contours(self, cnts, image):
        for c in cnts:
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
            area = cv2.contourArea(c)
            if not self.is_noise(c, shape, color):
                if color == Colors.red:
                    if shape == Shapes.triangle or shape == Shapes.circle or shape == Shapes.octagon:
                        o = Sign(c, shape, Colors.red)
                        result_array.append(o)
                elif color == Colors.blue:
                    if shape == Shapes.square or shape == Shapes.circle:
                        o = Sign(c, shape, Colors.blue)
                        result_array.append(o)
                elif color == Colors.yellow:
                    if shape == Shapes.square:
                        o = Sign(c, shape, Colors.yellow)
                        result_array.append(o)
            else:
                o = Sign(area, Shapes.noise, color)
                result_array.append(o)

        return result_array

    def is_noise(self, c, shape, color):
        ratio = 0.75
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area < 30:
            return True
        if color == Colors.red and shape == Shapes.square:
            return True
        if color == Colors.blue and (shape == Shapes.triangle or shape == Shapes.octagon):
            return True
        if color == Colors.yellow and (shape == Shapes.circle or shape == Shapes.triangle or shape == Shapes.octagon):
            return True
        if w > h and h/w < ratio:
            return True
        if h > w and w/h < ratio:
            return True

        return False

    def crop_image(self, o, image):
        if o.shape == Shapes.triangle or o.shape == Shapes.square or o.shape == Shapes.octagon:
            tmp_c = o.area
            x, y, w, h = cv2.boundingRect(tmp_c)
            o.image = image[y:y + h, x: x+w]
            o.coord_top_left = (x,y)
            o.coord_bottom_right = (x+w, y+h)

        elif o.shape == Shapes.circle:
            h_up = int(o.area[0] - o.area[2])
            h_bottom = int(o.area[0] + o.area[2])
            w_left = int(o.area[1] - o.area[2])
            w_right = int(o.area[1] + o.area[2])
            o.image = image[w_left: w_right, h_up: h_bottom]
            o.coord_top_left = (h_up, w_left)
            o.coord_bottom_right = (h_bottom, w_right)
        elif o.shape == Shapes.noise:
            pass
        else:
            raise ValueError('Should not exist here (error with shape recongision or noise detection)')

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

    def recognise_object(self, sign):
        if sign.shape != Shapes.noise:
            sign.sign_class_name = self.modelHandler.predict(sign.image)
            #height, width, c = sign.image.shape
            #if height != 0 and width !=0:
        else:
            sign.sign_class_name = self.modelHandler.get_noise_class()

    def show_results(self, objects, image):
        for o in objects:
            if o.shape == Shapes.noise:
                continue
            else:
                if o.shape == Shapes.triangle or o.shape == Shapes.square or o.shape == Shapes.octagon:
                    image = self.draw_bb(o, image)

                elif o.shape == Shapes.circle:
                    image = self.draw_bb(o, image)

                else:
                    raise ValueError("van más shape is?")

        return image

    def draw_bb(self, o, image):
        cv2.rectangle(image, o.coord_top_left, o.coord_bottom_right, (0, 255, 0), thickness=1)
        x = o.coord_top_left[0]
        y = o.coord_bottom_right[1]
        y -= 20
        cv2.putText(image, o.sign_class_name, (x, y)
                    , cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), thickness=1)
        return image
