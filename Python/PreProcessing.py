import imutils
import numpy as np
from DomainModels import *
import Models
import time
from concurrent.futures import ThreadPoolExecutor


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
        self.up_ratio = 0.25
        self.bottom_ratio = 0.3
        self.apply_roi = True

    def detect(self, output_obj):
        image = output_obj.original.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        shapes = []

        #color segmentation
        red_image = self.segment_colors(image.copy(), Colors.red)
        blue_image = self.segment_colors(image.copy(), Colors.blue)
        yellow_image = self.segment_colors(image.copy(), Colors.yellow)

        #contour detection by color masks
        red_cnts = self.detect_contour(red_image)
        blue_cnts = self.detect_contour(blue_image)
        yellow_cnts = self.detect_contour(yellow_image)

        #shape recognition
        red_shapes = self.get_shapes_from_contours(red_cnts, Colors.red)
        blue_shapes = self.get_shapes_from_contours(blue_cnts, Colors.blue)
        yellow_shapes = self.get_shapes_from_contours(yellow_cnts, Colors.yellow)

        #circle detection by masks
        red_circles = self.detect_circle(red_image, Colors.red)
        blue_circles = self.detect_circle(blue_image, Colors.blue)

        shapes += red_circles
        shapes += blue_circles
        shapes += red_shapes
        shapes += blue_shapes
        shapes += yellow_shapes

        for o in shapes:
            self.crop_image(o, image.copy())

        start = time.time()
        for o in shapes:
            self.recognise_object(o)
        print("Recognision time: {0}".format(time.time()-start))

        output_obj.gray = gray.copy()
        output_obj.red_mask = red_image.copy()
        output_obj.blue_mask = blue_image.copy()
        output_obj.yellow_mask = yellow_image.copy()
        output_obj.red_circles = self.create_circle_image(image.copy(), red_circles)
        output_obj.blue_circles = self.create_circle_image(image.copy(), blue_circles)
        output_obj.red_contours = self.draw_contours(red_cnts, image.copy())
        output_obj.blue_contours = self.draw_contours(blue_cnts, image.copy())
        output_obj.yellow_contours = self.draw_contours(yellow_cnts, image.copy())
        output_obj.objects = shapes
        output_obj.detected = self.show_results(shapes, image.copy())

        return output_obj

    def multithreading_detection(self, output_obj):
        frame = output_obj.original.copy()

        #image = self.create_roi_image(frame.copy())
        image = frame.copy()
        if self.apply_roi:
            image = self.create_roi_image(frame.copy())

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        shapes = []

        ex1 = ThreadPoolExecutor().submit(self.detect_with_thread, image.copy(), output_obj, Colors.red)
        ex2 = ThreadPoolExecutor().submit(self.detect_with_thread, image.copy(), output_obj, Colors.blue)
        ex3 = ThreadPoolExecutor().submit(self.detect_with_thread, image.copy(), output_obj, Colors.yellow)
        r1 = ex1.result()
        r2 = ex2.result()
        r3 = ex3.result()
        shapes += r1
        shapes += r2
        shapes += r3

        start = time.time()
        for o in shapes:
            self.recognise_object(o)
        print("Recognision time: {0}".format(time.time() - start))

        output_obj.gray = gray
        output_obj.objects = shapes
        output_obj.detected = self.show_results(shapes, frame.copy())

        return output_obj

    def detect_with_thread(self, image, output_obj, color):
        circles = []
        image_orig = image.copy()
        image = cv2.bilateralFilter(image, 3, 10, 10)
        color_mask_orig = self.segment_colors(image.copy(), color)
        color_mask = self.filter_mask(color_mask_orig.copy())
        cnts = self.detect_contour(color_mask)
        color_shapes = self.get_shapes_from_contours(cnts, color)
        if color == Colors.red or color == Colors.blue:
            circles = self.detect_circle(color_mask.copy(), color)
            color_shapes += circles

        for o in color_shapes:
            self.crop_image(o, image_orig)

        if color == Colors.red:
            output_obj.red_mask = color_mask_orig
            output_obj.red_mask_filter = color_mask
            output_obj.red_circles = self.create_circle_image(image.copy(), circles)
            output_obj.red_contours = self.draw_contours(cnts, image.copy())

        if color == Colors.blue:
            output_obj.blue_mask = color_mask_orig
            output_obj.blue_mask_filter = color_mask
            output_obj.blue_circles = self.create_circle_image(image.copy(), circles)
            output_obj.blue_contours = self.draw_contours(cnts, image.copy())

        if color == Colors.yellow:
            output_obj.yellow_mask = color_mask_orig
            output_obj.yellow_mask_filter = color_mask
            output_obj.yellow_contours = self.draw_contours(cnts, image.copy())

        return color_shapes

    def create_roi_image(self, image):
        h, w, c = image.shape
        up = int(h* self.up_ratio)
        bottom = int(h * (1-self.bottom_ratio))
        #o.image = image[y:y + h, x: x+w]
        """
               image[0: up, 0:w] = 0
        image[bottom:h, 0:w] = 0
        
        """

        image = image[up:bottom, 0:w]
        return image



    def detect_circle(self, image, color):
        circles_objects = []
        minDist = 100
        param1 = 500
        param2 = 20
        minRadius = 5
        maxRadius = 100

        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                                   minRadius=minRadius, maxRadius=maxRadius)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                o = Sign(i, Shapes.circle, color)
                circles_objects.append(o)
        return circles_objects

    def create_circle_image(self, image, circles):
        if circles is not None:
            for circle in circles:
                cv2.circle(image, (circle.area[0], circle.area[1]), circle.area[2], (0, 255, 0), 2)
        return image

    def recognise_shape_from_contour(self, c):
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

    def draw_contours(self, cnts, image):
        for c in cnts:
            cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        return image

    def detect_contour(self, image):
        cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        return cnts

    def get_shapes_from_contours(self, cnts, color):
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

    def segment_colors(self, image, color):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if color == Colors.red:
            color_lower = np.array([160, 50, 50])
            color_upper = np.array([180, 255, 255])
            color_lower2 = np.array([0, 100, 50])
            color_upper2 = np.array([10, 255, 255])
            mask = cv2.inRange(hsv_image, color_lower, color_upper)
            mask2 = cv2.inRange(hsv_image, color_lower2, color_upper2)
            mask = cv2.bitwise_or(mask, mask2)
            return mask

        if color == Colors.blue:
            color_lower = np.array([95, 100, 100])
            color_upper = np.array([110, 255, 255])
            mask = cv2.inRange(hsv_image, color_lower, color_upper)
            return mask

        if color == Colors.yellow:
            color_lower = np.array([10, 120, 120])
            color_upper = np.array([30, 255, 255])
            mask = cv2.inRange(hsv_image, color_lower, color_upper)
            return mask

        raise ValueError("Unable to segment color: {0}".format(color))

    def filter_mask(self, image):
        kernel = np.ones((5, 5), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel) #opening nem vált be
        return image

    def recognise_object(self, sign):
        if sign.shape != Shapes.noise:
            sign.sign_class_name = self.modelHandler.predict(sign.image)
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
        h, w, c = image.shape
        if self.apply_roi:
            y_dif = int( h * self.up_ratio)
            x = o.coord_top_left[0]
            y = y_dif + o.coord_top_left[1] - 2
            x2 = o.coord_bottom_right[0]
            y2 = y_dif + o.coord_bottom_right[1]
            cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), thickness=1)
            y -= 2
            cv2.putText(image, o.sign_class_name, (x, y)
                        , cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), thickness=1)
        else:
            cv2.rectangle(image, o.coord_top_left, o.coord_bottom_right, (0, 255, 0), thickness=1)
            x = o.coord_top_left[0]
            y = o.coord_top_left[1] - 2
            cv2.putText(image, o.sign_class_name, (x, y)
                        , cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), thickness=1)
        return image

    def imshow(self, title, image):
        cv2.imshow(title, image)
        cv2.waitKey()
        cv2.destroyAllWindows()
