import imutils
import numpy as np
from DomainModels import *
import Models
import time
from concurrent.futures import ThreadPoolExecutor




# Todo obj simítással/szűréssel kísérletezés szín szegmentálás után
# Todo színek szegmentálásának finomhangolása
# Todo fals obj tovább jut a zajszűrőn, külön SVM vagy új keras model egy háttér classal
# Todo új modellek létrehozása, kísérletezés, tanítás validálással és teszteléssel
# Todo új modellnél GPU kompatibilis architektra kutatása, készítése.


class PreProcessing:
    def __init__(self):
        self.modelHandler = Models.ModelHandler()
        self.up_ratio = 0.25
        self.bottom_ratio = 0.25
        self.apply_roi = True
        self.next_id = 0
        self.edited_max_area_ratio = 0.03
        self.edited_min_area_ratio = 0.0001
        self.edited_image_width = 0
        self.edited_image_height = 0
        self.edited_image_area = 0
        self.original_image_height = 0
        self.original_image_width = 0

    def detect(self, output_obj):
        image = output_obj.frame.copy()
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

        output_obj.gray_img = gray.copy()
        output_obj.red_mask_img = red_image.copy()
        output_obj.blue_mask_img = blue_image.copy()
        output_obj.yellow_mask_img = yellow_image.copy()
        output_obj.red_circles_objects = self.create_circle_image(image.copy(), red_circles)
        output_obj.blue_circles_objects = self.create_circle_image(image.copy(), blue_circles)
        output_obj.red_contours_img = self.draw_contours(red_cnts, image.copy())
        output_obj.blue_contours_img = self.draw_contours(blue_cnts, image.copy())
        output_obj.yellow_contours_img = self.draw_contours(yellow_cnts, image.copy())
        output_obj.all_objects = shapes
        output_obj.detected_img = self.show_results(shapes, image.copy())

        return output_obj

    def multithreading_detection(self, output_obj):
        frame = output_obj.frame.copy()

        image = output_obj.edited_frame.copy()
        h, w, c = image.shape
        self.original_image_height = h
        self.original_image_width = w
        self.edited_image_width = w
        self.edited_image_height = h
        self.edited_image_area = h * w
        #image = frame.copy()
        if self.apply_roi:
            image = self.create_roi_image(image.copy())
            h, w, c = image.shape
            self.edited_image_width = w
            self.edited_image_height = h
            self.edited_image_area = h * w
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

        output_obj.gray_img = gray
        output_obj.all_objects = shapes
        output_obj.detected_img = self.show_results(shapes, frame.copy())
        for o in shapes:
            if o.shape != Shapes.noise and o.sign_class_name != 'Noise':
                trackable_object = TrackableSign(self.next_id, o)
                self.next_id += 1
                output_obj.trackable_objects.append(trackable_object)


        return output_obj

    def detect_with_thread(self, image, output_obj, color):
        circles = []
        image_orig = image.copy()
        image = cv2.bilateralFilter(image, 3, 10, 10)
        color_mask_orig = self.segment_colors(image.copy(), color)
        #color_mask = self.segment_colors(image.copy(), color)
        color_mask = self.filter_mask(color_mask_orig.copy())
        cnts = []
        color_shapes = []
        color_circles = []
        if color == Colors.red or color == Colors.blue:
            circles = self.detect_circle(color_mask.copy(), color)
            for circle in circles:
                if not self.is_noise_circle(circle.area):
                    color_circles.append(circle)
        if color != Colors.blue:
            cnts = self.detect_contour(color_mask)
            color_shapes += self.get_shapes_from_contours(cnts, color)

        for item in color_circles:
            self.define_coords(item)
        color_circles = self.remove_noise_objects(color_circles)

        for item in color_shapes:
            self.define_coords(item)
        color_shapes = self.remove_noise_objects(color_shapes)

        color_circles = self.remove_redundant_circles(color_circles, color_shapes)
        color_shapes += color_circles
        color_shapes = self.remove_noise_objects(color_shapes)

        for o in color_shapes:
            self.crop_image(o, image_orig)

        if color == Colors.red:
            output_obj.red_mask_img = color_mask_orig
            output_obj.red_mask_filter_img = color_mask
            output_obj.red_circles_objects = self.create_circle_image(image.copy(), circles)
            output_obj.red_contours_img = self.draw_contours(cnts, image.copy())

        if color == Colors.blue:
            output_obj.blue_mask_img = color_mask_orig
            output_obj.blue_mask_filter_img = color_mask
            output_obj.blue_circles_objects = self.create_circle_image(image.copy(), circles)
           # output_obj.blue_contours_img = self.draw_contours(cnts, image.copy())

        if color == Colors.yellow:
            output_obj.yellow_mask_img = color_mask_orig
            output_obj.yellow_mask_filter_img = color_mask
            output_obj.yellow_contours_img = self.draw_contours(cnts, image.copy())

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
        minDist = 80
        dp = 1
        param1 = 100
        param2 = 20
        minRadius = 5
        maxRadius = 150

        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp, minDist, param1=param1, param2=param2,
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

    def is_noise_circle(self, circle):
        if int(circle[2]* circle[2] *3.14)/self.edited_image_area > self.edited_max_area_ratio:
            print(int(circle[2]* circle[2] *3.14)/self.edited_image_area)
            return True

        if int(circle[2]* circle[2] *3.14)/self.edited_image_area < self.edited_min_area_ratio:
            return True
        return False

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
        if shape == Shapes.square and color != Colors.yellow:
            return True
        if area < 30:
            return True
        if shape == Shapes.noise:
            return True
        if (area/self.edited_image_area) > self.edited_max_area_ratio:
            print(area/self.edited_image_area)
            return True
        if (area/self.edited_image_area) < self.edited_min_area_ratio:
            print((area/self.edited_image_area))
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

    def define_coords(self, o):
        bias = 5
        if o.color == Colors.yellow:
            bias = 25
        if o.shape == Shapes.circle:
            h_up = int(o.area[1] - o.area[2]) - bias
            h_bottom = int(o.area[1] + o.area[2]) + bias
            w_left = int(o.area[0] - o.area[2]) - bias
            w_right = int(o.area[0] + o.area[2]) + bias
            o.coord_top_left = (w_left, h_up)
            o.coord_bottom_right = (w_right, h_bottom)

            if w_right - w_left < 0 or h_bottom - h_up < 0:
                o.shape = Shapes.noise

        elif o.shape == Shapes.triangle or o.shape == Shapes.square or o.shape == Shapes.octagon:
            tmp_c = o.area
            x, y, w, h = cv2.boundingRect(tmp_c)
            x -= bias
            y -= bias
            w += bias *2
            h += bias *2

            if h == 0 or w == 0:
                o.shape = Shapes.noise

            o.coord_top_left = (x, y)
            o.coord_bottom_right = (x + w, y + h)
        elif o.shape == Shapes.noise:
            pass
    def crop_image(self, o, image):
        bias = 5
        y_dif = int(self.original_image_height * self.up_ratio)
        if o.shape == Shapes.triangle or o.shape == Shapes.square or o.shape == Shapes.octagon:
            x = o.coord_top_left[0]
            y = o.coord_top_left[1]
            x2 = o.coord_bottom_right[0]
            y2 = o.coord_bottom_right[1]
            o.image = image[y: y2, x: x2]


            h, w, c = o.image.shape
            if h == 0 or w == 0:
                o.shape = Shapes.noise
                print("contour crop failed")

        elif o.shape == Shapes.circle:
            x = o.coord_top_left[0]
            y = o.coord_top_left[1]
            x2 = o.coord_bottom_right[0]
            y2 = o.coord_bottom_right[1]

            o.image = image[y: y2, x: x2]


            h, w, c = o.image.shape
            if h == 0 or w == 0:
                o.shape = Shapes.noise
                print("circle crop failed")
        elif o.shape == Shapes.noise:
            pass
        else:
            raise ValueError('Should not exist here (error with shape recongision or noise detection)')


    def segment_colors(self, image, color):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if color == Colors.red:
            color_lower = np.array([160, 70, 70])
            color_upper = np.array([180, 255, 255])
            color_lower2 = np.array([0, 70, 70])
            color_upper2 = np.array([15, 255, 255])
            mask = cv2.inRange(hsv_image, color_lower, color_upper)
            mask2 = cv2.inRange(hsv_image, color_lower2, color_upper2)
            mask = cv2.bitwise_or(mask, mask2)
            return mask

        if color == Colors.blue:
            color_lower = np.array([95, 100, 100])
            color_upper = np.array([125, 255, 255])
            mask = cv2.inRange(hsv_image, color_lower, color_upper)
            return mask

        if color == Colors.yellow:
            color_lower = np.array([15, 160, 160])
            color_upper = np.array([30, 255, 255])
            mask = cv2.inRange(hsv_image, color_lower, color_upper)
            return mask

        raise ValueError("Unable to segment color: {0}".format(color))

    def filter_mask(self, image):
        kernel = np.ones((5, 5), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel) #opening nem vált be
        #image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel) #opening nem vált be

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
        color = (0, 0, 255)
        if o.sign_class_name == self.modelHandler.classes.get(len(self.modelHandler.classes) - 1):
            color = (255, 0, 0)
        if self.apply_roi:
            y_dif = int( h * self.up_ratio)
            x = o.coord_top_left[0]
            y = y_dif + o.coord_top_left[1]
            x2 = o.coord_bottom_right[0]
            y2 = y_dif + o.coord_bottom_right[1]
            o.real_coord_top_left = (x, y)
            o.real_coord_bottom_right = (x2, y2)
            #self.real_coord_top_left = (0, 0)
            #self.real_coord_bottom_right = (0, 0)
            cv2.rectangle(image, (x, y), (x2, y2), color, thickness=1)
            y -= 2
            cv2.putText(image, o.sign_class_name, (x, y)
                        , cv2.FONT_HERSHEY_PLAIN, 1, color, thickness=2)
        else:
            cv2.rectangle(image, o.coord_top_left, o.coord_bottom_right, color, thickness=1)
            o.real_coord_bottom_right = o.coord_bottom_right
            o.real_coord_top_left = o.real_coord_top_left
            x = o.coord_top_left[0]
            y = o.coord_top_left[1] - 2
            cv2.putText(image, o.sign_class_name, (x, y)
                        , cv2.FONT_HERSHEY_PLAIN, 1, color, thickness=2)
        """
        cv2.imshow("asd", image)
        cv2.waitKey()
        cv2.destroyWindow("asd")
        """
        return image

    def imshow(self, title, image):
        cv2.imshow(title, image)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def remove_noise_objects(self, color_shapes):
        result = []

        for item in color_shapes:
            if item.shape != Shapes.noise:
                result.append(item)

        return result

    def IOU(self, img1, img2):
        xA = max(img1[0], img2[0])
        yA = max(img1[1], img2[1])
        xB = min(img1[2], img2[2])
        yB = min(img1[3], img2[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (img1[2] - img1[0] + 1) * (img1[3] - img1[1] + 1)
        boxBArea = (img2[2] - img2[0] + 1) * (img2[3] - img2[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou

    def remove_redundant_circles(self, color_circles, color_shapes):
        result = []
        for circle in color_circles:
            redundant = False
            circle_coords = (circle.coord_top_left[0], circle.coord_top_left[1], circle.coord_bottom_right[0],
                             circle.coord_bottom_right[1])
            for shape in color_shapes:

                shape_coords = (shape.coord_top_left[0], shape.coord_top_left[1], shape.coord_bottom_right[0],
                             shape.coord_bottom_right[1])
                if self.IOU(circle_coords, shape_coords) > 0.1:
                    redundant = True
                    break
            if not redundant:
                result.append(circle)
        return result

