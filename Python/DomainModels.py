import enum
import cv2
import numpy as np
import pandas as pd
from openpyxl import load_workbook

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
        self.sign_class_id = None
        self.top3 = []

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
class predictionObject:
    def __init__(self, groundTruth, predicted, top3):
        self.groundTruth = groundTruth
        self.predicted = predicted
        self.top3 = top3
        self.isCorrectClass = self.groundTruth == self.predicted
        self.isCorrectCategory = False
        self.isCorrectTop3 = False
        if (0 <= groundTruth <= 9 or 14 <= groundTruth <= 15) and (0 <= predicted <= 9 or 14 <= predicted <= 15):
            self.isCorrectCategory = True
        elif (groundTruth == 10 or 17 <= groundTruth <= 30) and (predicted == 10 or 17 <= predicted <= 30):
            self.isCorrectCategory = True
        elif (31 <= groundTruth <= 38) and (31 <= predicted <= 38):
            self.isCorrectCategory = True
        elif (11 <= groundTruth <= 13 or groundTruth == 16) and (11 <= predicted <= 13 or predicted == 16):
            self.isCorrectCategory = True
        if top3[0] == groundTruth or top3[1] == groundTruth or top3[2] == groundTruth:
            self.isCorrectTop3 = True


class predictedResults:
    def __init__(self, video_num):
        self.predictions = []
        self.matrix = np.zeros((39, 39))
        self.video_num = video_num
        self.top3_list = []
        self.top1_list = []
        self.category_list = []

    def savePred(self, img, pred, top3):
        cv2.imshow("detected_obj", img)
        exit = False
        key = ""
        isSign = False
        class_num = -1
        while not exit:
            try:
                print("is traffic sign? enter/n")
                key = cv2.waitKey()
                if key == 13:
                    class_num = int(input("GroundTruth: "))
                    isSign = True
                    print('saved')
                    exit = True
                elif key == ord("n"):
                    exit = True
                    print('saved')
                else:
                    raise()
            except:
                pass
        cv2.destroyWindow("detected_obj")
        if isSign:
            self.addPred(class_num, pred, top3)

    def addPred(self, groundTruth, predicted, top3):
        val = 0
        pred = predictionObject(groundTruth, predicted, top3)

        if pred.isCorrectClass:
            val = 1
        self.top1_list.append(val)
        val = 0
        if pred.isCorrectCategory:
            val = 1
        self.category_list.append(val)
        val = 0
        if pred.isCorrectTop3:
            val = 1
        self.top3_list.append(val)




        self.predictions.append(pred)

        self.matrix[predicted, groundTruth] += 1

    def __del__(self):
        ok = False
        while not ok:
            try:
                df1 = pd.DataFrame(self.matrix)
                avg1 = str(round(sum(self.top1_list) / len(self.top1_list), 4) * 100) + "%"
                avg3 = str(round(sum(self.top3_list) / len(self.top3_list), 4) * 100) + "%"
                avg_cat = str(round(sum(self.category_list) / len(self.category_list), 4) * 100) + "%"

                book = load_workbook('Results/matrix.xlsx')
                writer = pd.ExcelWriter('Results/matrix.xlsx', engine='openpyxl')
                writer.book = book

                writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
                df1.to_excel(writer, str(self.video_num))

                df1 = pd.DataFrame([[avg1, avg3, avg_cat]])
                df1.to_excel(writer, str(self.video_num), startrow=42)

                writer.save()
                ok = True
            except:
                input('press enter to retry save matrix')


        #df1 = pd.DataFrame([[avg1, avg3, avg_cat]])
        #df1.to_excel("Results/matrix.xlsx", sheet_name=self.video_num, startrow=42, startcol=0)
        #print()

    #def write_matrix(self):



class TrackingStates(enum.Enum):
    init = 0,
    tracking = 1,
    lost = 2
