import keras

import PreProcessing
import time
import cv2
from OutputHandler import Output, TrackingStates
import glob
from createBackground import Annotation
from createBackground import createFileName
import os
import json

def test_all_GTSDB_images():
    default_path = "D:/FullIJCNN2013"
    annotations = create_annotation_list()
    IOU_list = []
    for filename in glob.glob(default_path + '/*.ppm'):
        tmp = filename.split("\\")
        img_name = tmp[len(tmp)-1]
        ann_cords_list = getAnnotationCordsForImage(img_name, annotations)
        output_obj = test_image(img_path=filename, ann_cords_list=ann_cords_list)
        IOU_list = IOU_list + output_obj.IOU_list
    avg = round(sum(IOU_list) / len(IOU_list))
    print('AVG IOU: {0}'.format(avg))

def create_annotation_list():
    ann_path = "D:/FullIJCNN2013/gt.txt"
    annotation_list = []
    f = open(ann_path, "r")
    for line in f:
        annotation = Annotation(line)
        annotation_list.append(annotation)
    f.close()
    return annotation_list

def getAnnotationCordsForImage(filename, annotation_objects_list):
    result = []
    for annotation in annotation_objects_list:
        if annotation.image_name == filename:
            result.append(annotation.coords)

    return result


def test_image(resolution = (0, 0), img_path="Test/maps_istvan.png", saveResult=False, ann_cords_list = []):
    image = cv2.imread(img_path)
    if resolution != (0, 0):
        image = cv2.resize(image, resolution)
    output_obj = Output(image, image.copy())
    pre_processing = PreProcessing.PreProcessing()
    #output_obj = pre_processing.detect(output_obj)
    if len(ann_cords_list) > 0:
        output_obj = pre_processing.multithreading_detection(output_obj, annotation_list=ann_cords_list)
    else:
        output_obj = pre_processing.multithreading_detection(output_obj)

    output_obj.show_output_frames(original=show_original, gray=show_gray, red_mask=show_red_mask,
                                  blue_mask=show_blue_mask, yellow_mask=show_yellow_mask,
                                  red_mask_filter=show_red_mask_filter,
                                  blue_mask_filter=show_blue_mask_filter,
                                  yellow_mask_filter=show_yellow_mask_filter,
                                  red_circles=show_red_circles,
                                  blue_circles=show_blue_circles, red_contours=show_red_contours,
                                  blue_contours=show_blue_contours,
                                  yellow_contours=show_yellow_contours, detected=show_result,
                                  objects=show_all_objects, recognised_objects=show_recognised_objects)

    if saveResult:
        tmp = img_path.split("/")
        tmp = tmp[len(tmp) - 1]
        img_name = "Results/"
        img_name += tmp[0:len(tmp) - 4]
        img_name += "_result.jpg"
        cv2.imwrite(img_name, output_obj.detected_img)


    cv2.waitKey()
    cv2.destroyAllWindows()
    return output_obj




def test_video(resolution=(0, 0), video="Test/video.mp4", multithreading=True, isTracking= True, annotation_path=None):
    speedUp = 1
    speedUpCounter = 0
    slowmo = 1
    pre_processing = PreProcessing.PreProcessing()
    cap = cv2.VideoCapture(video)
    tracker_list = []
    fps_list =[]
    frame_list = []
    tracking_time = []
    frame_counter = 0
    Annotation = None
    IOU_values = []
    annotation_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret and speedUp == speedUpCounter + 1:
            speedUpCounter = 0
            t = time.time()
            if resolution != (0, 0):
                frame = cv2.resize(frame, resolution)
            edited_img = frame.copy()
            if isTracking:
                for obj in tracker_list:
                    top_left, bottom_right = obj.getActualCoordsInTLBRFormat()
                    edited_img = subtract_trackable_object_from_image(edited_img.copy(), top_left, bottom_right)
            output_obj = Output(frame, edited_img)
            if annotation_path != None:
                file = createFileName(annotation_path, frame_counter, extension=".json")
                Annotation = JsonAnnotation(file)
                if Annotation.exist:
                    output_obj.annotations = Annotation.annotations

            output_obj = pre_processing.multithreading_detection(output_obj)
            if annotation_path != None:
                for item in output_obj.IOU_list:
                    IOU_values.append(item)
                for item in Annotation.annotations:
                    cv2.rectangle(output_obj.detected_img, item.top_left, item.bottom_right, (255,255,255), 1)

            if isTracking:
                tracker_list = update_tracking_list(tracker_list, output_obj.trackable_objects)
            key = cv2.waitKey(slowmo)
            if key == ord('q'):
                break
            elif key == ord('s'):
                if slowmo == 1000:
                    slowmo = 1
                else:
                    slowmo = 1000
            elif key == ord('p'):
                print("paused")
                cv2.waitKey()
                print("resumed")
            elif key == ord('+'):
                speedUp *=2
                speedUpCounter = 0
            elif key == ord('-'):
                if speedUp != 1:
                    speedUp /= 2
                    speedUpCounter = 0
            if isTracking:
                if len(tracker_list) != 0:
                    for trackingObj in tracker_list:
                        start = time.time()
                        trackingObj.tracking(frame)
                        tracking_time.append(time.time() - start)
                    update_tracking_list(tracker_list)
                    for trackingObj in tracker_list:
                        top_left, bottom_right = trackingObj.getActualCoordsInTLBRFormat()
                        output_obj.draw_tracking_obj_on_detected_V2(top_left, bottom_right, trackingObj.sign.sign_class_name)
            output_obj.show_output_frames(original=show_original, gray=show_gray, red_mask=show_red_mask,
                                          blue_mask=show_blue_mask, yellow_mask=show_yellow_mask,
                                          red_mask_filter=show_red_mask_filter,
                                          blue_mask_filter=show_blue_mask_filter,
                                          yellow_mask_filter=show_yellow_mask_filter,
                                          red_circles=show_red_circles,
                                          blue_circles=show_blue_circles, red_contours=show_red_contours,
                                          blue_contours=show_blue_contours,
                                          yellow_contours=show_yellow_contours, detected=show_result,
                                          objects=show_all_objects, recognised_objects=show_recognised_objects)

            end = time.time()
            fps = round(1/(end-t), 2)
            #print("FPS: {0}, frame-execution: {1}".format(fps, end-t))
            fps_list.append(fps)
            if annotation_path != None:
                frame_counter += 1
        elif not ret:
            cv2.destroyAllWindows()
            print("video ended")
            avg = round(sum(fps_list) / len(fps_list), 2)
            print("AVG fps: {0}".format(avg))
            if annotation_path != None:
                avg_IOU = round(sum(IOU_values) / len(IOU_values), 5)
                print("AVG IOU: {0}".format(avg_IOU))
            if isTracking:
                avg = round(sum(tracking_time) / len(tracking_time), 5)
                print("AVG tracking time: {0}".format(avg))
            break

        else:
            speedUpCounter += 1


def update_tracking_list(prev_list, new_list = None):
    max_item = 5
    result = []
    prev_list_selected = []
    #merge the two lists

    for item in prev_list:
        if item.state != TrackingStates.lost:
            prev_list_selected.append(item)

    if new_list != None:
        result = prev_list_selected + new_list
    else:
        for item in prev_list_selected:
            result.append(item)
    if len(result) > max_item:
        for i in range(len(result)-max_item):
            del result[0]
    return result

def subtract_trackable_object_from_image(image, top_left, bottom_right):
    image[top_left[1]:bottom_right[1], top_left[0]: bottom_right[0]] = (0, 0, 0)
    return image



def createImagesFromVideo(videoPath, outputRootPath):
    print("started")
    counter = 0
    tmp_name = os.path.basename(videoPath)
    name = tmp_name.split('.')[0]
    outputPath = outputRootPath + name + "/"
    if not os.path.isdir(outputPath):
        os.mkdir(outputPath)

    cap = cv2.VideoCapture(videoPath)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            filename = createFileName(outputPath, counter, extension=".jpg")
            cv2.imwrite(filename, frame)
            counter += 1
            if counter % 1000 == 0:
                print("Still working on it")
        else:
            break
    print("DONE")


def handleJsonFile(file):
    tmp = JsonAnnotation(file)

class JsonAnnotation:
    def __init__(self, file):
        self.exist = False
        self.annotations = []
        try:
            f = open(file, "r")
            self.exist = True
        except:
            pass
        if self.exist:
            f = open(file, "r")
            annotationParentList = json.load(f)[0]['annotations']
            for annDict in annotationParentList:
                self.annotations.append(AnnotationClass(annDict))


class AnnotationClass:
    def __init__(self, dict):
        self.className = dict['label']
        xy = dict['coordinates']
        self.width = int(xy['width'])
        self.height = int(xy['height'])
        self.x1 = int(xy['x']) - int(self.width / 2)
        self.y1 = int(xy['y']) - int(self.width / 2)
        self.x2 = self.x1 + self.width
        self.y2 = self.y1 + self.height
        self.coords = (self.x1, self.y1, self.x2, self.y2)
        self.top_left = (self.x1, self.y1)
        self.bottom_right = (self.x2, self.y2)


show_original = True
show_gray = False
show_red_mask = True
show_blue_mask = True
show_yellow_mask = False
show_red_mask_filter = True
show_blue_mask_filter = False
show_yellow_mask_filter = False
show_red_circles = False
show_blue_circles = False
show_red_contours = True
show_blue_contours = False
show_yellow_contours = False
show_result = True
show_all_objects = False
show_recognised_objects = False


if __name__ == '__main__':
    resolution = (0, 0)
    #resolution = (1280, 720)

    #test_all_GTSDB_images()
    #test_image(resolution=resolution, img_path="../Assets/images/curve_left/tmp.jpg")
    #test_image(resolution=resolution, img_path="D:/FullIJCNN2013/00002.ppm")
    file = "SH19"
    test_video(resolution=resolution, isTracking=False, video="../Assets/{0}.mp4".format(file))
    #createImagesFromVideo("../Assets/SH14.mp4", "D:/test_set/")
    #handler = Models.ModelHandler()
    #handler.evaulate()
    



#handler = Models.ModelHandler(createNewModel=True)
#handler.create_model4("MobileNetV2T3")
#handler.train_model("MobileNetV2T2")

#handler.create_mobile_64("MobileNetV2Input64")

"""


"""


"""

#test_select_ROI(resolution=resolution, image_path="Test/maps_istvan.png")
#test_select_ROI(video_path="Test/video.mp4")
    bbox_dict={
        1: (153, 367, 32, 30),
        2: (727, 415, 25, 23),
        3: (771, 448, 18, 23),
        4: (737, 391, 31, 30),
        5: (760, 485, 23, 30),
        6: (717, 433, 35, 31),
        7: (712, 407, 36, 29)
    }

    bbox_dict_bigger={
        1: (149, 364, 41, 38),
        2: (724, 411, 30, 28),
        3: (765, 444, 30, 30),
        4: (732, 385, 40, 40),
        5: (759, 482, 26, 34),
        6: (712, 431, 43, 38),
        7: (707, 401, 48, 40)
    }
#01 (153, 367, 32, 30) kisebb
#01 (149, 364, 41, 38) nagyobb
#02 (727, 415, 25, 23) kisebb
#02 (724, 411, 30, 28) nagyobb
#03 (771, 448, 18, 23) kisebb
#03 (765, 444, 30, 30) nagyobb
#04 (737, 391, 31, 30) kidebb
#04 (732, 385, 40, 40) nagyobb
#05 (760, 485, 23, 30) kisebb
#05 (759, 482, 26, 34) nagyobb
#06 (717, 433, 35, 31) kisebb park
#06 (712, 431, 43, 38) nagyobb park
#07 (712, 407, 36, 29) kisebb egyirany
#07 (707, 401, 48, 40) nagyobb park

#handler = Models.ModelHandler()

#handler.create_model4("MobileNetV2")
#handler.train_model("MobileNetV2")

#0.03  ennél nagyobb ne legyen egy objektum
#createBackground.createBackgroundsFromDataset("D:/FullIJCNN2013/", "D:/GTSRB/Final_Training/Images/00043/")


#createBackground.createBackgroundsFromDataset("D:/FullIJCNN2013/", "D:/GTSRB/Final_Training/Images/00043/")




model = keras.models.Sequential()
model.add(MobileNetV2(weights='imagenet'))
model.summary()
img_path = "D:/STOP_sign(1).jpg"
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])

for i in range(3):
    s = time.time()
    model.predict(x)
    print(time.time()-s)



    for i in range(1,8):
        for j in range (3):
            tracker_name, tracked_frames, total_frames, avg = testing_tracking_algorithm(j, "../Assets/" + str(i) + ".mp4", bbox_dict.get(i))
            print("Video #{0}, Algirithm {1} tracked {2}/{3} frames, avg processing time: {4}".format(i, tracker_name, tracked_frames, total_frames, avg))
            tracker_name, tracked_frames, total_frames, avg = testing_tracking_algorithm(j, "../Assets/" + str(i) + ".mp4", bbox_dict_bigger.get(i))
            print("Video #{0}, Algirithm {1} tracked {2}/{3} frames, avg processing time: {4}".format(i, tracker_name, tracked_frames, total_frames, avg))


    test_image(resolution=resolution)
    #test_video(resolution=resolution)
    #testing_with_second_monitor(resolution=resolution)
    
    test_video(resolution=resolution, multithreading=True)
    start = time.time()
    test_video(resolution=resolution, multithreading=True)
    end = time.time()
    print(end-start)
    
    #3szor fut le, első futási eredményt nem jelenítem meg, eredmények a 3. futás végén jelenítődnek meg
    compare_running_times(resolution=resolution)
    """


"""

def compare_running_times(resolution= (0, 0)):
    #első futás, nem számít
    test_video(resolution=resolution, multithreading=False)
    # test
    m_start = time.time()
    test_video(resolution=resolution, multithreading=False)
    m_end = time.time()

    start = time.time()
    test_video(resolution=resolution, multithreading=True)
    end= time.time()
    print("Single thread: {0}".format(end - start))
    print("Multithreading: {0}".format(m_end - m_start))


def testing_tracking_algorithm(tracker_num, video_path, bbox):
    tracker = cv2.TrackerMOSSE_create()
    tracker_name = "MOSSE"
    if tracker_num == 0:
        tracker = cv2.TrackerCSRT_create()
        tracker_name = "CSRT"
    elif tracker_num == 1:
        tracker = cv2.TrackerKCF_create()
        tracker_name = "KCF"


    cap = cv2.VideoCapture(video_path)
    fps_array = []
    tracked_frames = 0
    i = -1
    avg = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            start = time.time()
            if i == -1:
                tracker.init(frame, bbox)
                i = i +1

            else:
                i = i + 1
                (success, bbox) = tracker.update(frame)

                if success:
                    frame = draw_tracking_obj_on_detected(frame.copy(), bbox)
                    tracked_frames = tracked_frames + 1
            cv2.imshow("frame", frame)
            cv2.waitKey(1)
            end = time.time()
            fps = round(1 / (end - start))
            fps_array.append(fps)
        else:
            avg = sum(fps_array)/len(fps_array)
            break
    cap.release()
    return tracker_name, tracked_frames, i, avg

def draw_tracking_obj_on_detected(image, bbox):
    bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    top_left = (bbox[0], bbox[1])
    bottom_right = (bbox[0]+bbox[2], bbox[1]+bbox[3])
    cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 1)
    return image
    
def test_select_ROI(resolution=(0, 0), video_path=None, image_path=None, top_roi_rate=0.25, bottom_roi_rate=0.25):
    bbox = 0
    if video_path != None:
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if (resolution != (0, 0)):
                    frame = cv2.resize(frame, resolution)
                h, w, c = frame.shape
                up = int(h * top_roi_rate)
                bottom = int(h * (1 - bottom_roi_rate))

                frame = frame[up:bottom, 0:w]
                h, w, c = frame.shape
                bbox = cv2.selectROI("select roi", frame)
                cv2.destroyWindow("select roi")
                print(bbox)
                print("area: {0}".format(bbox[2]*bbox[3]))
                print((bbox[2]*bbox[3])/(h*w))
               # cv2.imshow("frame", frame)
                #cv2.waitKey()
               # cv2.destroyAllWindows()
    if image_path != None:
        img = cv2.imread(image_path)
        img = cv2.resize(img, resolution)
        h, w, c = img.shape
        up = int(h * top_roi_rate)
        bottom = int(h * (1 - bottom_roi_rate))

        img = img[up:bottom, 0:w]
        h, w, c = img.shape
        bbox = cv2.selectROI("select roi", img)
        cv2.destroyWindow("select roi")
        print(bbox)
        print("area: {0}".format(bbox[2] * bbox[3]))
        print((bbox[2] * bbox[3]) / (h * w))
        cv2.imshow("frame", img)
        cv2.waitKey()
        cv2.destroyAllWindows()

def convert_RGB2HSV(RGB):
    tmp = np.uint8([[[RGB[0], RGB[1], RGB[2]]]])
    hsv_color = cv2.cvtColor(tmp, cv2.COLOR_RGB2HSV)
    print(hsv_color)"""