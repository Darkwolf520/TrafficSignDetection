import PreProcessing
import time
import cv2
import numpy as np
import Models
from OutputHandler import Output
from mss import mss

show_original = True
show_gray = False
show_red_mask = True
show_blue_mask = True
show_yellow_mask = True
show_red_mask_filter = False
show_blue_mask_filter = False
show_yellow_mask_filter = False
show_red_circles = False
show_blue_circles = False
show_red_contours = False
show_blue_contours = False
show_yellow_contours = False
show_result = True
show_objects = True

def test_image(resolution = (0, 0)):
    image = cv2.imread("Test/maps_istvan.png")
    if resolution != (0, 0):
        image = cv2.resize(image, resolution)
    output_obj = Output(image)
    pre_processing = PreProcessing.PreProcessing()
    #output_obj = pre_processing.detect(output_obj)
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
                                  objects=show_objects)
    cv2.waitKey()
    cv2.destroyAllWindows()


def test_video(resolution=(0, 0), multithreading=True):

    pre_processing = PreProcessing.PreProcessing()
    cap = cv2.VideoCapture("Test/video.mp4")
    bbox = (0, 0, 0, 0)
    tracking = False
    tracker = cv2.TrackerCSRT_create()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            t = time.time()
            if resolution != (0, 0):
                frame = cv2.resize(frame, resolution)
            output_obj = Output(frame)
            if multithreading:
                output_obj = pre_processing.multithreading_detection(output_obj)
            else:
                output_obj = pre_processing.detect(output_obj)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('p'):
                print("paused")
                cv2.waitKey()
                print("resumed")
            elif key == ord("t"):
                bbox = cv2.selectROI("select roi", output_obj.original)
                cv2.destroyWindow("select roi")
                if bbox[2] > 5 and bbox[3] > 5:
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(output_obj.original, bbox)
                    tracking = True

            if tracking:
                (success, bbox) = tracker.update(output_obj.original)
                if success:
                    output_obj.draw_tracking_obj_on_detected(bbox)
                else:
                    tracking = False

            output_obj.show_output_frames(original=show_original, gray=show_gray, red_mask=show_red_mask,
                                          blue_mask=show_blue_mask, yellow_mask=show_yellow_mask,
                                          red_mask_filter=show_red_mask_filter,
                                          blue_mask_filter=show_blue_mask_filter,
                                          yellow_mask_filter=show_yellow_mask_filter,
                                          red_circles=show_red_circles,
                                          blue_circles=show_blue_circles, red_contours=show_red_contours,
                                          blue_contours=show_blue_contours,
                                          yellow_contours=show_yellow_contours, detected=show_result,
                                          objects=show_objects)

            end = time.time()
            print("FPS: {0}, frame-execution: {1}".format(round(1/(end-t), 2), end-t))
        else:
            cv2.destroyAllWindows()
            print("video ended")
            break


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

def testing_with_second_monitor(resolution= (0, 0)):
    pre_processing = PreProcessing.PreProcessing()
    sct = mss()
    mon = sct.monitors[0]
    mon = {'left':0,
           'top':0,
           'width': 1920,
           'height':1080}
    while True:
        img = sct.grab(mon)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)
        if resolution != (0, 0):
            img = cv2.resize(img, resolution)

        t = time.time()
        output_obj = Output(img)
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
                                      objects=show_objects)
        end = time.time()
        print("FPS: {0}, frame-execution: {1}".format(round(1 / (end - t), 2), end - t))

        key = cv2.waitKey(1)
        if key == ord("q"):
            cv2.destroyAllWindows()
            break


def testing_CSRT(tracker, video_path, bbox):
    tracker = cv2.TrackerCSRT_create()
    cap = cv2.VideoCapture(video_path)
    tracker = cv2.TrackerCSRT_create()
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            start = time.time()
            if i == 0:
                tracker.init(frame, bbox)

            else:
                (success, bbox) = tracker.update(frame)

                if success:
                    frame = draw_tracking_obj_on_detected(frame.copy(), bbox)
                else:
                    print("Tracked frames: {0}".format(i))
            cv2.imshow("frame", frame)
            cv2.waitKey(1)
            end = time.time()
            fps = round(1 / (end - start))
            print("processing time: {0}".format(fps))
            i = i + 1
        else:
            break
    cap.release()

def draw_tracking_obj_on_detected(image, bbox):
    bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    top_left = (bbox[0], bbox[1])
    bottom_right = (bbox[0]+bbox[2], bbox[1]+bbox[3])
    cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 1)
    return image
def test_select_ROI(video_path):
    bbox = 0
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if bbox == 0 :
                bbox = cv2.selectROI("select roi", frame)
                cv2.destroyWindow("select roi")
                print(bbox)
                break
            cv2.imshow("frame", frame)
            cv2.waitKey()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    resolution = (1360, 800)


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
#06 (712, 407, 36, 29) kisebb egyirany
#06 (707, 401, 48, 40) nagyobb park

    testing_CSRT(cv2.TrackerCSRT_create(), "../Assets/01.mp4", (153, 367, 32, 30))
    testing_CSRT(cv2.TrackerKCF_create(), "../Assets/01.mp4", (153, 367, 32, 30))
    testing_CSRT(cv2.TrackerMOSSE_create(), "../Assets/01.mp4", (153, 367, 32, 30))

"""
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