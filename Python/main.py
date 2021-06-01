import PreProcessing
import time
import cv2
import Models
from OutputHandler import Output

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



if __name__ == '__main__':
    resolution = (640, 480)

    #test_image(resolution=resolution)
    test_video(resolution=resolution)
    """
    test_video(resolution=resolution, multithreading=True)
    start = time.time()
    test_video(resolution=resolution, multithreading=True)
    end = time.time()
    print(end-start)
    
    """
    """
    #3szor fut le, első futási eredményt nem jelenítem meg, eredmények a 3. futás végén jelenítődnek meg
    compare_running_times(resolution=resolution)
    """