import MyInputHandler
import PreProcessing
import time
import cv2
import Models
from OutputHandler import Output


def test_image(resolution = (0, 0)):
    show_original = True
    show_gray = False
    show_red_mask = False
    show_blue_mask = False
    show_yellow_mask = False
    show_red_circles = False
    show_blue_circles = False
    show_red_contours = False
    show_blue_contours = False
    show_yellow_contours = False
    show_result = True
    show_objects = True


    image = cv2.imread("Test/maps_wess.png")
    if resolution != (0, 0):
        image = cv2.resize(image, resolution)
    output_obj = Output(image)
    pre_processing = PreProcessing.PreProcessing()
    #output_obj = pre_processing.detect(output_obj)
    output_obj = pre_processing.multithreading_detection(output_obj)

    output_obj.show_output_frames(original=show_original, gray=show_gray, red_mask=show_red_mask,
                                  blue_mask=show_blue_mask, yellow_mask=show_yellow_mask, red_circles=show_red_circles,
                                  blue_circles=show_blue_circles, red_contours=show_red_contours,
                                  blue_contours=show_blue_contours,
                                  yellow_contours=show_yellow_contours, detected=show_result, objects=show_objects)
    cv2.waitKey()
    cv2.destroyAllWindows()


def test_video(resolution=(0, 0), multithreading=False):
    show_original = True
    show_gray = True
    show_red_mask = True
    show_blue_mask = True
    show_yellow_mask = True
    show_red_circles = True
    show_blue_circles = True
    show_red_contours = True
    show_blue_contours = True
    show_yellow_contours = True
    show_result = True
    show_objects = True


    pre_processing = PreProcessing.PreProcessing()
    cap = cv2.VideoCapture("Test/video.mp4")

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

                output_obj.show_output_frames(original=show_original, gray=show_gray, red_mask=show_red_mask,
                                              blue_mask=show_blue_mask, yellow_mask=show_yellow_mask,
                                              red_circles=show_red_circles,
                                              blue_circles=show_blue_circles, red_contours=show_red_contours,
                                              blue_contours=show_blue_contours,
                                              yellow_contours=show_yellow_contours, detected=show_result,
                                              objects=show_objects)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('p'):
                print("paused")
                cv2.waitKey()
                print("resumed")


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
    start = time.time()
    test_video(resolution=resolution, multithreading=False)
    print("Single thread: {0}".format(time.time() - start))

    start = time.time()
    test_video(resolution=resolution, multithreading=True)
    print("Multithreading: {0}".format(time.time() - start))


if __name__ == '__main__':
    resolution = (640, 480)
    #test_image(resolution=resolution)
    test_video(resolution=resolution, multithreading=True)

    """
    #3szor fut le, első futási eredményt nem jelenítem meg, eredmények a 3. futás végén jelenítődnek meg
    compare_running_times(resolution=resolution)
    """