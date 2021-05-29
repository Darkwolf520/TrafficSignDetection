import MyInputHandler
import PreProcessing
import time
import cv2
import Models
from OutputHandler import Output

def test_image():
    image = cv2.imread("Test/maps_wess.png")
    pre_processing = PreProcessing.PreProcessing()
    img = pre_processing.detect(image, is_show=True)
    cv2.imshow("result", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def test_video():
    """






    self.red_contours = np.empty((0, 0))
    self.blue_contours = np.empty((0, 0))
    self.yellow_contours = np.empty((0, 0))
    self.detected = np.empty((0, 0))
    self.objects = []
    """
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
        t = time.time()
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))
        output_obj = Output(frame)
        output_obj = pre_processing.detect(output_obj)

        output_obj.show_output_frames(original=show_original, gray=show_gray, red_mask=show_red_mask,
                                      blue_mask=show_blue_mask, yellow_mask=show_yellow_mask, red_circles=show_red_circles,
                                      blue_circles=show_blue_circles, red_contours=show_red_contours, blue_contours=show_blue_contours,
                                      yellow_contours=show_yellow_contours, detected=show_result, objects=show_objects)

        key = cv2.waitKey(1)
        
        if key == ord('q'):
            break
        elif key == ord('p'):
            print("paused")
            cv2.waitKey()
            print("resumed")
        end = time.time()
        print("FPS: {0}, frame-execution: {1}".format(round(1/(end-t), 2), end-t))


if __name__ == '__main__':
    test_video()
