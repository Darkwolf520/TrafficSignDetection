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
    pre_processing.segment_colors()
    #pre_processing.detect_circle(pre_processing.image)
    """
    pre_processing = PreProcessing.PreProcessing()
    cap = cv2.VideoCapture("Test/video.mp4")

    while cap.isOpened():
        t = time.time()
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))
        output_obj = Output(frame)
        output_obj = pre_processing.detect(output_obj)
        """
        self.original = image
        self.gray = np.empty((0, 0))
        self.red_mask = np.empty((0, 0))
        self.blue_mask = np.empty((0, 0))
        self.yellow_mask = np.empty((0, 0))
        self.red_circles = np.empty((0, 0))
        self.blue_circles = np.empty((0, 0))
        self.red_contours = np.empty((0, 0))
        self.blue_contours = np.empty((0, 0))
        self.yellow_contours = np.empty((0, 0))
        self.detected = np.empty((0, 0))
        self.objects = []
        """
        output_obj.show_output_frames(original=True, gray=True, red_mask=True, blue_mask=True,
                                      yellow_mask=True, red_circles=True, blue_circles=True,
                                      red_contours=True, blue_contours=True, yellow_contours=True,
                                      detected=True, objects=True)

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
