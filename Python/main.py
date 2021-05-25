import MyInputHandler
import PreProcessing
import time
import cv2


def test_image():
    image = cv2.imread("test.ppm")
    pre_processing = PreProcessing.PreProcessing()
    pre_processing.detect(image, is_show=True)


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
        pre_processing.detect(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
        end = time.time()
        print("FPS: {0}, frame-execution: {1}".format(round(1/(end-t), 2), end-t))


if __name__ == '__main__':
    test_image()

