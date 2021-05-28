import MyInputHandler
import PreProcessing
import time
import cv2
import Models


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
        detected_frame = pre_processing.detect(frame)
        cv2.imshow('frame', frame)
        cv2.imshow('detected', detected_frame)
        if cv2.waitKey(1) == ord('q'):
            break
        end = time.time()
        print("FPS: {0}, frame-execution: {1}".format(round(1/(end-t), 2), end-t))


if __name__ == '__main__':
    test_video()
