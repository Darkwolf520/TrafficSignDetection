import MyInputHandler
import PreProcessing
import time
import cv2
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def test_image():
    image=cv2.imread("Test/maps_hungaria.png")
    pre_processing=PreProcessing.PreProcessing()
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
        print(time.time() - t)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    print(time.time()-t)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_video()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
