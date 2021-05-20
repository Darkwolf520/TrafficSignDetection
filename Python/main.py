import MyInputHandler
import PreProcessing
import time
import cv2
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.



def print_hi(name):

    """
    pre_processing.segment_colors()
    #pre_processing.detect_circle(pre_processing.image)
    """
    inputHandler = MyInputHandler.InputHandler()
    pre_processing = PreProcessing.PreProcessing(inputHandler.image)
    cap = cv2.VideoCapture("Test/video.mp4")

    while cap.isOpened():
        t = time.time()
        ret, frame = cap.read()
        pre_processing.image=frame
        pre_processing.segment_colors()
        print(time.time() - t)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    print(time.time()-t)
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
