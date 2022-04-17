import cv2 as cv
from modules.util import clean_folders

from config import single_path as folder


dispW=1020
dispH=768
flip=2
camSet1='nvarguscamerasrc sensor-id=0 wbmode=1 ! video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1,format=NV12 ! nvvidconv flip-method=0 ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=True'
camSet2='nvarguscamerasrc sensor-id=1 wbmode=1 ! video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1,format=NV12 ! nvvidconv flip-method=2 ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=True'

################################################################################
cap_left = cv.VideoCapture(camSet1)
cap_right = cv.VideoCapture(camSet2)
################################################################################



text_title="Calibration Assistant"
text_progress=""
capture_taken=0

capture_qty=40
clean_folders([folder])


while(True):
    if capture_taken==0:
        text_progress="Press 'C' to capture the images"
    elif capture_taken==capture_qty:
        text_progress="Done, hold 'Esc' to exit."
    else:
        text_progress="Stay " + str(capture_qty-capture_taken)+ " image(s) to take."

    _, left_frame_color = cap_left.read()
    _, right_frame_color = cap_right.read()

    # left_frame = cv.cvtColor(left_frame_color, cv.COLOR_BGR2GRAY)
    # right_frame = cv.cvtColor(right_frame_color, cv.COLOR_BGR2GRAY)

    if cv.waitKey(1) == ord('c'):
        if capture_taken<capture_qty:
            capture_taken = capture_taken + 1
            text_progress=""
            filename_l="{}left{:03d}".format(folder,capture_taken) + ".jpg"
            filename_r="{}right{:03d}".format(folder,capture_taken) + ".jpg"
            cv.imwrite(filename_l, left_frame_color)
            cv.imwrite(filename_r, right_frame_color)
            
    left_view = left_frame_color #cv.resize(left_frame_color,(480,270))
    right_view = right_frame_color #cv.resize(right_frame_color,(480,270))

    left_view=cv.flip(left_view,1)
    right_view=cv.flip(right_view,1)

    cv.putText(left_view, text_progress, (60, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv.putText(right_view, text_progress, (60, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv.imshow(text_title + ' Camera Left',left_view)
    cv.imshow(text_title + ' Camera Right',right_view)

    if cv.waitKey(1) == 27:
        break
    
cap_left.release()
cap_right.release()
cv.destroyAllWindows()


