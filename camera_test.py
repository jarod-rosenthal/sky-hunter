import cv2
print('cv2',cv2.__version__)
import numpy as np
print('numpy',np.__version__)
import time

# # =============================================================================
# # Camera settings
# # =============================================================================

flip1=0
flip2=4
dispW=640
dispH=480

camSet1='nvarguscamerasrc sensor-id=0 ee-mode=1 ee-strength=0 tnr-mode=2 tnr-strength=1 wbmode=1 ! video/x-raw(memory:NVMM), width=1920, height=1080, framerate=60/1,format=NV12 ! nvvidconv flip-method='+str(flip1)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=False'
capture1=cv2.VideoCapture(camSet1, cv2.CAP_GSTREAMER)

camSet2='nvarguscamerasrc sensor-id=1 ee-mode=1 ee-strength=0 tnr-mode=2 tnr-strength=1 wbmode=1 ! video/x-raw(memory:NVMM), width=1920, height=1080, framerate=60/1,format=NV12 ! nvvidconv flip-method='+str(flip2)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=False'
capture2 = cv2.VideoCapture(camSet2, cv2.CAP_GSTREAMER)

camSet3=3
capture3 = cv2.VideoCapture(camSet3)
capture3.set(3, 640) # set the resolution
capture3.set(4, 480)
capture3.set(cv2.CAP_PROP_BRIGHTNESS, 48)
capture3.set(cv2.CAP_PROP_CONTRAST, 0)
capture3.set(cv2.CAP_PROP_AUTOFOCUS, 1)

# # =============================================================================
# # Output video stream
# # =============================================================================

w = capture1.get(cv2.CAP_PROP_FRAME_WIDTH)
h = capture1.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = capture1.get(cv2.CAP_PROP_FPS)

gst_out_1 = "appsrc ! video/x-raw, format=BGR ! queue ! videoconvert ! video/x-raw,format=BGRx ! nvvidconv ! omxh264enc insert-sps-pps=true ! h264parse ! rtph264pay pt=96 ! queue ! application/x-rtp, media=video, encoding-name=H264 ! udpsink host=192.168.50.161 port=5000"
rtp_out_1 = cv2.VideoWriter(gst_out_1, cv2.CAP_GSTREAMER, 0, float(fps), (int(w), int(h)))

w = capture2.get(cv2.CAP_PROP_FRAME_WIDTH)
h = capture2.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = capture2.get(cv2.CAP_PROP_FPS)

gst_out_2 = "appsrc ! video/x-raw, format=BGR ! queue ! videoconvert ! video/x-raw,format=BGRx ! nvvidconv ! omxh264enc insert-sps-pps=true ! h264parse ! rtph264pay pt=96 ! queue ! application/x-rtp, media=video, encoding-name=H264 ! udpsink host=192.168.50.161 port=6000"
rtp_out_2 = cv2.VideoWriter(gst_out_2, cv2.CAP_GSTREAMER, 0, float(fps), (int(w), int(h)))

dtFIL=0
font=cv2.FONT_HERSHEY_SIMPLEX
timeStamp=time.time()

try:
    print ('Ctrl-C to end')
    while True:

        _, frame1 = capture1.read()
        _, frame2 = capture2.read()
        frame2 = cv2.flip(frame2, -1)
        _, frame3 = capture3.read()
        

        dt=time.time()-timeStamp
        timeStamp=time.time()
        dtFIL=.9*dtFIL + .1*dt
        fps=1/dtFIL
        
        cv2.rectangle(frame1,(0,0),(150,40),(0,0,255),-1)  
        cv2.putText(frame1,'fps: '+str(round(fps,1)),(0,30),font,1,(0,255,255),2) 
        
        cv2.rectangle(frame2,(0,0),(150,40),(0,0,255),-1)
        cv2.putText(frame2,'fps: '+str(round(fps,1)),(0,30),font,1,(0,255,255),2) 
        
        cv2.rectangle(frame3,(0,0),(150,40),(0,0,255),-1)
        cv2.putText(frame3,'fps: '+str(round(fps,1)),(0,30),font,1,(0,255,255),2) 
        
        cv2.imshow("Combo", np.hstack((frame1, frame2, frame3)))
        
        rtp_out_1.write(frame1)
        rtp_out_2.write(frame2)

        if cv2.waitKey(1)==ord('q'):
            break

except KeyboardInterrupt:
    print ('Stopped')

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    raise