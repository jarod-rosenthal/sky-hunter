import cv2
print('cv2',cv2.__version__)
import numpy as np
print('numpy',np.__version__)
import jetson.inference
import jetson.utils
from adafruit_servokit import ServoKit
import time, board, busio
from time import sleep
import argparse, sys

# # =============================================================================
# # Parse the command line
# # =============================================================================

parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.3, help="minimum detection threshold to use") 

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# # =============================================================================
# # Detectnet settings
# # =============================================================================

net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)

# # =============================================================================
# # Pan tilt settings
# # =============================================================================

print("Initializing Servos")
i2c_bus=(busio.I2C(board.SCL_1, board.SDA_1))
kit = ServoKit(channels=16, i2c=i2c_bus)
tilt=110
pan=90

# # pan 0 is towards the right from 90 
kit.servo[0].angle=pan
time.sleep(0.3)
kit.servo[1].angle=tilt
time.sleep(0.3)

# # =============================================================================
# # Camera settings
# # =============================================================================

flip1=2
dispW=640
dispH=480

camSet1='nvarguscamerasrc sensor-id=0 wbmode=1 ! video/x-raw(memory:NVMM), width=1920, height=1080, framerate=60/1,format=NV12 ! nvvidconv flip-method='+str(flip1)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=False'
capture1=cv2.VideoCapture(camSet1, cv2.CAP_GSTREAMER)

# # =============================================================================
# # Output video stream
# # =============================================================================

w = capture1.get(cv2.CAP_PROP_FRAME_WIDTH)
h = capture1.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = capture1.get(cv2.CAP_PROP_FPS)

gst_out_1 = "appsrc ! video/x-raw, format=BGR ! queue ! videoconvert ! video/x-raw,format=BGRx ! nvvidconv ! omxh264enc insert-sps-pps=true ! h264parse ! rtph264pay pt=96 ! queue ! application/x-rtp, media=video, encoding-name=H264 ! udpsink host=192.168.50.161 port=5000"
rtp_out_1 = cv2.VideoWriter(gst_out_1, cv2.CAP_GSTREAMER, 0, float(fps), (int(w), int(h)))

# # =============================================================================
# # Frame per second counter
# # =============================================================================

dtFIL=0
timeStamp=time.time()
(H, W) = (None, None)

# # =============================================================================
# # Core program
# # =============================================================================

try:
    print ('Ctrl-C to end')
    while True:
        _, frame1 = capture1.read()
        
        if W is None or H is None:
            (H, W) = frame1.shape[:2]

        ptz=cv2.cvtColor(frame1,cv2.COLOR_BGR2RGBA).astype(np.float32)
        ptz=jetson.utils.cudaFromNumpy(ptz)
        detections = net.Detect(ptz, overlay=opt.overlay)
        
        for detection in detections:
            ID=detection.ClassID
            y=int(detection.Top)
            x=int(detection.Left)
            h=int(detection.Bottom)
            w=int(detection.Right)
            item=net.GetClassDesc(ID)
            roi = (int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom))
            cv2.putText(frame1,item,(x,y+20),
                cv2.FONT_HERSHEY_SIMPLEX, .95, (0,0,255),2)
            cv2.rectangle(frame1,(x,y),(w,h),(0,255,255),3)
            rects = [(x,y)+(w,h)]
            if item == 'person':
                objX=(x//2)+(w//2)
                objY=(y//2)+(h//2)
                errorPan=objX-dispW/2
                errorTilt=objY-dispH/2
                if abs(errorTilt)>15:
                    tilt=tilt-errorTilt/100
                if tilt<95:
                    if abs(errorPan)>15:
                        pan=pan-errorPan/70
                if tilt>95:
                    if abs(errorPan)>15:
                        pan=pan+errorPan/70
                if pan>179: pan=179
                if pan<1: pan=1
                if tilt>179: tilt=179
                if tilt<1: tilt=1
                kit.servo[0].angle=pan
                kit.servo[1].angle=tilt        
        
        dt=time.time()-timeStamp
        timeStamp=time.time()
        dtFIL=.9*dtFIL + .1*dt
        fps=1/dtFIL
   
        cv2.rectangle(frame1,(0,0),(150,40),(0,0,255),-1)
        cv2.putText(frame1,'fps: '+str(round(fps,1)),(0,30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255),2) 
        cv2.imshow("frame1", frame1)
        rtp_out_1.write(frame1)

        if cv2.waitKey(1)==ord('q'):
            frame1.release()
            cv2.destroyAllWindows()
            break

except KeyboardInterrupt:
    print ('Stopped')
    frame1.release()
    cv2.destroyAllWindows()
    raise


