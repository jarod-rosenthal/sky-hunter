# from wsgiref.headers import tspecials
import cv2, sys, logging
print('cv2',cv2.__version__)
import numpy as np
print('numpy',np.__version__)
import jetson.inference
import jetson.utils
from adafruit_servokit import ServoKit
from datetime import datetime
import time, board, busio, imutils, pandas
from time import sleep
from datetime import datetime
from twilio.rest import Client
import os, argparse, math

# Create and configure logger
logging.basicConfig(filename="output.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


# # =============================================================================
# # Parse the command line
# # =============================================================================

parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="/dev/video3", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="rtp://192.168.50.161:7000", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 
parser.add_argument("--snapshots", type=str, default="out_images", help="output directory of detection snapshots")
parser.add_argument("--timestamp", type=str, default="%Y%m%d-%H%M%S-%f", help="timestamp format used in snapshot filenames")

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

os.makedirs(opt.snapshots, exist_ok=True)

# # =============================================================================
# # Detectnet settings
# # =============================================================================

net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)

# # =============================================================================
# # SMS parameters
# # =============================================================================

account_sid = os.environ['TWILIO_ACCOUNT_SID']
auth_token = os.environ['TWILIO_AUTH_TOKEN']
client = Client(account_sid, auth_token)

# # =============================================================================
# # Servo control settings
# # =============================================================================

print("Initializing Servos")
i2c_bus=(busio.I2C(board.SCL, board.SDA))
kit = ServoKit(channels=16, i2c=i2c_bus)
tilt=90
pan=90

kit.servo[0].angle=pan
time.sleep(0.3)
kit.servo[1].angle=tilt
time.sleep(0.3)

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

# # =============================================================================
# # Timers and detection setings
# # =============================================================================

now = datetime.now()
current_time = now.strftime("%d_%m_%Y_%H_%M")
timeMark=time.time()
dtFIL=0
font=cv2.FONT_HERSHEY_SIMPLEX
timeStamp=time.time()
lastDetected = datetime.fromtimestamp(time.time())
ts = str(datetime.fromtimestamp(time.time()))
motionCounter = 0
delay_counter = 0
currentFrame = 0
count = 0

# # =============================================================================
# # Output video stream
# # =============================================================================

w = capture1.get(cv2.CAP_PROP_FRAME_WIDTH)
h = capture1.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = capture1.get(cv2.CAP_PROP_FPS)

gst_out_1 = "appsrc ! video/x-raw, format=BGR ! queue ! videoconvert ! video/x-raw,format=BGRx ! nvvidconv ! omxh264enc insert-sps-pps=true ! h264parse ! \
    rtph264pay pt=96 ! queue ! application/x-rtp, media=video, encoding-name=H264 ! udpsink host=192.168.50.161 port=5000"
rtp_out_1 = cv2.VideoWriter(gst_out_1, cv2.CAP_GSTREAMER, 0, float(fps), (int(w), int(h)))

w = capture2.get(cv2.CAP_PROP_FRAME_WIDTH)
h = capture2.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = capture2.get(cv2.CAP_PROP_FPS)

gst_out_2 = "appsrc ! video/x-raw, format=BGR ! queue ! videoconvert ! video/x-raw,format=BGRx ! nvvidconv ! omxh264enc insert-sps-pps=true ! h264parse ! \
    rtph264pay pt=96 ! queue ! application/x-rtp, media=video, encoding-name=H264 ! udpsink host=192.168.50.161 port=6000"
rtp_out_2 = cv2.VideoWriter(gst_out_2, cv2.CAP_GSTREAMER, 0, float(fps), (int(w), int(h)))

# # =============================================================================
# # Output images to folder
# # =============================================================================

images_folder = 'out_images'
if not os.path.exists('out_images'):
    os.mkdir(images_folder)
    
KPS = 1 # number of images per second to capture
left_fps = round(capture1.get(cv2.CAP_PROP_FPS))
hop = round(left_fps / KPS)
right_fps = round(capture2.get(cv2.CAP_PROP_FPS))
hop = round(right_fps / KPS)
curr_frame = 0

# # =============================================================================
# # Output video to a folder
# # =============================================================================

video_folder = 'out_video'
if not os.path.exists('out_video'):
    os.mkdir(video_folder)
     
frame_width = int(capture1.get(3))
frame_height = int(capture1.get(4))

filename = './out_video/ptz-%s.mp4' % current_time
ptz_out = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'mp4v'), 20, (frame_width,frame_height))
n_frames = 0
max_number_framed_to_be_saved = 1000

# # =============================================================================
# # Output detection times to csv
# # =============================================================================

csv_folder = 'out_csv'
if not os.path.exists('out_csv'):
    os.mkdir(csv_folder)


filename = './out_video/ptz-%s.mp4' % current_time
status_list = [None,None]
times = []
df=pandas.DataFrame(columns=["Start","End"])
font=cv2.FONT_HERSHEY_SIMPLEX
sent = False

# # =============================================================================
# # Core program
# # =============================================================================

try:
    print ('Ctrl-C to end')
    while True:
        _, frame1 = capture1.read()
        _, frame2 = capture2.read()
        frame2 = cv2.flip(frame2, -1)
        frame3 = input.Capture()
        
        timestamp = datetime.fromtimestamp(time.time())
        detection = False
        status = 0
        count = 0
        height=frame3.shape[0]
        width=frame3.shape[1]
        
        left=cv2.cvtColor(frame1,cv2.COLOR_BGR2RGBA).astype(np.float32)
        left=jetson.utils.cudaFromNumpy(left)
        left_detect = net.Detect(left, overlay=opt.overlay)
        for detect in left_detect:
            ID=detect.ClassID
            y=int(detect.Top)
            x=int(detect.Left)
            h=int(detect.Bottom)
            w=int(detect.Right)
            item=net.GetClassDesc(ID)
            print(item,x,y,w,h,timestamp)
            cv2.putText(frame1,item,(x,y+20),font,.95,(0,0,255),2)
            cv2.rectangle(frame1,(x,y),(w,h),(0,255,255),3)
            if item == 'airplane' and detection == False:
                detection = True
                status=1
                objX=x+w/2
                objY=y+h/2
                pan = objX//3 - 5
                tilt = objY//4 - 20
                if pan>179: pan=180
                if pan<1: pan=0
                if tilt>179: tilt=180
                if tilt<1: tilt=0
                # kit.servo[0].angle=pan
                # kit.servo[1].angle=tilt
                print('left', pan, tilt)
                break
           
        right=cv2.cvtColor(frame2,cv2.COLOR_BGR2RGBA).astype(np.float32)
        right=jetson.utils.cudaFromNumpy(right)
        right_detect = net.Detect(right, overlay=opt.overlay) 
        for detect in right_detect:
            ID=detect.ClassID
            y=int(detect.Top)
            x=int(detect.Left)
            h=int(detect.Bottom)
            w=int(detect.Right)
            item=net.GetClassDesc(ID)
            print(item,x,y,w,h,timestamp)
            cv2.putText(frame2,item,(x,y+20),font,.95,(0,0,255),2)
            cv2.rectangle(frame2,(x,y),(w,h),(0,255,255),3)
            if item == 'airplane' and detection == False:
                detection = True
                status=1
                objX=x+w/2
                objY=y+h/2
                pan = objX//3 - 40
                tilt = objY//6 + 90
                if pan>179: pan=180
                if pan<1: pan=0
                if tilt>179: tilt=180
                if tilt<1: tilt=0
                # kit.servo[0].angle=pan
                # kit.servo[1].angle=tilt
                print('right', pan, tilt)
                break  
            
        detections = net.Detect(frame3, overlay=opt.overlay) 
        print("detected {:d} objects in image".format(len(detections)))
            
        for idx, detection in enumerate(detections):
            ID=detection.ClassID
            y=int(detection.Top)
            x=int(detection.Left)
            h=int(detection.Bottom)
            w=int(detection.Right)
            item=net.GetClassDesc(ID)
            roi = (int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom))
            print(detection)
            # print(item,x,y,w,h,timestamp)
            if item == 'airplane':
                
                detection = True
                status=1
                snapshot = jetson.utils.cudaAllocMapped(width=roi[2]-roi[0], height=roi[3]-roi[1], format=frame3.format)
                jetson.utils.cudaCrop(frame3, snapshot, roi)
                jetson.utils.cudaDeviceSynchronize()
                jetson.utils.saveImage(os.path.join(opt.snapshots, f"{timestamp}-{idx}.jpg"), snapshot)
                del snapshot
                objX=(x//2)+(w//2)
                objY=(y//2)+(h//2)
                errorPan=objX-width/2
                errorTilt=objY-height/2
                if abs(errorTilt)>20:
                    tilt=tilt+errorTilt/100
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
                lastDetected = timestamp
                print('ptz', pan, tilt)
        
        # if detection is True:
        #     if curr_frame % hop == 0:
        #         cv2.imwrite(os.path.join(images_folder,"ptzcam-{:d}.jpg".format(curr_frame)), ptzcam) 
        #         print("{} images are extacted in {}.")
        #     curr_frame += 1
    
        if detection == True and (timestamp - lastDetected).seconds >= 360 and sent != True:
            message = client.messages \
                .create(
                    body='SkyHunter Detection',
                    from_=os.environ['TWILIO_NUMBER'],
                    to=os.environ['CELL']
                )
            print(message.sid)
            sent = True
        
        #list of status for every frame
        status_list.append(status)
        status_list=status_list[-2:]
        #Record datetime in a list when change occurs
        if status_list[-1]==1 and status_list[-2]==0:
            times.append(datetime.now())
        if status_list[-1]==0 and status_list[-2]==1:
            times.append(datetime.now())
        
        dt=time.time()-timeStamp
        timeStamp=time.time()
        dtFIL=.9*dtFIL + .1*dt
        fps=1/dtFIL
        
        cv2.rectangle(frame1,(0,0),(150,40),(0,0,255),-1)  
        cv2.putText(frame1,'fps: '+str(round(fps,1)),(0,30),font,1,(0,255,255),2) 
        # frame1 = cv2.flip(frame1, -1)
        cv2.rectangle(frame2,(0,0),(150,40),(0,0,255),-1)
        cv2.putText(frame2,'fps: '+str(round(fps,1)),(0,30),font,1,(0,255,255),2) 
        output.Render(frame3)
        output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))
        # cv2.imshow("Frame", np.vstack((frame1, frame2)))
        # cv2.moveWindow("Frame", 0, 560)

        rtp_out_1.write(frame1)
        rtp_out_2.write(frame2)

        if cv2.waitKey(1)==ord('q'):
            print (' Exiting Program')
            
            for i in range(0,len(times),2):
                df = df.append({'Start':times[i],'End':times[i+1]},ignore_index=True)
                
            df.to_csv('./out_csv/times-%s.csv' % current_time)
            break

except KeyboardInterrupt:
    print (' Exiting Program')
    
    for i in range(0,len(times),2):
        df = df.append({'Start':times[i],'End':times[i+1]},ignore_index=True)
        
    df.to_csv('./out_csv/times-%s.csv' % current_time)
    frame2.release()
    frame1.release()
    cv2.destroyAllWindows()
    raise