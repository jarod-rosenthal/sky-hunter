from wsgiref.headers import tspecials
import cv2
print('cv2',cv2.__version__)
import numpy as np
print('numpy',np.__version__)
import jetson.inference
import jetson.utils
from adafruit_servokit import ServoKit
from datetime import datetime
from twilio.rest import Client
import time, board, busio, imutils
import pandas, os, argparse


# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-v", "--video",
# 	help="path to the (optional) video file")
# ap.add_argument("-b", "--buffer", type=int, default=32,
# 	help="max buffer size")
# args = vars(ap.parse_args())


# # =============================================================================
# # SMS PARAMETERS
# # =============================================================================

account_sid = os.environ['TWILIO_ACCOUNT_SID']
auth_token = os.environ['TWILIO_AUTH_TOKEN']
client = Client(account_sid, auth_token)

# # =============================================================================
# # SERVO CONTROL
# # =============================================================================

print("Initializing Servos")
i2c_bus=(busio.I2C(board.SCL_1, board.SDA_1))
kit = ServoKit(channels=16, i2c=i2c_bus)
tilt=90
pan=90

# # pan 0 is towards the right from 90 
kit.servo[0].angle=pan
time.sleep(0.3)
kit.servo[1].angle=tilt
time.sleep(0.3)

# # =============================================================================
# # USER-SET PARAMETERS
# # =============================================================================

FRAMES_TO_PERSIST = 10
MIN_SIZE_FOR_MOVEMENT = 2000
SEC_SINCE_LAST_DETECTION = 2
MOTION_COUNTER = 2

# # =============================================================================
# # Grid pattern for motion detection
# # =============================================================================

exec(open('grid').read())

# # =============================================================================
# # Camera Settings
# # =============================================================================
# videobalance contrast=1.3 brightness=-.2 saturation=1.2 

flip=2
dispW=640
dispH=480

camSet='nvarguscamerasrc sensor-id=0 ee-mode=1 ee-strength=0 tnr-mode=2 tnr-strength=1 wbmode=1 ! video/x-raw(memory:NVMM), width=1920, height=1080, framerate=60/1,format=NV12 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=False'
capture = cv2.VideoCapture(camSet, cv2.CAP_GSTREAMER)

ptzSet='nvarguscamerasrc sensor-id=1 ee-mode=1 ee-strength=0 tnr-mode=2 tnr-strength=1 wbmode=1 ! video/x-raw(memory:NVMM), width=1920, height=1080, framerate=60/1,format=NV12 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=False'
ptzCam=cv2.VideoCapture(ptzSet, cv2.CAP_GSTREAMER)

# # =============================================================================
# # Output video stream
# # =============================================================================

w = ptzCam.get(cv2.CAP_PROP_FRAME_WIDTH)
h = ptzCam.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = ptzCam.get(cv2.CAP_PROP_FPS)

gst_out_1 = "appsrc ! video/x-raw, format=BGR ! queue ! videoconvert ! video/x-raw,format=BGRx ! nvvidconv ! omxh264enc insert-sps-pps=true ! h264parse ! rtph264pay pt=96 ! queue ! application/x-rtp, media=video, encoding-name=H264 ! udpsink host=192.168.50.161 port=5000"
rtp_out_1 = cv2.VideoWriter(gst_out_1, cv2.CAP_GSTREAMER, 0, float(fps), (int(w), int(h)))
gst_out_2 = "appsrc ! video/x-raw, format=BGR ! queue ! videoconvert ! video/x-raw,format=BGRx ! nvvidconv ! omxh264enc insert-sps-pps=true ! h264parse ! rtph264pay pt=96 ! queue ! application/x-rtp, media=video, encoding-name=H264 ! udpsink host=192.168.50.161 port=6000"
rtp_out_2 = cv2.VideoWriter(gst_out_2, cv2.CAP_GSTREAMER, 0, float(fps), (int(w), int(h)))
# gst_out_3 = "appsrc ! video/x-raw, format=GRAY8 ! nvvidconv ! omxh264enc insert-sps-pps=true ! h264parse ! rtph264pay pt=96 ! queue ! application/x-rtp, media=video, encoding-name=H264 ! udpsink host=192.168.50.161 port=7000"
# rtp_out_3 = cv2.VideoWriter(gst_out_3, cv2.CAP_GSTREAMER, 0, float(fps), (int(w), int(h)), False)

# # =============================================================================
# # Output images to folder
# # =============================================================================

images_folder = 'out_images'
if not os.path.exists('out_images'):
    os.mkdir(images_folder)
    
KPS = 1 # number of images per second to capture
p_fps = round(ptzCam.get(cv2.CAP_PROP_FPS))
hop = round(p_fps / KPS)
f_fps = round(ptzCam.get(cv2.CAP_PROP_FPS))
hop = round(f_fps / KPS)
curr_frame = 0

# # =============================================================================
# # Output video to a folder
# # =============================================================================

video_folder = 'out_video'
if not os.path.exists('out_video'):
    os.mkdir(video_folder)
     
frame_width = int(ptzCam.get(3))
frame_height = int(ptzCam.get(4))
now = datetime.now()
current_time = now.strftime("%d_%m_%Y_%H_%M")
filename = './out_video/ptz-%s.mp4' % current_time
ptz_out = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'mp4v'), 20, (frame_width,frame_height))
n_frames = 0
max_number_framed_to_be_saved = 1000

# # =============================================================================
# # Timers and detection setings
# # =============================================================================

dtFIL=0
font=cv2.FONT_HERSHEY_SIMPLEX
timeStamp=time.time()
lastDetected = datetime.fromtimestamp(time.time())
ts = str(datetime.fromtimestamp(time.time()))
net=jetson.inference.detectNet('ssd-mobilenet-v2',threshold=.4)
motionCounter = 0
delay_counter = 0
currentFrame = 0
count = 0

# # =============================================================================
# # Output detection times to csv
# # =============================================================================

csv_folder = 'out_csv'
if not os.path.exists('out_csv'):
    os.mkdir(csv_folder)
    
current_time = now.strftime("%d_%m_%Y_%H_%M")
filename = './out_video/ptz-%s.mp4' % current_time

status_list = [None,None]
times = []
df=pandas.DataFrame(columns=["Start","End"])
first_frame = None
next_frame = None


def process_img(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_canny = cv2.Canny(img_gray, 100, 100)
    kernel = np.ones((2, 3))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=1)
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)
    return img_erode

def get_centeroid(cnt):
    length = len(cnt)
    sum_x = np.sum(cnt[..., 0])
    sum_y = np.sum(cnt[..., 1])
    return int(sum_x / length), int(sum_y / length)

def get_centers(img):
    contours, hierarchies = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            yield get_centeroid(cnt)
            
def get_rows(img, centers, row_amt, row_h):
    centers = np.array(centers)
    d = row_h / row_amt
    for i in range(row_amt):
        f = centers[:, 1] - d * i
        a = centers[(f < d) & (f > 0)]
        yield a[a.argsort(0)[:, 0]]
        
# def tracker(x,y,w,h):
#     objCenter = ((x+y)/2, (w+h)/2)
#     center_x, center_y = (x+y)/2, (w+h)/2
#     pan = center_x/4 
#     tilt = center_y/3
#     return pan, tilt

# =============================================================================
# CORE PROGRAM
# =============================================================================

y1 = 0
M = dispH//8
N = dispW//8
        
try:
    print ('Ctrl-C to end')
    while True:
        
        _, fisheye = capture.read()
        height=fisheye.shape[0]
        width=fisheye.shape[1]
        _, ptzcam = ptzCam.read()
        height=ptzcam.shape[0]
        width=ptzcam.shape[1]

        if not _:
            print("CAPTURE ERROR")
            break
 
        timestamp = datetime.fromtimestamp(time.time())
        ptz_text = "Scanning"
        status = 0
        text = "Scanning"
        detection = False
        
        img_processed = process_img(fisheye)
        centers = list(get_centers(img_processed))
        h, w, c = fisheye.shape
        count = 0
        
        for row in get_rows(fisheye, centers, 6, h):
            # cv2.polylines(fisheye, [row], False, (255, 0, 255), 2)
            for x, y in row:
                count += 1
                # print(centers)
                cv2.circle(fisheye, (x, y), 10, (0, 0, 255), -1)  
                cv2.putText(fisheye, str(count), (x - 10, y + 5), 1, cv2.FONT_HERSHEY_PLAIN, (0, 255, 255), 2)
                
        frame1=cv2.cvtColor(fisheye,cv2.COLOR_BGR2RGBA).astype(np.float32)
        frame1=jetson.utils.cudaFromNumpy(frame1)
        detections=net.Detect(frame1, width, height)
        
        for detect in detections:
            ID=detect.ClassID
            y=int(detect.Top)
            x=int(detect.Left)
            h=int(detect.Bottom)
            w=int(detect.Right)
            item=net.GetClassDesc(ID)
            # print(item,x,y,w,h,timestamp)
            cv2.putText(fisheye,item,(x,y+20),font,.95,(0,0,255),2)
            cv2.rectangle(fisheye,(x,y),(w,h),(0,255,255),3)
            if item == 'airplane' and ptz_text != "Tracking" and (timestamp - lastDetected).seconds >= 5:
                text = "Moving PTZ"
                status=1
                objCenter = ((x+y)/2, (w+h)/2)
                center_x, center_y = (x+y)/2, (w+h)/2
                pan = center_x/3.5 
                tilt = center_y/3.75
                errorPan=center_x-width/2
                errorTilt=center_y-height/2
                if abs(errorTilt)>50: tilt=tilt+errorTilt/120
                if abs(errorPan)>50: pan=pan-errorPan/120
                if pan>179: pan=179
                if pan<1: pan=1
                if tilt>179: tilt=179
                if tilt<1: tilt=1
                kit.servo[0].angle=pan
                kit.servo[1].angle=tilt
                print(pan, tilt)
                
        ptz=cv2.cvtColor(ptzcam,cv2.COLOR_BGR2RGBA).astype(np.float32)
        ptz=jetson.utils.cudaFromNumpy(ptz)
        ptz_detections=net.Detect(ptz, width, height)
        
        for detect in ptz_detections:
            ID=detect.ClassID
            y=int(detect.Top)
            x=int(detect.Left)
            h=int(detect.Bottom)
            w=int(detect.Right)
            item=net.GetClassDesc(ID)
            # print(item,x,y,w,h,timestamp)
            cv2.putText(ptzcam,item,(x,y+20),font,.95,(0,0,255),2)
            cv2.rectangle(ptzcam,(x,y),(w,h),(0,255,255),3)
            if item == 'airplane':
                ptz_text = "Tracking"
                status=1
                objX=(x//2)+(w//2)
                objY=(y//2)+(h//2)
                errorPan=objX-width/2
                errorTilt=objY-height/2
                if abs(errorTilt)>20:
                    tilt=tilt+errorTilt/100
                if tilt<90:
                    if abs(errorPan)>15:
                        pan=pan-errorPan/70
                if tilt>90:
                    if abs(errorPan)>15:
                        pan=pan+errorPan/70
                if pan>179: pan=179
                if pan<1: pan=1
                if tilt>179: tilt=179
                if tilt<1: tilt=1
                kit.servo[0].angle=pan
                kit.servo[1].angle=tilt
                lastDetected = timestamp
                detection = True
                # print(pan, tilt)
                
        if tilt>90:
            # print('Rotate Image')
            ptzcam = cv2.rotate(ptzcam, cv2.ROTATE_180)
            
        
        for y in range(0,dispH,M):
            for x in range(0, dispW, N):
                y1 = y + M
                x1 = x + N
                tiles = fisheye[y:y+M,x:x+N]
                cv2.rectangle(fisheye, (x, y), (x1, y1), (255, 255, 255), 1)
        
        if detection == True:
            if curr_frame % hop == 0:
                cv2.imwrite(os.path.join(images_folder,"ptzcam-{:d}.jpg".format(curr_frame)), ptzcam) 
                print("{} images are extacted in {}.")
            curr_frame += 1
    
        if detection == True and (timestamp - lastDetected).seconds >= 360:
            message = client.messages \
                .create(
                    body='SkyHunter Detection',
                    from_=os.environ['TWILIO_NUMBER'],
                    to=os.environ['CELL']
                )
            print(message.sid)
        
       
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
        
        cv2.putText(fisheye, "Status: {}".format(text), (3, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 204, 0), 2)
        cv2.rectangle(fisheye,(0,0),(58,33),(255,0,0),-1)
        cv2.putText(fisheye, str(round(fps,1)),(1,25),font,0.75,(0,255,0),2)
        cv2.putText(fisheye, ts, (10, fisheye.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(ptzcam, "Status: {}".format(ptz_text), (3, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 204, 0), 2)
        cv2.rectangle(ptzcam,(0,0),(58,33),(255,0,0),-1)
        cv2.putText(ptzcam, str(round(fps,1)),(1,25),font,0.75,(0,255,0),2)
        cv2.putText(ptzcam, ts, (10, ptzcam.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        # cv2.rectangle(diff,(0,0),(58,33),(255,0,0),-1)
        # cv2.putText(diff, str(round(fps,1)),(1,25),font,0.75,(0,255,0),2)
        
        # delta = imutils.resize(delta, width=640, height=480)
        # fisheye = imutils.resize(fisheye, width=640, height=480)
        # ptzcam = imutils.resize(ptzcam, width=640, height=480)
        # cv2.imshow("Sky Tracker", np.hstack((fisheye, ptzcam)))
        # cv2.imshow("Delta", diff) 
        # cv2.moveWindow("Delta",0,730)   
        # cv2.moveWindow("Sky Tracker",0,530)
        # combo = np.hstack((fisheye, ptzcam))

        rtp_out_1.write(ptzcam)
        rtp_out_2.write(fisheye)
        # rtp_out_3.write(diff)

        # Interrupt by pressing q to quit the program
        ch = cv2.waitKey(1)
        if ch & 0xFF == ord('q'):
            cv2.waitKey(0)
            for i in range(0,len(times),2):
                df = df.append({'Start':times[i],'End':times[i+1]},ignore_index=True)

            df.to_csv('./out_csv/times-%s.csv' % current_time)
            capture.release()
            ptzCam.release()
            cv2.destroyAllWindows()
            break
        
except KeyboardInterrupt:
    print ('Stopped')
    # Cleanup when closed
    for i in range(0,len(times),2):
        df = df.append({'Start':times[i],'End':times[i+1]},ignore_index=True)
        
    df.to_csv('./out_csv/times-%s.csv' % current_time)
    cv2.waitKey(0)
    capture.release()
    ptzCam.release()
    cv2.destroyAllWindows()
    raise