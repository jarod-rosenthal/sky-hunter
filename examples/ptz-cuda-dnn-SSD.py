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

parser.add_argument("input_URI", type=str, default="csi://0", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="rtp://192.168.50.161:8000", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
# parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--overlay", type=str, default="labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.3, help="minimum detection threshold to use") 
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

# account_sid = os.environ['TWILIO_ACCOUNT_SID']
# auth_token = os.environ['TWILIO_AUTH_TOKEN']
# client = Client(account_sid, auth_token)

# # =============================================================================
# # Servo control settings
# # =============================================================================

print("Initializing Servos")
i2c_bus=(busio.I2C(board.SCL_1, board.SDA_1))
kit = ServoKit(channels=16, i2c=i2c_bus)
tilt=90
pan=90

kit.servo[0].angle=pan
time.sleep(0.3)
kit.servo[1].angle=tilt
time.sleep(0.3)

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

KPS = 0.5 # number of images per second to capture
fps = round(30)
hop = round(fps / KPS)
curr_frame = 0

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
        frame3 = input.Capture()
        height=frame3.shape[0]
        width=frame3.shape[1]
        
        timestamp = datetime.fromtimestamp(time.time())
        detection = False
        status = 0
        count = 0
            
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
            if item == 'person':
                detection = True
                status=1
                objX=(x//2)+(w//2)
                objY=(y//2)+(h//2)
                errorPan=objX-width/2
                errorTilt=objY-height/2
                if abs(errorTilt)>20:
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
                lastDetected = timestamp
                
        if detection is True:
            if curr_frame % hop == 0:
                snapshot = jetson.utils.cudaAllocMapped(width=roi[2]-roi[0], height=roi[3]-roi[1], format=frame3.format)
                jetson.utils.cudaCrop(frame3, snapshot, roi)
                jetson.utils.cudaDeviceSynchronize()
                jetson.utils.saveImage(os.path.join(opt.snapshots, f"{timestamp}-{idx}.jpg"), snapshot)
                del snapshot
            curr_frame += 1
    
        # if detection == True and (timestamp - lastDetected).seconds >= 360 and sent != True:
        #     message = client.messages \
        #         .create(
        #             body='SkyHunter Detection',
        #             from_=os.environ['TWILIO_NUMBER'],
        #             to=os.environ['CELL']
        #         )
        #     print(message.sid)
        #     sent = True
        
        status_list.append(status)
        status_list=status_list[-2:]
        if status_list[-1]==1 and status_list[-2]==0:
            times.append(datetime.now())
        if status_list[-1]==0 and status_list[-2]==1:
            times.append(datetime.now())

        output.Render(frame3)
        output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))
        
        if cv2.waitKey(1)==ord('q'):
            print (' Exiting Program')
            
            for i in range(0,len(times),2):
                df = df.append({'Start':times[i],'End':times[i+1]},ignore_index=True)
                
            df.to_csv('./out_csv/times-%s.csv' % current_time)
            cv2.destroyAllWindows()
            break

except KeyboardInterrupt:
    print (' Exiting Program')
    
    for i in range(0,len(times),2):
        df = df.append({'Start':times[i],'End':times[i+1]},ignore_index=True)
        
    df.to_csv('./out_csv/times-%s.csv' % current_time)
    cv2.destroyAllWindows()
    raise