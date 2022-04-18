# from wsgiref.headers import tspecials
import cv2, sys, logging
print('cv2',cv2.__version__)
import numpy as np
print('numpy',np.__version__)
import jetson.inference
import jetson.utils
import argparse
             
# # =============================================================================
# # Parse the command line
# # =============================================================================

parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

# ssd-mobilenet-v2, coco-dog, coco-airplane, facenet
# cd jetson-inference/tools,  ./download-models.sh
#  --output-width=640, --output-height=480, --output-frameRate=30, --output-codec=h265
parser.add_argument("input_URI", type=str, default="/dev/video3", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="rtp://192.168.50.161:7000", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)
 
# net = jetson.inference.detectNet(argv=["--model=./models/ssd-mobilenet.onnx", "--labels=./models/labels.txt", "--input-blob=input_0", "--output-cvg=scores", "--output-bbox=boxes"], threshold=0.5)
net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)
display = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)
camera = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)

try:
    print ('Ctrl-C to end')
    while True:
        frame = camera.Capture()

        detections = net.Detect(frame, overlay=opt.overlay)
        for detect in detections:
            ID=detect.ClassID
            y=int(detect.Top)
            x=int(detect.Left)
            h=int(detect.Bottom)
            w=int(detect.Right)
            item=net.GetClassDesc(ID)
            print(item,x,y,w,h)
        
        display.Render(frame)
        display.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

        # print out performance info
        net.PrintProfilerTimes()

        # exit on input/output EOS
        if not camera.IsStreaming() or not display.IsStreaming():
            break

        # escape key to quit
        key = cv2.waitKey(5)
        if key == 27:
            frame.capture.release()
            cv2.destroyAllWindows()
            break

except KeyboardInterrupt:
    print (' Exiting Program')
    frame.capture.release()
    cv2.destroyAllWindows()
    raise

