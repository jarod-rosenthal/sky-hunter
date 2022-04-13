# Jetson Nano  Pan Tilt Tracking

## Requirements:
Hardware: NVIDIA Jetson Nano 4GB

Software: [JetPack 4.6.1](https://developer.nvidia.com/embedded/downloads/) 

## Servo Library
	pip3 install adafruit-circuitpython-ServoKit
	# To make compatible with python3.6.9
	pip3 uninstall Adafruit-PlatformDetect
	pip3 install Adafruit-PlatformDetect==3.18.0

## Notes

#### Check available camera fomats
	v4l2-ctl -d /dev/video1 --list-formats-ext

#### Tests CSI camera with gstreamer
	gst-launch-1.0 nvarguscamerasrc sensor_id=0 ! nvoverlaysink
	
	gst-launch-1.0 nvarguscamerasrc sensor_id=0 ! \
	   'video/x-raw(memory:NVMM),width=1920, height=1080, framerate=30/1' ! \
	   nvvidconv flip-method=0 ! 'video/x-raw,width=960, height=540' ! \
	   nvvidconv ! nvegltransform ! nveglglessink -e

#### To launch camera with 4l2src
	gst-launch-1.0 v4l2src device=/dev/video0 ! xvimagesink -e

#### Starts a RTSP server
	sudo docker run --rm -it -e RTSP_PROTOCOLS=tcp -p 8554:8554 -p 1935:1935 -p 8888:8888 aler9/rtsp-simple-server

#### Publishes a stream to the RTSP server
	sudo ffmpeg -f v4l2 -framerate 24 -video_size 6400x480 -i /dev/video0 -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/webCamStream
	   
#### Deepstream reference apps
	https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps
	
	deepstream-app -c source8_1080p_dec_infer-resnet_tracker_tiled_display_fp16_nano.txt
	
#### Deepstream sample Python apps
	https://github.com/NVIDIA-AI-IOT/deepstream_python_apps

#### Run Deepstream deepstream_test_3.py
	cd /opt/nvidia/deepstream/deepstream-6.0/sources/
	git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git
	cd /opt/nvidia/deepstream/deepstream-6.0/sources/deepstream_python_apps/apps/deepstream-test3/
	
	python3 deepstream_test_3.py rtsp://127.0.0.1:8554/webCamStream

#### To test servos are connected properly
	i2cdetect -y -r 1

#### Output
	     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
	00:          -- -- -- -- -- -- -- -- -- -- -- -- -- 
	10: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
	20: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
	30: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
	40: 40 -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
	50: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
	60: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
	70: 70 -- -- -- -- -- -- --

#### To test movement
	python3
	>> from adafruit_servokit import ServoKit
	>> Kit=ServoKit(channels=16)
	>> Kit.servo[0].angle=90

#### To launch the mipi csi camera with imagenet
	~/jetson-inference/build/aarch64/bin/imagenet csi://0

#### To launch the usb camera with imagenet
	~/jetson-inference/build/aarch64/bin/imagenet /dev/video1 

#### To list attached cameras
	v4l2-ctl --list-devices

#### Sample output:
	vi-output, imx219 6-0010 (platform:54080000.vi:0):
		/dev/video0

	WebCamera (usb-70090000.xusb-2.1):
		/dev/video1

#### Jetson Setup
	sudo -H pip3 install -U jetson-stats

	sudo nvpmodel -m 0

	sudo jetson_clocks

#### Setup Jetson-Inference
	git clone --recursive https://github.com/dusty-nv/jetson-inference
	
	sudo apt-get install git cmake libpython3-dev python3-numpy

	cd jetson-inference

	mkdir build && cd build

	cmake ../

	make

	sudo make install

	sudo ldconfig

#### Handy commands

	python3 -c 'import sys; print(sys.path)'

	python3 -c 'import cv2; print(cv2.__version__)'

	export PYTHONPATH=/usr/local/opencv/3.6.9/:$PYTHONPATH

	systemctl set-default multi-user.target

	sudo systemctl stop gdm3

	systemctl set-default graphical.target

	sudo systemctl start gdm3

lsb_release -a # Ubuntu Version

	No LSB modules are available.
	Distributor ID: Ubuntu
	Description:    Ubuntu 18.04.4 LTS
	Release:        18.04
	Codename:       bionic

cat /etc/nv_tegra_release # Jetpack Version

	# R32 (release), REVISION: 7.1, GCID: 29818004, BOARD: t210ref 
	# EABI: aarch64, DATE: Sat Feb 19 17:05:08 UTC 2022

nvcc --version # Cuda Version

	nvcc: NVIDIA (R) Cuda compiler driver
	Copyright (c) 2005-2021 NVIDIA Corporation
	Built on Sun_Feb_28_22:34:44_PST_2021
	Cuda compilation tools, release 10.2, V10.2.300
	Build cuda_10.2_r440.TC440_70.29663091_0

## MISC NOTES for running virtual environments:

### Install base apps:

	sudo apt update -y

	sudo apt install nano python-pip python3-pip python3-venv python3-setuptools libgdm-dev libnss3-dev libssl-dev libsqlite3-dev \
	libreadline-dev libbz2-dev libdb-dev libdb++-dev libgdbm-dev libgdbm-dev libffi-dev -y

#### Install Jetson Stats: (Optional)

	sudo -H pip3 install -U jetson-stats
	sudo /usr/bin/jetson_clocks.sh

### Install Python3.8:

	wget https://www.python.org/ftp/python/3.8.3/Python-3.8.3.tar.xz -O ~/Downloads/Python-3.8.3.tar.xz
	tar -xvf ~/Downloads/Python-3.8.3.tar.xz -C ~/Downloads
	mkdir build-python-3.8.3 && cd $_
	../Python-3.8.3/configure --enable-optimizations
	make -j$(nproc)
	sudo -H make altinstall

### Create Python3.8.3 Virtual Environment:

	# Create python3.8 virtual environment
	/usr/local/bin/python3.8 -m venv ~/.py3.8.3
	# Create bash alias to activate
	echo "alias activate='source ~/.py3.8.3/bin/activate'" >> ~/.bashrc
	# Load changes
	source ~/.bashrc
	# Activate virtual environment
	activate
	# Link python3.6 cv2 module to python3.8 site-packages
	ln -s /usr/lib/python3.6/dist-packages/cv2/python-3.6/cv2.cpython-36m-aarch64-linux-gnu.so ~/.py3.8.0/lib/python3.8/site-packages/cv2.so
	# Install dependencies
	pip install --upgrade pip
	pip install -U numpy 
	pip install adafruit-circuitpython-servokit
	# pip install -r requirements.txt

### OpenCV and Adafruit libraries are now installed. 
