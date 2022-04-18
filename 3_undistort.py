#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

import cv2 as cv
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--width", type=int, default=1020)
    parser.add_argument("--height", type=int, default=768)

    parser.add_argument("--left_k_new_param", type=float, default=0.9)
    parser.add_argument("--left_k_filename", type=str, default="left_K_fisheye.csv")
    parser.add_argument("--left_d_filename", type=str, default="left_D_fisheye.csv")
    parser.add_argument("--right_k_new_param", type=float, default=0.9)
    parser.add_argument("--right_k_filename", type=str, default="right_K_fisheye.csv")
    parser.add_argument("--right_d_filename", type=str, default="right_D_fisheye.csv")

    args = parser.parse_args()

    return args


flip1=0
flip2=2

dispW=1020
dispH=768

camSet1='nvarguscamerasrc sensor-id=0 wbmode=1 ee-mode=1 ee-strength=0 tnr-mode=2 tnr-strength=1 ! video/x-raw(memory:NVMM), width=3264, height=2464, framerate=21/1,format=NV12 ! nvvidconv flip-method='+str(flip1)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! videobalance contrast=1.3 brightness=-.2 saturation=1.2 ! appsink drop=True'
camSet2='nvarguscamerasrc sensor-id=1 wbmode=1 ee-mode=1 ee-strength=0 tnr-mode=2 tnr-strength=1 ! video/x-raw(memory:NVMM), width=3264, height=2464, framerate=21/1,format=NV12 ! nvvidconv flip-method='+str(flip2)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! videobalance contrast=1.3 brightness=-.2 saturation=1.2 ! appsink drop=True'

def main():
    # コマンドライン引数
    args = get_args()

    cap_device = args.device
    filepath = args.file
    cap_width = args.width
    cap_height = args.height

    left_k_new_param = args.left_k_new_param
    left_k_filename = args.left_k_filename
    left_d_filename = args.left_d_filename

    right_k_new_param = args.right_k_new_param
    right_k_filename = args.right_k_filename
    right_d_filename = args.right_d_filename

    # カメラ準備
    cap = None
    if filepath is None:
        left_cap = cv.VideoCapture(camSet1)
        left_cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
        left_cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

        right_cap = cv.VideoCapture(camSet2)
        right_cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
        right_cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    else:
        left_cap = cv.VideoCapture(filepath)
        right_cap = cv.VideoCapture(filepath)

    # キャリブレーションデータの読み込み
    left_camera_mat = np.loadtxt(left_k_filename, delimiter=',')
    left_dist_coef = np.loadtxt(left_d_filename, delimiter=',')

    right_camera_mat = np.loadtxt(right_k_filename, delimiter=',')
    right_dist_coef = np.loadtxt(right_d_filename, delimiter=',')

    left_new_camera_mat = left_camera_mat.copy()
    left_new_camera_mat[(0, 1), (0, 1)] = left_k_new_param * left_new_camera_mat[(0, 1),
                                                                  (0, 1)]

    right_new_camera_mat = right_camera_mat.copy()
    right_new_camera_mat[(0, 1), (0, 1)] = right_k_new_param * right_new_camera_mat[(0, 1),
                                                                  (0, 1)]
    while (True):
        ret, left_frame = left_cap.read()
        ret, right_frame = right_cap.read()
        left_undistort_image = cv.fisheye.undistortImage(
            left_frame,
            left_camera_mat,
            D=left_dist_coef,
            Knew=left_new_camera_mat,
        )
        right_undistort_image = cv.fisheye.undistortImage(
            right_frame,
            right_camera_mat,
            D=right_dist_coef,
            Knew=right_new_camera_mat,
        )

        # cv.imshow('left original', left_frame)
        # cv.imshow('left undistort', left_undistort_image)
        # cv.imshow('right original', right_frame)
        # cv.imshow('right undistort', right_undistort_image)

        key = cv.waitKey(5)
        if key == 27:
            break
        elif key == ord('s'): # wait for 's' key to save and exit
            cv.imwrite('images/stereoLeft/imageL06.png', left_undistort_image)
            cv.imwrite('images/stereoRight/imageR06.png', right_undistort_image)
            print("images saved!")
            # num += 1

        cv.imshow('Img 1',left_undistort_image)
        cv.imshow('Img 2',right_undistort_image)
       
        if key == 27:  # ESC
            left_cap.release()
            right_cap.release()
            cv.destroyAllWindows()
            break

    left_cap.release()
    right_cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()