"""
Petites fonctions utiles
Auteur: Marinouille
"""

import os
import cv2 as cv
import numpy as np


def coins_damier(patternSize,squaresize):
    objp = np.zeros((patternSize[0]*patternSize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:patternSize[0], 0:patternSize[1]].T.reshape(-1, 2)
    #
    objpy = np.flip(objp[:,1])
    objp[:,1]=objpy
    #
    # objpx = np.flip(objp[:,0])
    # objp[:,0]=objpx
    #
    objp*=squaresize
    return objp

def centres_damier(patternSize,squaresize):
    objp = np.zeros(((patternSize[0]-1)*(patternSize[1]-1), 3), np.float32)
    objp[:, :2] = np.mgrid[0:patternSize[0]-1, 0:patternSize[1]-1].T.reshape(-1, 2)
    objp*=squaresize
    objp+=np.array([0.5,0.5,0])*squaresize
    return objp

def read_images(image):
    color=cv.imread(image)
    gray=cv.cvtColor(color, cv.COLOR_BGR2GRAY)
    return color, gray


def write_ply(fn, verts, colors):
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


def find_corners(fname,patternSize):
    color = cv.imread(fname)
    # Transformation de l'image en nuance de gris pour analyse
    gray = cv.cvtColor(color, cv.COLOR_BGR2GRAY)
    # On cherche les coins sur le damier
    ret, corners = cv.findChessboardCorners(gray, patternSize, None)
    return ret, corners, color

def refine_corners(patternSize, imgpoints, corners, color, criteria, detected_path, view, i, p=True):
    # On ajoute les points dans le tableau 2D
    corners2= cv.cornerSubPix(cv.cvtColor(color, cv.COLOR_BGR2GRAY), corners, (11, 11),(-1, -1), criteria)
    imgpoints.append(corners2)
    # On dessine et affiche les coins sur l'image
    if p:
        _ = cv.drawChessboardCorners(color, patternSize, corners2, True)
        fname='{}{:03d}.jpg'.format(view, i+1)
        cv.imwrite(detected_path + fname, color)


def clean_folders(output_paths):
    for path in output_paths:
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            for file in os.scandir(path):
                if file.name.endswith(".jpg"):
                    os.unlink(file.path)

def draw_reprojection(color, objPoints, imgPoints, cameraMatrix, distCoeffs, patternSize, squaresize, folder, i):
    """ Pour une image, reprojeter des points et les axes"""
    # Vérification de la calibration de la caméra en reprojection:
    # Montrer axes
    ret, rvecs, tvecs = cv.solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs)
    img	=cv.drawFrameAxes(color.copy(), cameraMatrix, distCoeffs, rvecs, tvecs, 3*squaresize, 5)
    cv.imwrite('{}reprojection_axes_{}.jpg'.format(folder, i), img) #Z est en bleu, Y en vert, X en rouge
    pts, jac = cv.projectPoints(objPoints, rvecs, tvecs, cameraMatrix, distCoeffs)
    img = cv.drawChessboardCorners(color.copy(), patternSize, pts, 1)
    cv.imwrite('{}reprojection_points_{}.jpg'.format(folder, i), img)
    return img

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        pt1=tuple(pt1[0]); pt2=tuple(pt2[0])
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,pt1,5,color,-1)
        img2 = cv.circle(img2,pt2,5,color,-1)
    return img1,img2


def readXML(fname):
    s = cv.FileStorage()
    s.open(fname, cv.FileStorage_READ)
    K=s.getNode('K').mat()
    R=s.getNode('R').mat()
    t=s.getNode('t').mat()
    D=s.getNode('coeffs').mat()
    imageSize=s.getNode('imageSize').mat()
    imageSize=(int(imageSize[0][0]),int(imageSize[1][0]))
    E=s.getNode('E').mat()
    F=s.getNode('F').mat()
    s.release()
    return K,D,R,t,imageSize, E, F
