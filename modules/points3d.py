import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob, os


from modules.util import *


class Camera():
    """
    Classe Camera : sert à handle les infos des caméras

    """

    def __init__(self, cameraMatrix, distCoeffs, rectificationMatix, projectionMatrix, map_x, map_y):

        self.K=cameraMatrix
        self.D=distCoeffs
        self.R=rectificationMatix
        self.P=projectionMatrix
        self.map_x=map_x
        self.map_y=map_y
        self.Q=None

    def set_images(self, fname):
        # LIRE IMAGES ORIGINALES -----------------------------------------------
        self.not_rectified=cv.imread(fname)
        assert self.not_rectified is not None, 'Image non lue. Vérifier path'
        # ----------------------------------------------------------------------
        # RECTIFICATION --------------------------------------------------------
        self.rectified = cv.remap(self.not_rectified, self.map_x, self.map_y, interpolation=cv.INTER_LINEAR)
        # ----------------------------------------------------------------------



def get_cameras(left_xml, right_xml, alpha=0):

    """
    Sert à initialiser des objets de la classe Camera correctement à partir de fichiers de calibration .xml
    """

    # LIRE FICHIERS DE CALIBRATION ---------------------------------------------
    K1,d1, _, _ ,imageSize, _, _ = readXML(left_xml) # left
    K2,d2, R, T ,_, _, _ = readXML(right_xml) # right
    # --------------------------------------------------------------------------

    # RESOLUTION ET TAILLE -----------------------------------------------------
    height, width = imageSize[0], imageSize[1]
    # --------------------------------------------------------------------------

    # RECTIFICATION ------------------------------------------------------------
    R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(cameraMatrix1=K1, cameraMatrix2=K2,distCoeffs1=d1,distCoeffs2=d2,R=R, T=T, flags=cv.CALIB_ZERO_DISPARITY, alpha=alpha, imageSize=(width,height), newImageSize=(width,height))

    map_left_x, map_left_y = cv.initUndistortRectifyMap(K1, d1, R1, P1, (width, height), cv.CV_32FC1)
    map_right_x, map_right_y = cv.initUndistortRectifyMap(K2, d2, R2, P2, (width, height), cv.CV_32FC1)
    # --------------------------------------------------------------------------

    # CAMÉRAS ------------------------------------------------------------------
    cam1 = Camera(K1,d1,R1,P1,map_left_x,map_left_y); cam1.Q=Q
    cam2 = Camera(K2,d2,R2,P2,map_right_x,map_right_y); cam2.r12=R; cam2.t12=T
    # --------------------------------------------------------------------------

    return cam1, cam2

def calcul_mesh(rectifiedL, rectifiedR, Q, downscale=1, sigma_wls=1.5, fbs_spatial=8.0, bilateral_on=False):

    """
    Args:
        rectifiedL : np.array()
            Image à analyser pour la caméra de gauche. L'image doit être rectifiée.
        rectifiedR : np.array()
            Image à analyser pour la caméra de gauche. L'image doit être rectifiée.
        Q : matrice
            Matrice Q calculée lors de la calibration (référentiel caméra de gauche)

        PARAMÈTRES POUR LE PREPROCESSING:
        downscale : int
            facteur de downscale des images pour l'analyse
        sigma_wls : float [0.8, 2]
            «SigmaColor is a parameter defining how sensitive the filtering process is to source image edges. Large values can lead to disparity leakage through low-contrast edges. Small values can make the filter too sensitive to noise and textures in the source image. Typical values range from 0.8 to 2.0.»
        fbs_spatial: float
            «Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough (see sigmaColor ).»
        bilateral_on : Bool
            True: un filtre fastBilateralSolverFilter est appliqué sur l'image
            False : le filtre n'est pas appliqué
    """

    # CREATION STEREO MATCHERS -------------------------------------------------
    num_disp = 5*16
    min_disp= 1
    wsize = 5
    left_matcher = cv.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=wsize,
            P1=24*wsize*wsize,
            P2=96*wsize*wsize,
            preFilterCap=63,
            mode=cv.STEREO_SGBM_MODE_SGBM_3WAY)
    right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
    # --------------------------------------------------------------------------

    # FORMATAGE PRE-CALCUL -----------------------------------------------------
    # DOWNSCALE
    # downscale=2
    new_num_disp = int(num_disp / downscale)
    n_width = int(rectifiedL.shape[1] * 1/downscale)
    n_height = int(rectifiedR.shape[0] * 1/downscale)
    downL = cv.resize(rectifiedL, (n_width, n_height))
    downR = cv.resize(rectifiedR, (n_width, n_height))

    # GRAY SCALE
    grayL_down = cv.cvtColor(downL,cv.COLOR_BGR2GRAY)
    grayR_down = cv.cvtColor(downR,cv.COLOR_BGR2GRAY)

    # SMOOTH
    grayL_down = cv.medianBlur(grayL_down,3)
    grayR_down = cv.medianBlur(grayR_down,3)
    # --------------------------------------------------------------------------

    # CALCULS DISPARITÉ --------------------------------------------------------
    displ = left_matcher.compute(cv.UMat(grayL_down),cv.UMat(grayR_down))
    dispr = right_matcher.compute(cv.UMat(grayR_down),cv.UMat(grayL_down))
    displ = np.int16(cv.UMat.get(displ))
    dispr = np.int16(cv.UMat.get(dispr))
    # --------------------------------------------------------------------------

    # WLS FILTER ---------------------------------------------------------------
    lambda_wls = 8000.0
    # sigma_wls = 1.5
    wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lambda_wls)
    wls_filter.setSigmaColor(sigma_wls)
    filtered_disp = wls_filter.filter(displ, rectifiedL, None, dispr)
    # --------------------------------------------------------------------------

    # MASK ---------------------------------------------------------------------
    # CONFIDENCE MAP
    conf_map=wls_filter.getConfidenceMap()
    # ROI
    ROI = np.array(wls_filter.getROI())*downscale
    #MASK
    mask=np.zeros(conf_map.shape, conf_map.dtype)
    mask[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]]=1
    # FORMAT DISPARITY
    disparity=filtered_disp.astype(np.float32)/16.0*mask
    # --------------------------------------------------------------------------

    # BILATERAL FILTER ---------------------------------------------------------
    if bilateral_on :
        # fbs_spatial=8.0
        fbs_luma=8.0
        fbs_chroma=8.0
        fbs_lambda=128.0
        solved_filtered_disp = cv.ximgproc.fastBilateralSolverFilter(rectifiedL, filtered_disp, conf_map/255.0, None, fbs_spatial, fbs_luma, fbs_chroma, fbs_lambda)
        disparity=solved_filtered_disp.astype(np.float32)/16.0*mask
    # --------------------------------------------------------------------------

    # REPROJECT TO 3D ----------------------------------------------------------
    # Note: If one uses Q obtained by stereoRectify, then the returned points are represented in the first camera's rectified coordinate system
    points = cv.reprojectImageTo3D(disparity, Q, handleMissingValues=True)

    # depth map
    depth_map=points[:,:,2]*mask
    # -------------------------------------------------------------------------

    return points, mask, depth_map


def save_mesh(rectified, points, mask, mesh_name):

    # SAVEGARDER MESH ----------------------------------------------------------
    colors = cv.cvtColor(rectified, cv.COLOR_BGR2RGB)
    colors_valides = colors[mask.astype(bool)]
    points_valides=points[mask.astype(bool)]
    mask2 = points_valides[:,2]<30
    out_fn = '{}.ply'.format(mesh_name)
    write_ply(out_fn, points_valides[mask2], colors_valides[mask2])
    # --------------------------------------------------------------------------


def get_median(corners_rec,points,n=5):
    pts_rec=[]
    for i in range(len(corners_rec)):
        col, row = int(np.round(corners_rec[i,0,0])), int(np.round(corners_rec[i,0,1]))
        # Moyenne :
        pts = points[row-n:row+n+1, col-n:col+n+1, :]
        xmed=np.median(pts[:,:,0])
        ymed=np.median(pts[:,:,1])
        zmed=np.median(pts[:,:,2])
        pt=np.array([xmed, ymed, zmed])
        # pt = points[row,col]
        pts_rec.append(pt)
    return np.array(pts_rec)




def coins_mesh(patternSize,rectified, points, winSize=(11,11)):
    """
    Arrange des vecteurs contenant les points théoriques et les points calculés à partir de la carte de disparité
    Args:
        rectified : image rectifiée
        points : nuage de point 3d dans le référentiel rectifié
    Returns:
        pts_rec
    """

    # DÉTECTION DES COINS DANS L'IMAGE RECTIFIÉE -------------------------------
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Trouver coins dans l'image originale prise par la caméra
    gray=cv.cvtColor(rectified, cv.COLOR_BGR2GRAY)
    ret, corners_rec = cv.findChessboardCorners(gray, patternSize, None)
    # assert ret==True, "coins non détectés"

    if ret :
        corners_rec = cv.cornerSubPix(gray, corners_rec, winSize,(-1, -1), criteria)
        # COINS DE LA CARTE DE PROFONDEUR --------------------------------------
        pts_rec=get_median(corners_rec, points, n=5).T #points 3D dans le réféntiel de la caméra rectifiée
        # ----------------------------------------------------------------------
        return pts_rec, corners_rec

    else:
        return None, None

def get_rec(objp, r, t, R, P ):
    # COINS THÉORIQUES ---------------------------------------------------------
    world=objp.T
    unrec=r@world+t #coins théoriques dans ref cam non rectifiée
    rec=R@unrec #points théoriques dans ref cam rectifiée
    # corners_rec, _ = cv.projectPoints(rec, np.zeros((3,1)), np.zeros((3,1)), P[:,:3], np.zeros((1,4))) #points théoriques dans ref image cam rectifiée
    # --------------------------------------------------------------------------
    return rec

def triangulation_rec( patternSize, rectifiedL, rectifiedR, P1, P2, winSize=(11,11) ):


    # POINTS IMAGES DÉTECTÉS (LA MESURE) ---------------------------------------
    # Référentiel rectifié
    grayl=cv.cvtColor(rectifiedL, cv.COLOR_BGR2GRAY)
    grayr=cv.cvtColor(rectifiedR, cv.COLOR_BGR2GRAY)
    # Détection des coins
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ret_l, corners_l = cv.findChessboardCorners(grayl, patternSize, None)
    ret_r, corners_r = cv.findChessboardCorners(grayr, patternSize, None)
    if ret_l*ret_r :
        corners_rec_l= cv.cornerSubPix(grayl, corners_l, winSize,(-1, -1), criteria)
        corners_rec_r= cv.cornerSubPix(grayr, corners_r, winSize,(-1, -1), criteria)
    # --------------------------------------------------------------------------

    # TRIANGULATION -----------------------------------------------------------
    # Formatage
    N=corners_rec_l.shape[0]
    projPoints1 = np.zeros((2,N));
    projPoints2 = np.zeros((2,N))
    for i in range(N):
        projPoints1[:,i]=corners_rec_l[i,0];
        projPoints2[:,i]=corners_rec_r[i,0]

    # triangulatePoints: If the projection matrices from stereoRectify are used, then the returned points are represented in the first camera's rectified coordinate system.
    points4D=cv.triangulatePoints(P1, P2, projPoints1, projPoints2)
    points3D=cv.convertPointsFromHomogeneous(points4D.T)
    X,Y,Z=points3D[:,0,0], points3D[:,0,1], points3D[:,0,2]
    points_rec1=np.stack((X,Y,Z)) # Référentiel rectifié
    # --------------------------------------------------------------------------

    return points_rec1

def err_points(patternSize, pts_th, pts_cal):
    """
    Calcule l'erreur entre les points théoriques pts_th et les points calculés pts_cal. Les points sont les coins d'un damier.
    Args:
        patternSize : (row,col)
        pts_th : points théoriques.
        pts_cal : points calculées.
    Returns:
        errtot : erreur RMS totale sur la position
        (errx,erry,errz): erreur RMS sur les coordonées x,y,z
                          - en coordonées cartésiennes -
    """

    arr=np.zeros((patternSize[1],patternSize[0]))
    x,y,z=arr.copy(), arr.copy(), arr.copy()
    xo,yo,zo=arr.copy(), arr.copy(), arr.copy()
    j=0;n=patternSize[0]
    for i in range(patternSize[1]):
        x[i,:]=pts_cal[0,j:j+n]; y[i,:]=pts_cal[1,j:j+n]; z[i,:]=pts_cal[2,j:j+n]
        xo[i,:]=pts_th[0,j:j+n]; yo[i,:]=pts_th[1,j:j+n]; zo[i,:]=pts_th[2,j:j+n]
        j+=n
    # x,y,z = pts_cal[0,:], pts_cal[1,:], pts_cal[2,:]
    # xo,yo,zo = pts_th[0,:], pts_th[1,:], pts_th[2,:]

    # Erreur en coordonées cartésiennes:
    errX=xo-x; errY=yo-y; errZ=zo-z
    # Erreurs RMS
    errx = np.sqrt((errX**2).mean())
    erry = np.sqrt((errY**2).mean())
    errz = np.sqrt((errZ**2).mean())
    errtot=np.sqrt( (errX**2 + errY**2 + errZ**2).mean() )

    zmean=np.mean(zo)

    return errtot, zmean



def triangulation_world(patternSize, squaresize, K1, K2, D1, D2, left, right):

    # LIRE IMAGES ET TROUVER POINTS --------------------------------------------
    grayl=cv.cvtColor(left, cv.COLOR_BGR2GRAY)
    grayr=cv.cvtColor(right, cv.COLOR_BGR2GRAY)
    # trouver
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ret_l, corners_l = cv.findChessboardCorners(grayl, patternSize, None)
    ret_r, corners_r = cv.findChessboardCorners(grayr, patternSize, None)
    if ret_l*ret_r :
        pts_l= cv.cornerSubPix(grayl, corners_l, (11, 11),(-1, -1), criteria)
        pts_r= cv.cornerSubPix(grayr, corners_r, (11, 11),(-1, -1), criteria)
    # --------------------------------------------------------------------------


    # MATRICES DE PROJECTION ---------------------------------------------------
    objp = coins_damier(patternSize,squaresize)
    ret, rvec1, t1 = cv.solvePnP(objp, pts_l, K1, D1 )
    ret, rvec2, t2 = cv.solvePnP(objp, pts_r, K2, D2 )
    r1, _=cv.Rodrigues(rvec1); r2, _=cv.Rodrigues(rvec2)
    p1=np.column_stack((r1,t1)); Proj1=K1@p1
    p2=np.column_stack((r2,t2)); Proj2=K2@p2
    # --------------------------------------------------------------------------

    # UNDISTORT POINTS ---------------------------------------------------------
    ptsl=cv.undistortPoints(pts_l, K1, D1, None, None, K1) #not rectification
    ptsr=cv.undistortPoints(pts_r, K2, D2, None, None, K2) #not rectification
    N=ptsl.shape[0]
    projPoints1 = np.zeros((2,N))
    projPoints2 = np.zeros((2,N))
    for i in range(N):
        projPoints1[:,i]=ptsl[i,0]
        projPoints2[:,i]=ptsr[i,0]
    # --------------------------------------------------------------------------

    # TRIANGULATION ------------------------------------------------------------
    points4D=cv.triangulatePoints(Proj1, Proj2, projPoints1, projPoints2)
    points3D=cv.convertPointsFromHomogeneous(points4D.T)

    X,Y,Z=points3D[:,0,0], points3D[:,0,1], points3D[:,0,2]
    points=np.stack((X,Y,Z))
    # --------------------------------------------------------------------------

    return points







def err_rt(objp, not_rectified, rectified, K, D, R, P, r, t ):

    # DÉTECTION DES COINS DANS L'IMAGE NON RECTIFIÉE ---------------------------
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Trouver coins dans l'image originale prise par la caméra de gauche
    gray=cv.cvtColor(not_rectified, cv.COLOR_BGR2GRAY)
    ret, corners_unrec = cv.findChessboardCorners(gray, patternSize, None)
    if ret :
        corners_unrec = cv.cornerSubPix(gray, corners_unrec, (11,11),(-1, -1), criteria)
    assert ret==True, "coins non détectés"
    # --------------------------------------------------------------------------

    # DÉTECTION DES COINS DANS L'IMAGE RECTIFIÉE -------------------------------
    # On détecte les coins dans l'image de gauche rectifiée.
    gray=cv.cvtColor(rectified, cv.COLOR_BGR2GRAY)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ret, corners_rec = cv.findChessboardCorners(gray, patternSize, None)
    if ret:
        corners_rec = cv.cornerSubPix(gray, corners_rec, (11,11),(-1, -1), criteria)
    assert ret==True, "coins non détectés"
    # --------------------------------------------------------------------------

    # PROJECTION DES COORD MONDE -----------------------------------------------
    # Coordonées des coins du damier dans le référentiel monde:
    world=objp.T
    # Rodrigues
    rvec, _ =cv.Rodrigues(r)

    # ->Référentiel image de la caméra non rectifiée
    img_unrec, _=cv.projectPoints(world, rvec, t, K, D )

    # ->Référentiel image de la caméra rectifiée
    unrec=r@world+t #coins théoriques dans ref cam non rectifiée
    rec=R@unrec #points théoriques dans ref cam rectifiée
    img_rec, _ = cv.projectPoints(rec, np.zeros((3,1)), np.zeros((3,1)), P[:,:3], np.zeros((1,4))) #points théoriques dans ref image cam rectifiée
    # --------------------------------------------------------------------------

    # COMPARAISON DES POINTS DÉTECTÉS ET CEUX PROJETÉS -------------------------
    # Référentiel non rectifié
    err_unrec = np.sqrt( np.mean( np.sum( (corners_unrec-img_unrec)**2, axis=2 )  )  )
    # Référentiel rectifié
    err_rec = np.sqrt( np.mean( np.sum( (corners_rec-img_rec)**2, axis=2 ) )  )
    # --------------------------------------------------------------------------
    return err_unrec, err_rec


def find_rt(patternSize, objp, not_rectified, K, D, winSize=(11,11)):
    """
    Trouver la transformation (r,t) qui amène du référentiel de la caméra rectifiée (Xc,Yc,Zc)_rec au référentiel monde (Xw, Yw,Zw)

    R1: performs a change of basis from the unrectified first camera's coordinate system to the rectified first camera's coordinate system.
    P1 : projects points given in the rectified first camera coordinate system into the rectified first camera's image
    r,t : performs a change of basis form the world coordinate system to the first camera's distorted and unrectifed coordinate system

    Coordonées cam rectifiée: rec=(Xc,Yc,Zc)_rec
    Coordonées cam non-rectifiée : unrec=R1.T@rec=(Xc,Yc,Zc)
    Coordonnées world : world=r.T@(unrec-t1)=(Xw,Yw,Zw)
    """

    # DÉTECTION DES COINS DANS L'IMAGE NON RECTIFIÉE ---------------------------
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Trouver coins dans l'image originale prise par la caméra
    gray=cv.cvtColor(not_rectified, cv.COLOR_BGR2GRAY)
    ret, corners_unrec = cv.findChessboardCorners(gray, patternSize, None)
    if ret :
        corners_unrec = cv.cornerSubPix(gray, corners_unrec, winSize,(-1, -1), criteria)
    assert ret==True, "coins non détectés"
    # --------------------------------------------------------------------------

    # RÉSOLUTION POUR TROUVER R,T (unrec=R@world+T) ----------------------------
    ret, rvec, t = cv.solvePnP(objp, corners_unrec, K, D )
    r,_=cv.Rodrigues(rvec)
    # --------------------------------------------------------------------------

    return ret, r, t
