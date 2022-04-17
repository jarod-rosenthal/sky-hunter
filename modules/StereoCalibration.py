import numpy as np
import cv2 as cv
import glob, os

from modules.util import draw_reprojection, clean_folders, coins_damier, find_corners


# Classe contenant les fonctions de calibration
class StereoCalibration():

    def __init__(self, patternSize, squaresize):
        """
        || Constructeur ||
        patternSize = (nb points per row, nb points per col)
        squaresize : taille damier en mètres
        """

        # Damier
        self.patternSize=patternSize
        self.squaresize=squaresize #utiliser des unités SI svp
        self.objp = coins_damier(patternSize,squaresize)
        self.color_flag=cv.COLOR_BGR2GRAY

        # Critères ------------------------------------------------------------
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        # ----------------------------------------------------------------------

        # Flags de claibration -------------------------------------------------
        self.not_fisheye_flags = 0
        # self.not_fisheye_flags = cv.CALIB_FIX_K3|cv.CALIB_ZERO_TANGENT_DIST
        self.fisheye_flags=cv.CALIB_RATIONAL_MODEL|cv.CALIB_FIX_K5|cv.CALIB_FIX_K6|cv.CALIB_ZERO_TANGENT_DIST
        # ----------------------------------------------------------------------


        # Déclaration des attributs --------------------------------------------
        # Folders contenant les images à analyser
        self.single_path=None
        self.stereo_path=None

        # Folders ou enregistrer les images détectées
        self.single_detected_path=None
        self.stereo_detected_path=None

        # Variables pour la calibration
        self.imageSize1, self.imageSize2 = None, None
        self.err1, self.M1, self.d1, self.r1, self.t1 = None, None, None, None, None
        self.err2, self.M2, self.d2, self.r2, self.t2 = None, None, None, None, None
        self.errStereo, self.R, self.T = None, None, None
        # points de la calibration individuelle
        self.objpoints_l,self.objpoints_r, self.imgpoints_l, self.imgpoints_r=None, None, None, None

        # Pour que dessine
        self.draw=False
        # ----------------------------------------------------------------------



    def __read_single(self, view):
        """
        ||Private method||
        view  (str): 'left' ou 'right'
        """
        images = np.sort(glob.glob(self.single_path + '{}*.jpg'.format(view)))
        assert len(images) != 0, "Images pas trouvées. Vérifier le path"

        # Parcours le dossier pour trouver le damier sur les images
        objpoints=[]; imgpoints=[]
        for i in range(len(images)):
            color = cv.imread(images[i])
            gray = cv.cvtColor(color, self.color_flag)
            ret, corners = cv.findChessboardCorners(gray, self.patternSize, None)
            if ret==True:
                objpoints.append(self.objp)
                corners2= cv.cornerSubPix(gray, corners, (3, 3),(-1, -1), self.criteria);
                imgpoints.append(corners2)
                # Dessiner les chessboard -------------------------------------
                if self.draw==True:
                    _ = cv.drawChessboardCorners(color, self.patternSize, corners2, True)
                    fname='{}{:03d}.jpg'.format(view, i+1)
                    cv.imwrite(self.single_detected_path + fname, color)
                # --------------------------------------------------------------
        imageSize=gray.shape
        return objpoints, imgpoints, imageSize


    def __read_stereo(self, draw=False):
        """
        ||Private method||
        """
        images_left = np.sort(glob.glob(self.stereo_path + 'left*.jpg'))
        images_right = np.sort(glob.glob(self.stereo_path + 'right*.jpg'))
        assert (len(images_right) != 0 or len(images_left) != 0  ), "Images pas trouvées. Vérifier le path"

        objpoints=[]; imgpoints_left=[]; imgpoints_right=[]
        for i in range(len(images_right)):

            color_l = cv.imread(images_left[i])
            gray_l = cv.cvtColor( color_l, self.color_flag)
            ret_l, corners_l = cv.findChessboardCorners(gray_l, self.patternSize, None)

            color_r = cv.imread(images_right[i])
            gray_r = cv.cvtColor( color_r, self.color_flag)
            ret_r, corners_r = cv.findChessboardCorners(gray_r, self.patternSize, None)

            if ret_l*ret_r==1:
                objpoints.append(self.objp)

                corners2_l= cv.cornerSubPix(gray_l, corners_l, (3, 3),(-1, -1), self.criteria_cal)
                imgpoints_left.append(corners2_l)
                corners2_r= cv.cornerSubPix(gray_r, corners_r, (3, 3),(-1, -1), self.criteria_cal)
                imgpoints_right.append(corners2_r)

                # Dessiner les chessboard -------------------------------------
                if self.draw==True:
                    _ = cv.drawChessboardCorners(color_l, self.patternSize, corners2_l, True)
                    fname='{}{:03d}.jpg'.format('left', i+1)
                    cv.imwrite(self.stereo_detected_path + fname, color_l)
                    _ = cv.drawChessboardCorners(color_r, self.patternSize, corners2_r, True)
                    fname='{}{:03d}.jpg'.format('right', i+1)
                    cv.imwrite(self.stereo_detected_path + fname, color_r)
                # --------------------------------------------------------------

        return objpoints, imgpoints_left, imgpoints_right

    def calibrateSingle(self, single_path, single_detected_path, fisheye=False, draw=False):
        """
        ||Public method||
        Calibration individuelle de 2 caméras

        Args:
            single_path (str): "path_to_single_images/"
            single_detected_path (str): "path_to_single_images_detected/"
            fisheye (Bool): True pour caméra fisheye
        """
        self.single_path=single_path
        self.single_detected_path=single_detected_path
        clean_folders([single_detected_path])

        self.objpoints_l, self.imgpoints_l, self.imageSize1 = self.__read_single('left')
        self.objpoints_r, self.imgpoints_r, self.imageSize2 = self.__read_single('right')

        if fisheye==True:
            flags=self.fisheye_flags
        else:
            flags=self.not_fisheye_flags

        # GAUCHE
        self.err1, self.M1, self.d1, self.r1, self.t1, stdDeviationsIntrinsics1, stdDeviationsExtrinsics1, self.perViewErrors1 = cv.calibrateCameraExtended(self.objpoints_l, self.imgpoints_l, self.imageSize1, None, None, flags=flags)

        # DROITE
        self.err2, self.M2, self.d2, self.r2, self.t2, stdDeviationsIntrinsics2, stdDeviationsExtrinsics2, self.perViewErrors2 = cv.calibrateCameraExtended(self.objpoints_r, self.imgpoints_r, self.imageSize2, None, None, flags=flags)

        # Enlever les outliers -------------------------------------------------
        # gauche
        objpoints_l=np.array(self.objpoints_l.copy())
        imgpoints_l=np.array(self.imgpoints_l.copy())
        indices=np.indices(self.perViewErrors1.shape)[0]
        self.indexes_l=indices[self.perViewErrors1<self.err1*1]
        if len(self.indexes_l)>0:
            objpoints_l=objpoints_l[self.indexes_l]
            imgpoints_l=imgpoints_l[self.indexes_l]
            self.err1, self.M1, self.d1, self.r1, self.t1, stdDeviationsIntrinsics1, stdDeviationsExtrinsics1, self.perViewErrors1 = cv.calibrateCameraExtended(objpoints_l, imgpoints_l, self.imageSize1, None, None, flags=flags)

        # droite
        objpoints_r=np.array(self.objpoints_r.copy())
        imgpoints_r=np.array(self.imgpoints_r.copy())
        indices=np.indices(self.perViewErrors2.shape)[0]
        self.indexes_r=indices[self.perViewErrors2<self.err2*1]
        if len(self.indexes_r)>0:
            objpoints_r=objpoints_r[self.indexes_r]
            imgpoints_r=imgpoints_r[self.indexes_r]
            self.err2, self.M2, self.d2, self.r2, self.t2, stdDeviationsIntrinsics2, stdDeviationsExtrinsics2, self.perViewErrors2 = cv.calibrateCameraExtended(objpoints_r, imgpoints_r, self.imageSize2, None, None, flags=flags)
        # ----------------------------------------------------------------------

        # Print erreur de reprojection
        print('Erreur de reprojection RMS calibration individuelle')
        print(self.err1, self.err2)


    def calibrateStereo(self, stereo_path, stereo_detected_path, fisheye=False, calib_2_sets=False, draw=False):
        """
        ||Public method||
        Calibration stéréo de 2 caméra

        Args:
            stereo_path (str): "path_to_stereo_images/"
            fisheye (Bool): True pour caméra fisheye
            calib_2_sets (Bool):
                True: pour utiliser des sets de photos différents (un pour la calibration individuelle des caméra et un pour la calibration stéréo)
                False: pour utiliser un seul set de photo pour la calibration individuelle et la calibration stéréo
            single_path: "path_to_single_images/"
        """
        self.stereo_path=stereo_path
        self.stereo_detected_path=stereo_detected_path
        clean_folders([stereo_detected_path])
        flags=0

        if fisheye:
            flags += self.fisheye_flags
        else:
            flags += self.not_fisheye_flags

        if self.R is not None:
            flags+= cv.CALIB_USE_EXTRINSIC_GUESS

        if calib_2_sets:
            flags+= cv.CALIB_FIX_INTRINSIC
        else:
            flags+= cv.CALIB_USE_INTRINSIC_GUESS

        print('calibration stéréo')
        # calibration stereo
        objpoints, imgpoints_l, imgpoints_r = self.__read_stereo()

        self.errStereo, _, _, _, _, self.R, self.T, self.E, self.F, self.perViewErrors = cv.stereoCalibrateExtended(objpoints, imgpoints_l, imgpoints_r, self.M1, self.d1, self.M2,self.d2, self.imageSize1, self.R, self.T, flags=flags)
        print(len(self.perViewErrors))

        # Enlever les outliers -------------------------------------------------
        indices=np.indices(self.perViewErrors.shape)[0]
        b=self.perViewErrors<self.errStereo*1
        B=b[:,0]*b[:,1]
        indexes=indices[B][:,0]
        if len(indexes)>0:
            objpoints=np.array(objpoints)[indexes]
            imgpoints_l=np.array(imgpoints_l)[indexes]
            imgpoints_r=np.array(imgpoints_r)[indexes]
            # re-calculs
            self.errStereo, _, _, _, _, self.R, self.T, self.E, self.F, self.perViewErrors = cv.stereoCalibrateExtended(objpoints, imgpoints_l, imgpoints_r, self.M1, self.d1, self.M2,self.d2, self.imageSize1, self.R, self.T, flags=flags)
            print(len(self.perViewErrors))
        # ----------------------------------------------------------------------

        # Print erreur de reprojection
        print('Erreur de reprojection RMS calibration stereo')
        print(self.errStereo)

    def calibrateIntrinsics(self, single_path, single_detected_path, cibles, cibles_l, cibles_r, fisheye=False):
        """
        ||Public method||
        Calibration individuelle de 2 caméras AVEC DES CIBLES

        Args:
            single_path (str): "path_to_single_images/"
            single_detected_path (str): "path_to_single_images_detected/"
            fisheye (Bool): True pour caméra fisheye
        """
        self.single_path=single_path
        self.single_detected_path=single_detected_path
        clean_folders([single_detected_path])

        self.objpoints_l, self.imgpoints_l, self.imageSize1 = self.__read_single('left')
        self.objpoints_r, self.imgpoints_r, self.imageSize2 = self.__read_single('right')

        if fisheye==True:
            flags=self.fisheye_flags
        else:
            flags=self.not_fisheye_flags

        # GAUCHE
        self.err1, self.M1, self.d1, self.r1, self.t1, stdDeviationsIntrinsics1, stdDeviationsExtrinsics1, self.perViewErrors1 = cv.calibrateCameraExtended(self.objpoints_l, self.imgpoints_l, self.imageSize1, None, None, flags=flags)

        # DROITE
        self.err2, self.M2, self.d2, self.r2, self.t2, stdDeviationsIntrinsics2, stdDeviationsExtrinsics2, self.perViewErrors2 = cv.calibrateCameraExtended(self.objpoints_r, self.imgpoints_r, self.imageSize2, None, None, flags=flags)

        # Enlever les outliers -------------------------------------------------
        # gauche
        objpoints_l=np.array(self.objpoints_l.copy())
        imgpoints_l=np.array(self.imgpoints_l.copy())
        indices=np.indices(self.perViewErrors1.shape)[0]
        self.indexes_l=indices[self.perViewErrors1<self.err1*1]
        if len(self.indexes_l)>0:
            objpoints_l=objpoints_l[self.indexes_l]
            imgpoints_l=imgpoints_l[self.indexes_l]
        # droite
        objpoints_r=np.array(self.objpoints_r.copy())
        imgpoints_r=np.array(self.imgpoints_r.copy())
        indices=np.indices(self.perViewErrors2.shape)[0]
        self.indexes_r=indices[self.perViewErrors2<self.err2*1]
        if len(self.indexes_r)>0:
            objpoints_r=objpoints_r[self.indexes_r]
            imgpoints_r=imgpoints_r[self.indexes_r]
        # ----------------------------------------------------------------------

        # Ajouter les points des cibles
        objpoints_l=list(objpoints_l)
        objpoints_r=list(objpoints_r)
        imgpoints_l=list(imgpoints_l)
        imgpoints_r=list(imgpoints_r)

        objpoints_l.append(cibles)
        objpoints_r.append(cibles)
        imgpoints_l.append(cibles_l)
        imgpoints_r.append(cibles_r)

        self.err1, self.M1, self.d1, self.r1, self.t1, stdDeviationsIntrinsics1, stdDeviationsExtrinsics1, self.perViewErrors1 = cv.calibrateCameraExtended(objpoints_l, imgpoints_l, self.imageSize1, None, None, flags=flags)
        self.err2, self.M2, self.d2, self.r2, self.t2, stdDeviationsIntrinsics2, stdDeviationsExtrinsics2, self.perViewErrors2 = cv.calibrateCameraExtended(objpoints_r, imgpoints_r, self.imageSize2, None, None, flags=flags)

        # Print erreur de reprojection
        print('Erreur de reprojection RMS calibration individuelle')
        print(self.err1, self.err2)


    def calibrateExtrinsics(self, stereo_path, stereo_detected_path, cibles, cibles_l, cibles_r, fisheye=False, calib_2_sets=False):
        """
        ||Public method||
        Calibration stéréo de deux caméra AVEC DES CIBLES
        Args:
            stereo_path (str): "path_to_stereo_images/"
            fisheye (Bool): True pour caméra fisheye
            calib_2_sets (Bool):
                True: pour utiliser des sets de photos différents (un pour la calibration individuelle des caméra et un pour la calibration stéréo)
                False: pour utiliser un seul set de photo pour la calibration individuelle et la calibration stéréo
            single_path: "path_to_single_images/"
        """
        self.stereo_path=stereo_path
        self.stereo_detected_path=stereo_detected_path
        clean_folders([stereo_detected_path])
        flags=0

        if fisheye:
            flags += self.fisheye_flags
        else:
            flags += self.not_fisheye_flags

        if self.R is not None:
            flags+= cv.CALIB_USE_EXTRINSIC_GUESS

        if calib_2_sets:
            flags+= cv.CALIB_FIX_INTRINSIC
        else:
            flags+= cv.CALIB_USE_INTRINSIC_GUESS

        print('calibration stéréo')
        # calibration stereo
        objpoints, imgpoints_l, imgpoints_r = self.__read_stereo()

        self.errStereo, _, _, _, _, self.R, self.T, self.E, self.F, self.perViewErrors = cv.stereoCalibrateExtended(objpoints, imgpoints_l, imgpoints_r, self.M1, self.d1, self.M2,self.d2, self.imageSize1, self.R, self.T, flags=flags)
        print(len(self.perViewErrors))

        # Enlever les outliers -------------------------------------------------
        indices=np.indices(self.perViewErrors.shape)[0]
        b=self.perViewErrors<self.errStereo*1
        B=b[:,0]*b[:,1]
        indexes=indices[B][:,0]
        if len(indexes)>0:
            objpoints=np.array(objpoints)[indexes]
            imgpoints_l=np.array(imgpoints_l)[indexes]
            imgpoints_r=np.array(imgpoints_r)[indexes]
        # ----------------------------------------------------------------------


        # Ajouter les points des cibles
        objpoints=list(objpoints)
        imgpoints_l=list(imgpoints_l)
        imgpoints_r=list(imgpoints_r)

        objpoints.append(cibles)
        imgpoints_l.append(cibles_l)
        imgpoints_r.append(cibles_r)

        self.errStereo, _, _, _, _, self.R, self.T, self.E, self.F, self.perViewErrors = cv.stereoCalibrateExtended(objpoints, imgpoints_l, imgpoints_r, self.M1, self.d1, self.M2,self.d2, self.imageSize1, self.R, self.T, flags=flags)
        print(len(self.perViewErrors))

        # Print erreur de reprojection
        print('Erreur de reprojection RMS calibration stereo')
        print(self.errStereo)



    def saveResultsXML(self, left_name='cam1', right_name='cam2'):

        # Enregistrer caméra 1:
        s = cv.FileStorage()
        s.open('{}.xml'.format(left_name), cv.FileStorage_WRITE)
        s.write('K', self.M1)
        s.write('R', np.eye(3))
        s.write('t', np.zeros((1,3)))
        s.write('coeffs', self.d1)
        s.write('imageSize', self.imageSize1)
        s.write('E', self.E)
        s.write('F', self.F)
        s.release()

        # Enregistrer caméra 2:
        s = cv.FileStorage()
        s.open('{}.xml'.format(right_name), cv.FileStorage_WRITE)
        s.write('K', self.M2)
        s.write('R', self.R)
        s.write('t', self.T)
        s.write('coeffs', self.d2)
        s.write('imageSize', self.imageSize2)
        s.write('E', self.E)
        s.write('F', self.F)
        s.release()

    def reprojection(self, folder, number=None):
        """ Dessiner la reprojection """
        clean_folders([folder])

        images_left = np.sort(glob.glob(self.single_path + 'left*.jpg'))
        images_right = np.sort(glob.glob(self.single_path + 'right*.jpg'))
        assert (len(images_right) != 0 or len(images_left) != 0  ), "Images pas trouvées. Vérifier le path"
        if number is not None:
            images_left = images_left[:number]
            images_right = images_right[:number]

        index=0
        for i in range(len(images_left)):
            ret_l, corners_l, gray_l = find_corners(images_left[i], self.patternSize)
            if ret_l==True:
                draw_reprojection(cv.imread(images_left[i]), self.objpoints_l[index], self.imgpoints_l[index], self.M1, self.d1, self.patternSize, self.squaresize, folder, "left_{}".format(i))
                index+=1

        jndex=0
        for j in range(len(images_right)):
            ret_r, corners_r, gray_r = find_corners(images_right[j], self.patternSize)
            if ret_r==True:
                draw_reprojection(cv.imread(images_right[j]), self.objpoints_r[jndex], self.imgpoints_r[jndex], self.M2, self.d2, self.patternSize, self.squaresize, folder, "right_{}".format(j))
                jndex+=1
