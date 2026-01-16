# -*- coding: utf-8 -*-
"""
Created on 2 Dec 2020
Updated 9 Jan 2023
@author: chatoux
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import glob

def CameraCalibration(Size,path,path1image, savename):
    # 
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    ############ to adapt ##########################
    objp = np.zeros((Size[0]*Size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:Size[0], 0:Size[1]].T.reshape(-1, 2)
    # 
    objp[:, :2] *= 40
    #################################################
    # 
    objpoints = []  # 
    imgpoints = []  # 
    ######################################
    images = glob.glob(path)
    #################################################
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # 
        ######################################
        ret, corners = cv.findChessboardCorners(gray, (Size[0],Size[1]), None)
        #################################################
        print(ret)
        #
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            # 
            ######################################
            cv.drawChessboardCorners(img, Size, corners2, ret)
            #################################################
            cv.namedWindow('img', cv.WINDOW_KEEPRATIO)
            cv.imshow('img', img)
            cv.waitKey(500)
    cv.destroyAllWindows()

    # 
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print('camraMatrix\n',mtx)
    print('dist\n',dist)

    ############ to adapt ##########################
    img = cv.imread(path1image)
    #################################################
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    print('newcameramtx\n',newcameramtx)

    # 
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
    # 
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv.namedWindow('img', cv.WINDOW_KEEPRATIO)
    cv.imshow('img', dst)
    cv.waitKey(0)
    ############ to adapt ##########################
    cv.imwrite(savename, dst)
    #################################################

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("total error: {}".format(mean_error / len(objpoints)))

    return newcameramtx, dst

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color, 2)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def StereoCalibrate(rect1, rect2):
    # opencv 4.5
    sift = cv.SIFT_create()
    # 
    kp1, des1 = sift.detectAndCompute(rect1, None)
    kp2, des2 = sift.detectAndCompute(rect2, None)
    # 
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    print(pts1.shape)

    # 
    img3 = cv.drawMatches(rect1,kp1,rect2,kp2,good, None, flags=2)
    cv.namedWindow('img', cv.WINDOW_KEEPRATIO)
    cv.imshow('img', img3)
    cv.waitKey(0)

    #
    identity = np.eye(3)
    E, maskE = cv.findEssentialMat(pts1, pts2, identity, method=cv.FM_LMEDS)
    print('E\n',E)
    # 
    retval, R, t, maskP = cv.recoverPose(E, pts1, pts2, identity, maskE)
    print('R\n', R)
    print('t\n', t)

    #
    F, maskF = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)
    print('F\n', F)

    return pts1, pts2, F, maskF

def EpipolarGeometry(pts1, pts2, F, maskF,img1, img2):
    r,c = img1.shape

    # 
    pts1F = pts1[maskF.ravel() == 1]
    pts2F = pts2[maskF.ravel() == 1]

    # 
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(img1,img2,lines1,pts1F,pts2F)
    # 
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

    cv.namedWindow('Fright1', cv.WINDOW_KEEPRATIO)
    cv.imshow('Fright1', img5)
    cv.namedWindow('Fright2', cv.WINDOW_KEEPRATIO)
    cv.imshow('Fright2', img6)

    cv.namedWindow('Fleft1', cv.WINDOW_KEEPRATIO)
    cv.imshow('Fleft1', img4)
    cv.namedWindow('Fleft2', cv.WINDOW_KEEPRATIO)
    cv.imshow('Fleft2', img3)
    cv.waitKey(0)


    plt.show()

    # 
    retval, H1, H2 = cv.stereoRectifyUncalibrated(pts1, pts2, F, (c,r))
    print(H1)
    print(H2)
    # 
    im_dst1 = cv.warpPerspective(img1, H1, (c,r))
    im_dst2 = cv.warpPerspective(img2, H2, (c,r))
    cv.namedWindow('left', 0)
    cv.imshow('left', im_dst1)
    cv.namedWindow('right', 0)
    cv.imshow('right', im_dst2)
    cv.waitKey(0)

def update_disparity(imgL, imgR):
    min_disp = cv.getTrackbarPos('Min Disp', 'Disparity')
    num_disp = cv.getTrackbarPos('Nb Disp', 'Disparity') * 16
    block_size = cv.getTrackbarPos('Block Size', 'Disparity')
    uniqueness_ratio = cv.getTrackbarPos('Uniq Ratio',
    'Disparity')
    speckle_window_size = cv.getTrackbarPos('Spec Window',
    'Disparity')
    speckle_range = cv.getTrackbarPos('Spec Range', 'Disparity')
    stereo = cv.StereoSGBM_create(minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=16,
    uniquenessRatio=uniqueness_ratio,
    speckleWindowSize=speckle_window_size,
    speckleRange=speckle_range)
    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    cv.imshow('Disparity', (disparity - min_disp) / num_disp)

def DepthMapfromStereoImages(imgL, imgR):
    # Définir les paramètres pour l'algorithme de correspondance stéréo
    global window_size
    window_size = 3
    min_disp = 16
    num_disp = 112 - min_disp
    stereo = cv.StereoSGBM_create(minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=16,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=16,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32)
    # Calculer la map de disparité
    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    disparity = cv.normalize(disparity, disparity, alpha=0, beta=255,
    norm_type=cv.NORM_MINMAX)
    disparity = np.uint8(disparity)
    cv.namedWindow('Disparity', cv.WINDOW_KEEPRATIO)
    # cv.resizeWindow('Disparity', 1980, 1280)
    cv.createTrackbar('Min Disp', 'Disparity', 16, 100, lambda x:
    update_disparity(imgL, imgR))
    cv.createTrackbar('Nb Disp', 'Disparity', 6, 20, lambda x:
    update_disparity(imgL, imgR))
    cv.createTrackbar('Block Size', 'Disparity', 16, 50, lambda x:
    update_disparity(imgL, imgR))
    cv.createTrackbar('Uniq Ratio', 'Disparity', 10, 50, lambda
    x: update_disparity(imgL, imgR))
    cv.createTrackbar('Spec Window', 'Disparity', 100, 200,
    lambda x: update_disparity(imgL, imgR))
    cv.createTrackbar('Spec Range', 'Disparity', 32, 100, lambda x:
    update_disparity(imgL, imgR))
    cv.imshow('Disparity', (disparity - min_disp) / num_disp)
    update_disparity(imgL, imgR)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    ############ to adapt ##########################
    path1 = 'Images/rouge/*.jpg'
    path2 = 'Images/bleu/*.jpg'
    # image à rectifier
    path1image = 'Images/rouge2.jpg'
    savename1 = 'Images/rouge2rect.png'
    path2image = 'Images/bleu2.jpg'
    savename2 = 'Images/bleu2rect.png'
    Size = [8, 5]
    #################################################
    cameraMatrix1, rect1 = CameraCalibration(Size,path1,path1image, savename1)
    cameraMatrix2, rect2 = CameraCalibration(Size,path2,path2image, savename2)
    ############ to adapt ##########################
    imageL = cv.imread(savename1, 0)
    imageR = cv.imread(savename2, 0)
    #################################################
    pts1, pts2, F, maskF = StereoCalibrate(imageL, imageR)

    EpipolarGeometry(pts1, pts2, F, maskF, imageL, imageR)

    ############ to adapt ##########################
    nameL = 'Images/aloeL.jpg'
    nameR = 'Images/aloeR.jpg'
    imageL = cv.imread(nameL, 0)
    imageR = cv.imread(nameR, 0)
    #################################################
    DepthMapfromStereoImages(imageL, imageR)

