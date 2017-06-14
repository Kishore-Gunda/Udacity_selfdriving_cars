import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from collections import deque
#%matplotlib inline

##########################################################################################################################

#calculate the distortion
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('camera_cal/calibration*.jpg')
count=0

for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (9,6), corners, ret)
        count += 1
        plt.clf()
        plt.imshow(img)
        plt.savefig('output_images/calibrated/output' + str(count) + '.jpg')

###############################################################################################################################
# Remove distortion from images
def undistort_image(image, count, Display=True, read = True):
    if read:
        img = cv2.imread(image)
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    undist = cv2.undistort_image(img, mtx, dist, None, mtx)
    if Display:
        count = count+1
        plt.clf()
        plt.imshow(cv2.cvtColor(undist, cv2.COLOR_BGR2RGB))
        plt.savefig('output_images/undistored/output_und' + str(count) + '.jpg')	
        return undist, count
    else:
        return undist, count

images = glob.glob('test_images/test*.jpg')
count =0
for image in images:
    undist, count = undistort_image(image,count)



###############################################################################################################################
# Perform perspective transform
def Bird_EyeView(img,count1, show=True, read = True):
    count= 0
    if read:
        undist, count = undistort_image(img,count = 0, Display = False)
    else:
        undist, count = undistort_image(img,count = 0, Display = False, read=False) 
    img_size = (undist.shape[1], undist.shape[0])
    offset = 0
    src = np.float32([[490, 482],[810, 482],
                      [1250, 720],[40, 720]])
    dst = np.float32([[0, 0], [1280, 0], 
                     [1250, 720],[40, 720]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(undist, M, img_size)
    if show:
        count1 = count1+1
        plt.clf()
        plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        plt.savefig('output_images/warped/output_wrp' + str(count1) + '.jpg')
        return warped, M,count1
    else:
        return warped, M,count1

count1 =0
for image in glob.glob('test_images/test*.jpg'):
    warped, M,count1= Bird_EyeView(image, count1)

###################################################################################################################################

# Create binary thresholded images to isolate lane line pixels
def channel_Selection_thresholds(image, count2, Display=True):
    img,count1, M = Bird_EyeView(image, count1=0, show = False)
   
    b_channel = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:,:,2]       
    
    l_channel = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)[:,:,0]
    
    l_thresh_min = 225
    l_thresh_max = 255
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1	

	
    b_thresh_min = 155
    b_thresh_max = 200
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh_min) & (b_channel <= b_thresh_max)] = 1

    
    combined_binary = np.zeros_like(l_binary)
    combined_binary[(l_binary == 1) | (b_binary == 1)] = 1

    if Display == True:
        count2 = count2+1
        plt.clf()
        plt.imshow(b_binary, cmap='gray')
        plt.savefig('output_images/apply_thresholds/b_binary/output_b_binary' + str(count2) + '.jpg')

        plt.clf()
        plt.imshow(l_binary, cmap='gray')
        plt.savefig('output_images/apply_thresholds/l_binary/output_l_binary' + str(count2) + '.jpg')		

        plt.clf()
        plt.imshow(combined_binary, cmap='gray')
        plt.savefig('output_images/apply_thresholds/combined_binary/output_combined_binary' + str(count2) + '.jpg')

        return combined_binary, count2        
        
    else: 
        return combined_binary, count2
count2 =0		
for image in glob.glob('test_images/test*.jpg'):
    combined_binary, count2 = channel_Selection_thresholds(image,count2)		
		
#################################################################################################################		
	