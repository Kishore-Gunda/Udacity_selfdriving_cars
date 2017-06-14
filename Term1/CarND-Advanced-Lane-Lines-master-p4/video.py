import numpy as np
import cv2
import glob
import Preprocessing
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Preprocessing import Bird_EyeView
from moviepy.editor import VideoFileClip
from collections import deque
#%matplotlib inline


###########################################################################################
#correct the distortion (duplicated)
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

####################################################################################################

class Line_pipeline:
        
    def search_if_found(selfxy, x, y):
        '''
        This function is called when the lanes lines are detected, it will search in the close proximity of -+ 25 pixels. 
        '''
        xvalues = []
        yvalues = []
        if selfxy.found == True: 
            i = 720
            j = 630
            while j >= 0:
                yval = np.mean([i,j])
                xval = (np.mean(selfxy.fit0))*yval**2 + (np.mean(selfxy.fit1))*yval + (np.mean(selfxy.fit2))
                x_idx = np.where((((xval - 25) < x)&(x < (xval + 25))&((y > j) & (y < i))))
                x_window, y_window = x[x_idx], y[x_idx]
                if np.sum(x_window) != 0:
                    np.append(xvalues, x_window)
                    np.append(yvalues, y_window)
                i -= 90
                j -= 90
        if np.sum(xvalues) == 0: 
            selfxy.found = False 
        return xvalues, yvalues, selfxy.found
		
    def values_sort(selfxy, xvalues, yvalues):
        sorted_index = np.argsort(yvalues)
        sorted_yvalues = yvalues[sorted_index]
        sorted_xvalues = xvalues[sorted_index]
        return sorted_xvalues, sorted_yvalues  
		
    def search_if_not_found(selfxy, x, y, image):
        '''
        This is called initially if the lane lines are not dectected 
        '''
        xvalues = []
        yvalues = []
        if selfxy.found == False: 
            i = 720
            j = 630
            while j >= 0:
                histogram = np.sum(image[j:i,:], axis=0)
                if selfxy == Right_ppl:
                    peak = np.argmax(histogram[640:]) + 640
                else:
                    peak = np.argmax(histogram[:640])
                x_idx = np.where((((peak - 25) < x)&(x < (peak + 25))&((y > j) & (y < i))))
                x_window, y_window = x[x_idx], y[x_idx]
                if np.sum(x_window) != 0:
                    xvalues.extend(x_window)
                    yvalues.extend(y_window)
                i -= 90
                j -= 90
        if np.sum(xvalues) > 0:
            selfxy.found = True
        else:
            yvalues = selfxy.Y
            xvalues = selfxy.X
        return xvalues, yvalues, selfxy.found
		
    def calculate_coordinates(selfxy, polynomial):
        bottom = polynomial[0]*720**2 + polynomial[1]*720 + polynomial[2]
        top = polynomial[0]*0**2 + polynomial[1]*0 + polynomial[2]
        return bottom, top    
		
    def radius_curvature(selfxy, xvalues, yvalues):
        ym_per_pix = 30./720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meteres per pixel in x dimension
        fit_cr = np.polyfit(yvalues*ym_per_pix, xvalues*xm_per_pix, 2)
        curverad = ((1 + (2*fit_cr[0]*np.max(yvalues) + fit_cr[1])**2)**1.5) \
                                     /np.absolute(2*fit_cr[0])
        return curverad  
    def __init__(selfxy):
        # check if the line was found previously ?
        selfxy.found = False
        
        # Remember x and y values of lanes in previous frame
        selfxy.X = None
        selfxy.Y = None
        
        # Store recent x intercepts to average across frames
        selfxy.x_int = deque(maxlen=10)
        selfxy.top = deque(maxlen=10)
        
        # compare previous and current frames x intercept
        selfxy.lastx_int = None
        selfxy.last_top = None
        
        # store radius of curvature
        selfxy.radius = None
        
        # Store recent polynomial coefficients for averaging across frames
        selfxy.fit0 = deque(maxlen=10)
        selfxy.fit1 = deque(maxlen=10)
        selfxy.fit2 = deque(maxlen=10)
        selfxy.fitx = None
        selfxy.pts = []
        
        # Count the number of frames
        selfxy.count = 0
		
# Video Processing Pipeline
################################################# COMMENTS ##################################################################
#
#Step4: Fitting a polynomial to the lane lines
#
# - Identifying the peaks in histograms using the with histogram
#   function.
# - Find the all the non zero pixels near to histogram using the
#  np.nonzero function
# - fitting a polynomial to each line to using nummpy function
#   fit.polyfit
#
##Step5: Calculation the vehical position
#- The center of the car was taken averaging the right and left polynomials 
#- The center of the road was calculated by taking the horizonal axis divide by 2 
#- Now the distance from the center of the road to the vehical is calculated by substracting the value from step1 and step 2
#- The pixel value are converted to meter by multiplying by 3.7/700
#
##Step6: Radius of curvature
# Below code is used to calculate the radius of curvature
#
#        ym_per_pix = 30./720 # meters per pixel in y dimension
#    xm_per_pix = 3.7/700 # meteres per pixel in x dimension
#    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
#    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
#    left_curverad = ((1 + (2*left_fit_cr[0]*np.max(lefty) + left_fit_cr[1])**2)**1.5) \
#                                 /np.absolute(2*left_fit_cr[0])
#    right_curverad = ((1 + (2*right_fit_cr[0]*np.max(lefty) + right_fit_cr[1])**2)**1.5) \
#                                    /np.absolute(2*right_fit_cr[0])
#                                   
# Source:  http://www.intmath.com/applications-differentiation/8-radius-curvature.php
# 
#Average was taken for final radius 

######################################################################################################################
def process_vid(image):
    img_size = (image.shape[1], image.shape[0])
    
    # Calibrate camera and undistort_image image
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    undistort_image = cv2.undistort_image(image, mtx, dist, None, mtx)
    
    # Perform perspective transform
    offset = 0
    src = np.float32([[490, 482],[810, 482],
                      [1250, 720],[0, 720]])
    dst = np.float32([[0, 0], [1280, 0], 
                     [1250, 720],[40, 720]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(undistort_image, M, img_size)
    
    # Generate binary thresholded images
    b_channel = cv2.cvtColor(warped, cv2.COLOR_RGB2Lab)[:,:,2]
    l_channel = cv2.cvtColor(warped, cv2.COLOR_RGB2LUV)[:,:,0]  
    
    # Set the upper and lower thresholds for the b channel
    b_thresh_min = 145
    b_thresh_max = 200
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh_min) & (b_channel <= b_thresh_max)] = 1
    
    # Set the upper and lower thresholds for the l channel
    l_thresh_min = 215
    l_thresh_max = 255
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

    combined_binary = np.zeros_like(b_binary)
    combined_binary[(l_binary == 1) | (b_binary == 1)] = 1
    
    # Identify all non zero pixels in the image
    x, y = np.nonzero(np.transpose(combined_binary)) 

    if Left_ppl.found == True: # Search for left lane pixels around previous polynomial
        leftxs, leftys, Left_ppl.found = Left_ppl.search_if_found(x, y)
        
    if Right_ppl.found == True: # Search for right lane pixels around previous polynomial
        rightxs, rightys, Right_ppl.found = Right_ppl.search_if_found(x, y)

            
    if Right_ppl.found == False: # Perform blind search for right lane lines
        rightxs, rightys, Right_ppl.found = Right_ppl.search_if_not_found(x, y, combined_binary)
            
    if Left_ppl.found == False:# Perform blind search for left lane lines
        leftxs, leftys, Left_ppl.found = Left_ppl.search_if_not_found(x, y, combined_binary)

    leftys = np.array(leftys).astype(np.float32)
    leftxs = np.array(leftxs).astype(np.float32)
    rightys = np.array(rightys).astype(np.float32)
    rightxs = np.array(rightxs).astype(np.float32)
            
    # define left line with available pixels
    left_line = np.polyfit(leftys, leftxs, 2)
    
    # intercepts calculation to extend for top and bottom of warped image
    leftxs_int, left_top = Left_ppl.calculate_coordinates(left_line)
    
    # Average intercepts for n frames
    Left_ppl.x_int.append(leftxs_int)
    Left_ppl.top.append(left_top)
    leftxs_int = np.mean(Left_ppl.x_int)
    left_top = np.mean(Left_ppl.top)
    Left_ppl.lastx_int = leftxs_int
    Left_ppl.last_top = left_top
    
    # Add averaged intercepts tox and y vals
    leftxs = np.append(leftxs, leftxs_int)
    leftys = np.append(leftys, 720)
    leftxs = np.append(leftxs, left_top)
    leftys = np.append(leftys, 0)
    
    # Sort pixels based on yvalues
    leftxs, leftys = Left_ppl.values_sort(leftxs, leftys)
    
    Left_ppl.X = leftxs
    Left_ppl.Y = leftys
    
    # again calcualte polynomial with intercepts and average across n frames
    left_line = np.polyfit(leftys, leftxs, 2)
    Left_ppl.fit0.append(left_line[0])
    Left_ppl.fit1.append(left_line[1])
    Left_ppl.fit2.append(left_line[2])
    left_line = [np.mean(Left_ppl.fit0), 
                np.mean(Left_ppl.fit1), 
                np.mean(Left_ppl.fit2)]
    
    # Fit polynomial to detected pixels
    left_fitx = left_line[0]*leftys**2 + left_line[1]*leftys + left_line[2]
    Left_ppl.fitx = left_fitx
    
    # Calculate right polynomial fit based on detected pixels
    rightx_fit = np.polyfit(rightys, rightxs, 2)

    # Calculate intercepts to extend the polynomial to the top and bottom of warped image
    rightx_int, right_top = Right_ppl.calculate_coordinates(rightx_fit)
    
    # Average intercepts across 5 frames
    Right_ppl.x_int.append(rightx_int)
    rightx_int = np.mean(Right_ppl.x_int)
    Right_ppl.top.append(right_top)
    right_top = np.mean(Right_ppl.top)
    Right_ppl.lastx_int = rightx_int
    Right_ppl.last_top = right_top
    rightxs = np.append(rightxs, rightx_int)
    rightys = np.append(rightys, 720)
    rightxs = np.append(rightxs, right_top)
    rightys = np.append(rightys, 0)
    
    # Sort right lane pixels
    rightxs, rightys = Right_ppl.values_sort(rightxs, rightys)
    Right_ppl.X = rightxs
    Right_ppl.Y = rightys
    
    # Recalculate polynomial with intercepts and average across n frames
    rightx_fit = np.polyfit(rightys, rightxs, 2)
    Right_ppl.fit0.append(rightx_fit[0])
    Right_ppl.fit1.append(rightx_fit[1])
    Right_ppl.fit2.append(rightx_fit[2])
    rightx_fit = [np.mean(Right_ppl.fit0), np.mean(Right_ppl.fit1), np.mean(Right_ppl.fit2)]
    
    # Fit polynomial to detected pixels
    right_fitx = rightx_fit[0]*rightys**2 + rightx_fit[1]*rightys + rightx_fit[2]
    Right_ppl.fitx = right_fitx
        
    #Compute radius of curvature for each lane in meters
    left_curverad = Left_ppl.radius_curvature(leftxs, leftys)
    right_curverad = Right_ppl.radius_curvature(rightxs, rightys)
        
    # take radius for every 3 frames
    if Left_ppl.count % 3 == 0:
        Left_ppl.radius = left_curverad
        Right_ppl.radius = right_curverad
        
    # calculate the lane position w.r.t to lane center
    position = (rightx_int+leftxs_int)/2
    distance_from_center = abs((640 - position)*3.7/700) 
                
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    warp_zero = np.zeros_like(combined_binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts_left = np.array([np.flipud(np.transpose(np.vstack([Left_ppl.fitx, Left_ppl.Y])))])
    pts_right = np.array([np.transpose(np.vstack([right_fitx, Right_ppl.Y]))])
    pts = np.hstack((pts_left, pts_right))
    cv2.polylines(color_warp, np.int_([pts]), isClosed=False, color=(0,0,255), thickness = 40)
    cv2.fillPoly(color_warp, np.int_(pts), (34,255,34))
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    result = cv2.addWeighted(undistort_image, 1, newwarp, 0.5, 0)

    # Print distance from center on video
    if position > 0:
        cv2.putText(result,'Position = {:1.2}m left'.format(distance_from_center), (np.int(img_size[0]/2)-100,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    else:
        cv2.putText(result,'Position = {:1.2}m right'.format(-distance_from_center), (np.int(img_size[0]/2)-100,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

    cv2.putText(result,'Left curve radius = {:.0f}m'.format(Left_ppl.radius), (np.int(img_size[0]/2)-100,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    cv2.putText(result,'Right curve radius = {:.0f}m'.format(Right_ppl.radius), (np.int(img_size[0]/2)-100,150), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
         
    Left_ppl.count += 1  
    return result
	
############################################## project video ################################
	
Left_ppl = Line_pipeline()
Right_ppl = Line_pipeline()
video_output = 'result.mp4'
video = VideoFileClip("project_video.mp4")
annotated_video = video.fl_image(process_vid) 
annotated_video.write_videofile(video_output, audio=False)	

################################ Challange ################################################
Left_ppl = Line_pipeline()
Right_ppl = Line_pipeline()
challenge_output = 'challenge_result.mp4'
video = VideoFileClip("challenge_video.mp4")
challenge_clip = video.fl_image(process_vid) 
challenge_clip.write_videofile(challenge_output, audio=False)
	