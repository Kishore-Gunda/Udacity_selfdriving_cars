import numpy as np
import cv2
import glob
import Preprocessing
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Preprocessing import channel_Selection_thresholds
from Preprocessing import Bird_EyeView
from moviepy.editor import VideoFileClip
from collections import deque
#%matplotlib inline

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
def find_fill_lanes(image, count3):
    
    combined_binary, count2 = channel_Selection_thresholds(image,count2=0, Display=False)
    img,count1, M = Bird_EyeView(image,count1=0, show = False)
    right_p_x = []
    right_p_y = []
    left_p_x = []
    left_p_y = []
    
    x, y = np.nonzero(np.transpose(combined_binary))
    i = 720
    j = 630
    while j >= 0:
        histogram = np.sum(combined_binary[j:i,:], axis=0)
        left_peak = np.argmax(histogram[:640])
        x_idx = np.where((((left_peak - 25) < x)&(x < (left_peak + 25))&((y > j) & (y < i))))
        x_window, y_window = x[x_idx], y[x_idx]
        if np.sum(x_window) != 0:
            left_p_x.extend(x_window.tolist())
            left_p_y.extend(y_window.tolist())

        right_peak = np.argmax(histogram[640:]) + 640
        x_idx = np.where((((right_peak - 25) < x)&(x < (right_peak + 25))&((y > j) & (y < i))))
        x_window, y_window = x[x_idx], y[x_idx]
        if np.sum(x_window) != 0:
            right_p_x.extend(x_window.tolist())
            right_p_y.extend(y_window.tolist())
        i -= 90
        j -= 90

    left_p_y = np.array(left_p_y).astype(np.float32)
    left_p_x = np.array(left_p_x).astype(np.float32)
    right_p_y = np.array(right_p_y).astype(np.float32)
    right_p_x = np.array(right_p_x).astype(np.float32)
    left_p_fit = np.polyfit(left_p_y, left_p_x, 2)
    left_fitx = left_p_fit[0]*left_p_y**2 + left_p_fit[1]*left_p_y + left_p_fit[2]
    right_p_fit = np.polyfit(right_p_y, right_p_x, 2)
    right_fitx = right_p_fit[0]*right_p_y**2 + right_p_fit[1]*right_p_y + right_p_fit[2]
    rightx_int = right_p_fit[0]*720**2 + right_p_fit[1]*720 + right_p_fit[2]
    right_p_x = np.append(right_p_x,rightx_int)
    right_p_y = np.append(right_p_y, 720)
    right_p_x = np.append(right_p_x,right_p_fit[0]*0**2 + right_p_fit[1]*0 + right_p_fit[2])
    right_p_y = np.append(right_p_y, 0)
    leftx_int = left_p_fit[0]*720**2 + left_p_fit[1]*720 + left_p_fit[2]
    left_p_x = np.append(left_p_x, leftx_int)
    left_p_y = np.append(left_p_y, 720)
    left_p_x = np.append(left_p_x,left_p_fit[0]*0**2 + left_p_fit[1]*0 + left_p_fit[2])
    left_p_y = np.append(left_p_y, 0)
    lsort = np.argsort(left_p_y)
    rsort = np.argsort(right_p_y)
    left_p_y = left_p_y[lsort]
    left_p_x = left_p_x[lsort]
    right_p_y = right_p_y[rsort]
    right_p_x = right_p_x[rsort]
    left_p_fit = np.polyfit(left_p_y, left_p_x, 2)
    left_fitx = left_p_fit[0]*left_p_y**2 + left_p_fit[1]*left_p_y + left_p_fit[2]
    right_p_fit = np.polyfit(right_p_y, right_p_x, 2)
    right_fitx = right_p_fit[0]*right_p_y**2 + right_p_fit[1]*right_p_y + right_p_fit[2]
    
    # Measure Radius of Curvature for each lane line
    ym_per_pix = 30./720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meteres per pixel in x dimension
    left_fit_cr = np.polyfit(left_p_y*ym_per_pix, left_p_x*xm_per_pix, 2)
    right_fit_cr = np.polyfit(right_p_y*ym_per_pix, right_p_x*xm_per_pix, 2)
    left_curverad = ((1 + (2*left_fit_cr[0]*np.max(left_p_y) + left_fit_cr[1])**2)**1.5) \
                                 /np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*np.max(left_p_y) + right_fit_cr[1])**2)**1.5) \
                                    /np.absolute(2*right_fit_cr[0])
    
    
    # Calculate the position of the vehicle
    center = abs(640 - ((rightx_int+leftx_int)/2))
    
    offset = 0 
    img_size = (img.shape[1], img.shape[0])
    src = np.float32([[490, 482],[810, 482],
                      [1250, 720],[40, 720]])
    dst = np.float32([[0, 0], [1280, 0], 
                     [1250, 720],[40, 720]])
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    warp_zero = np.zeros_like(combined_binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts_left = np.array([np.flipud(np.transpose(np.vstack([left_fitx, left_p_y])))])
    pts_right = np.array([np.transpose(np.vstack([right_fitx, right_p_y]))])
    pts = np.hstack((pts_left, pts_right))
    cv2.polylines(color_warp, np.int_([pts]), isClosed=False, color=(0,0,255), thickness = 40)
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (combined_binary.shape[1], combined_binary.shape[0]))
    result = cv2.addWeighted(mpimg.imread(image), 1, newwarp, 0.5, 0)
    count3 = count3+1
    plt.clf()
    plt.imshow(cv2.cvtColor((Bird_EyeView(image,count1=0, show=False)[0]), cv2.COLOR_BGR2RGB))
    plt.plot(left_fitx, left_p_y, color='blue', linewidth=3)
    plt.plot(right_fitx, right_p_y, color='blue', linewidth=3)
    plt.savefig('output_images/draw_lines/output_drawlines' + str(count3) + '.jpg')
    plt.clf()
    plt.imshow(result)
    plt.savefig('output_images/fill_lines/output_filllines' + str(count3) + '.jpg')


    return count3		
	
count3 =0			 
for image in glob.glob('test_images/test*.jpg'):
    count3 = find_fill_lanes(image, count3)			 