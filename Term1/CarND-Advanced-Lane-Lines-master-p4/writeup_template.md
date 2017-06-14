# Writeup Template

##Step 1: Camera Calibration
####calculate the distortion 

 - converted the image to grey scale. 
 - using the function find chessboard corner found the corner 

 **output images path** : output_images\calibrated

####Remove distortion from images
 - from the previous step we would have converted the 3d point to 2d 
 - now using the objpoints and function calibratecamera, undistort we remove the distortion.
 - 
 **output images path** : output_images\undistored
 
##Step2 :Perspective transform
 Aim of this step is to transfor the images to bird eye form so that the lanes appear to be parallel.
 Used the below function
 - getPerspectiveTransform  
 - warpPerspective
the source and destionation points are selected manually by location the lanes on the road.

  **output images path** : output_images\warped

##Step 3: Apply Binary Thresholds
 In this step i tried to apply different colour spaces to the wrapped image so that only the lane lines are visible, after many traila nd error menthods forun thís two combination to be intresting.

 - The L channel from LUV colour space with the minimum and maximun
   thersholds as 255 and 255 did an awesome job to find the while lanes
   but completely ignored the other.

 - The B channel from LAB colour space with the minimum and maximun
   thersholds as 150 and 205 did an awesome job to find the yellow lanes
   but completely ignored the other.

Combined the above two binary image to find the yellow and white lines

  **output images path1** : output_images\apply_thresholds\b_binary
  **output images path2** : output_images\apply_thresholds\l_binary
  **output images path3** : output_images\apply_thresholds\combined_binary

##Step4: Fitting a polynomial to the lane lines

 - Identifying the peaks in histograms using the with histogram
   function.
 - Find the all the non zero pixels near to histogram using the
   np.nonzero function
 - fitting a polynomial to each line to using nummpy function
   fit.polyfit

##Step5: Calculation the vehical position
 - The center of the car was taken averaging the right and left polynomials 
 - The center of the road was calculated by taking the horizonal axis divide by 2 
 - Now the distance from the center of the road to the vehical is calculated by substracting the value from step1 and step 2
 - The pixel value are converted to meter by multiplying by 3.7/700

##Step6: Radius of curvature
Below code is used to calculate the radius of curvature

        ym_per_pix = 30./720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meteres per pixel in x dimension
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    left_curverad = ((1 + (2*left_fit_cr[0]*np.max(lefty) + left_fit_cr[1])**2)**1.5) \
                                 /np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*np.max(lefty) + right_fit_cr[1])**2)**1.5) \
                                    /np.absolute(2*right_fit_cr[0])
                                   
 Source:  http://www.intmath.com/applications-differentiation/8-radius-curvature.php
 
 Average was taken for final radius 
 
  **output images path** : output_images\draw_lines
  
 Final step for preprocessing the image is to plot the line on the image and fill the gap with some colour to indicate the place the vehical is moving. 
 
 **output images path** : output_images\fill_lines

##Step7 Implement the above on the video stream
As video is a collection of image i have chosen the below method for a smoother curve.

 - First apply the line finding function on the starting frame, while
   appling store the value of the polynomials where the peaks in
   histogram was found.
 - For the next frame, first i will check near to the previous pixel
   values where the peaks in histogram was found because if we do so we
   dont need to search for the complete image.
 - If not found then should search for the complete image.

youtube link:
https://youtu.be/aMTPtIK4odE

 
 
  



                           

