# **Finding Lane Lines on the Road** 

First will explaine Some intermidiate functions which will be used in the final pipeline
#line_seg
Defining a function to find the y coridinates if x coridinates, slope and y intercept is known
Return : This function return an array with two points of a lines in linear corodinates system

#extrapolate_lines
extrapolate_lines function will merge the output of HoughLinesP function and forms an extrpolated line
Return: An image which is of the size of the original image but only lines drawn on it and other pixels blacked
	Step1)  Call the hough function and store all the lines in the output parameter
	step2)  Extract the each line from the output of hough transforms
	Step3) finding the slope and y intercept
	Step4) collect the all the lines with positive slope and with in the provided    limitations of slope in the vertical stack using vstack
	Step5)collect the all the lines with positive slope and with in the provided limitations of slope in the vertical stack using vstack
	Step6) Take the median for the list of all the points with positive slope and negative slope
	Step7)by using the previously defined line_seg function find the four co ordinates provide m, b , x1 and x2
	Step8) Now output of line_seg contains two lines with the positive and negative slopes
	Step9)  Fill the original image will all zeros
	Step10 ) Draw the lines on this blacked image with provide line corodinates in by the function line_seg
	
**Pineline for Finding Lane Lines on the Road**

#drawing_lines_pipeline

1)  I have converted my image to grey scale.
2) Used Gaussian Noise functions to eliminate the noise 
3)Next is canny edge function is used to find the edges 
4)Next is to select the region of intreset by using the provided function 
5) Call the extrapolate_lines function.
6)select the region of intreset again to get the better image
7)weighted_img is the function which combines the houge function output and the initial image, which draws the lines on the final image.

Thus we have finally drawn the lines on the image.

#Finding the lines in the video stream
Next once the lines are extrapolated used the same for the video stream (as the video stream is a combination of multiple image frames)

For the challange the region of intrest is little bit changed and it works

### 2. Identify potential shortcomings with your current pipeline


´Technique used to extrapolaed the lines can be implemented in a better way.


----------


### 3. Suggest possible improvements to your pipeline

As of now no improvements 
