# **Finding Lane Lines on the Road** 

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

**Pineline for Finding Lane Lines on the Road**
Below are the steps used to build my Pipeline  (Name of pipeline: findingline_pipeline):

1)  I have converted my image to grey scale.
2) Used Gaussian Noise functions to eliminate the noise with kernel value 7
3)Next is canny edge function is used to find the edges and here is the lower and higher values defined for canny edge function 50 , 150 respectively.
4)Next is to select the region of intreset by using the provided function region_of_intrest with the parameters [[(440,300),(500, 300),(880, 539),(80,539)]]
5)used the  hough_lines functions to draw the lines on the image with several trail and error finally concluded for the below values rho=2, theta=np.pi/180, threshold=3, min_line_len=20, max_line_len=10
6)weighted_img is the function which combines the houge function output and the initial image, which draws the lines on the final image.

Next step is to extrapolate the lines drawn by the houge function.
For this I have used method of average the gradients and intercepts of the postive and negative hough lines

Next once the lines are extrapolated used the same for the video stream (as the video stream is a combination of multiple image frames)

### 2. Identify potential shortcomings with your current pipeline


´Technique used t extrapolaed the lines can be implemented in a better way.


----------


### 3. Suggest possible improvements to your pipeline

As of now no improvements 
