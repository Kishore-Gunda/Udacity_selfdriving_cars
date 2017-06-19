#Udacity Self Driving Car Engineer Nanodegree - Project 5
The main aim of this project is to find the vehical in the video stream. 

To complete this project the below steps I have followed

(Note: The ouput images for steps can be found in the Project5_vehical_detection.ipynb file, no images attached in this file)

 - **Step1:** Define a function to draw boxes

####Preprocessing

 - **Step 2:** Convert the provided input data to YUV and colour space and apply the HOG transform
 - **Step3:**  Prepare the normalised input data and lables

####Training and Testing 

 - **Step 4:** Select the best classifier and train and test the data

####Finding vehicals

 - **Step 5:** Define the sliding window and draw rectangles
 - **Step 6:** Define the class for video pipelines and draw boxes for found lines

###**Step1: Define a function to draw boxes**
Defines a fucntion called draw_boxes which takes the image , box boundiries, color and thickness and input and return and image with boxes drawn on it, uses open cv function cv2.rectangle to do this.

###**Step2:Convert the provided input data to YUV and colour space and apply the HOG transform**

In this step I defined two functions

 - features_extration_multiple_images -->for array of images 
 -  features_extration_single_image --> for single image

the above functions are used to resize the image to specific mentioned size using cv2.resize, convert the images to specified colour spaces and finally apply the HOG transforms using hog.compute.

Performance is tested with multiple color spaces  (RGB, HSV, LUV, YUV) and finally found that the performance is better with the YUV colour space.

###**Step3: Prepare the normalised input data and lables**

 - Convert the features to zero mean and unit variance,
 - Prepared the labels for the provied input data vehical as "1" and non
   vehicals as "0"
   

### **Step 4: Select the best classifier and train and test the data**
The next step is to select the classifier. The below three classifier are choosen to select one among them.

 - Linear Support Vector Machine 
 - Logistic Regression Classifier
 - Multi-layer Perceptron

These were the results of training and testing on each one:

Classifier	                   Training Accuracy	         Test Accuracy	                 Prediction Time
LinearSVC	                       1.00	                         .982	                            .0016sec
Logistic Regression	            1.00	                         .987	                            .0002
Multi-layer Perceptron	    1.00	                         .992	                            .0008

from the above results i have selected the MLP classifier.

### **Step 5: Define the sliding window**

 - In sliding window approach  I have choosen one slice of image at
   once and applied the HOG transforms on that particular window.
 - Two minimize the search time and speed up pipeline  , i have
   eliminated the upper part of the image.
 - The sliding varies in various rages with a overlap of 80%  is used,
   to not miss the far and near vehicals  which varies in size.
 - For better results, i have used the function predict_proba which
   results probability of each class i.e car or non cars.
 - The thershold i have selected is 99% to avoid the false predictions
   for each window.
 - Once the search for vehicals in the windows of all the sizes are done , then the
   boundary boxes are drawn to black image of same size(masked image).
 - Next I use the OpenCV function cv2.findContours to find all of the
   objects in the masked image and once the contours are found
 - Next I used the OpenCV function cv2.boundingRect on each contour to
   get the coordinates of the bounding rect for each vehicle.
 - Finally merged the orinal image and the annotated image which finds
   the vehicals.

###**Step 6: Define the class for video pipelines and draw boxes for found lines**

 - Finally to find the vehicals in the input video stream, a new class
   is created to store all the rectangle boundaries in past 12 frames in a list.
   
 - Then in each frame combine the rectangle from the present frame to
   previous frame   and group it using cv2.groupRectangles function
   which combines the over lapping rectangles in to a single rectangle
   
 - Here an addtional advantge is the threshold setting which helps to
   eleminate the false finding. In detail if parameter set to 10 means
   it will look for the ten overlaps and ignore the rest. So here the
   vehical should be detected in 10 frames out of 12. Thus false
   positive will be eleminated.

##**Discussion**

 - The most challanging part is to eleminate the false positives,
   because the false positive may lead to the erronious results
 - My trainend classifier still finds some false positive, this can be
   improved with some more trainning data with vehical and non vehical
   data.
 - Now my classifier use 4 frames per second , number of frames per
   second is quite lower this should be even more in the real time for
   better preformance.
 - The sliding window still does some unnecessary stuff, which should be
   reduced to improve the performance.

