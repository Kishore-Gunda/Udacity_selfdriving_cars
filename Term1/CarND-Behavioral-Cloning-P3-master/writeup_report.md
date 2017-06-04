#**Behavioral Cloning** 

This project **Behavioral Cloning** is done as a part of [selfdriving car Engineer NanoDegree]((https://www.udacity.com/drive) from udacity.

---

**Behavioral Cloning Project**

The approach follwed in this project is as follows:
* I have used the data given by udacity to train the model, i have not generated the data using the simulator.
* used keras to design the neural network
* The data provided by udacity is split into th etraining and validation set
* Using the model.h5 file it is tested in the simulator to complete a lap.


## Pre processing

**model.py** has all the helper functions for preprocesing the data and also the corresponding function to train  and validate the data. 

**Pre processing in detail **

The data from udacity has 3 images from three different camers i.e center left and right.
as show below.
Left   | Center | Right
-------|------- |-------
![left](./images/left.png) | ![center](./images/center.png) | ![right](./images/right.png)

while generating the batch randomly an image will be selected from one of the three camera angle.
During the preprocessing state only the steering angles for the left and right cameras are modified accordingly because the steering angle is provided w.r.t to center camera so some bais added to nullify the effect.

The given sample data has 24108 images i.e 8036 for each camera angle.

Pre processing pipeline: 
* *random shear* will be applied from 90% of  the images then 
* *crop* , the images will be croped  35% from top and 10% bottom then
* *flip* flips image with 0.5 probability and respective steering angles are also flipped
* then the *gamma correction* 
* then resizing the image to 64 X 64 to reduce the effort on the model.
* then the images are fed to the network.

## Network Architecture
**model.py** has the implemented model. the model is taken from [Nvidia End to End learning](https://arxiv.org/pdf/1604.07316.pdf)


Layer (type)                         |    Output Shape     |  Param #  | Connected to    
-------------------------------------|---------------------|---------- | -----------------------
lambda_1 (Lambda)                    | (None, 64, 64, 3)   |  0      |     lambda_input_1[0][0]             
convolution2d_1 (Convolution2D)      | (None, 32, 32, 24) |   1824   |     lambda_1[0][0]                   
activation_1 (Activation)            | (None, 32, 32, 24) |   0     |      convolution2d_1[0][0]            
maxpooling2d_1 (MaxPooling2D)        | (None, 31, 31, 24)  |  0      |     activation_1[0][0]               
convolution2d_2 (Convolution2D)      | (None, 16, 16, 36)  |  21636  |     maxpooling2d_1[0][0]             
activation_2 (Activation)            | (None, 16, 16, 36) |   0       |   convolution2d_2[0][0]            
maxpooling2d_2 (MaxPooling2D)  |  (None, 15, 15, 36)  |  0     |      activation_2[0][0]               
convolution2d_3 (Convolution2D)|  (None, 8, 8, 48)    |  43248   |    maxpooling2d_2[0][0]             
activation_3 (Activation)    |    (None, 8, 8, 48)   |   0      |     convolution2d_3[0][0]            
maxpooling2d_3 (MaxPooling2D)   | (None, 7, 7, 48)  |    0     |      activation_3[0][0]               
convolution2d_4 (Convolution2D) | (None, 7, 7, 64)  |    27712  |     maxpooling2d_3[0][0]             
activation_4 (Activation)    |    (None, 7, 7, 64)   |   0       |    convolution2d_4[0][0]            
maxpooling2d_4 (MaxPooling2D)  |  (None, 6, 6, 64) |     0    |       activation_4[0][0]               
convolution2d_5 (Convolution2D) | (None, 6, 6, 64) |     36928   |    maxpooling2d_4[0][0]           
activation_5 (Activation)   |     (None, 6, 6, 64)  |    0    |       convolution2d_5[0][0]            
maxpooling2d_5 (MaxPooling2D)  |  (None, 5, 5, 64)  |    0       |    activation_5[0][0]               
flatten_1 (Flatten)      |        (None, 1600)    |      0       |    maxpooling2d_5[0][0]           
dense_1 (Dense)            |      (None, 1164)     |     1863564   |  flatten_1[0][0]                 
activation_6 (Activation)     |   (None, 1164)    |      0        |   dense_1[0][0]                   
dense_2 (Dense)            |      (None, 100)        |   116500   |   activation_6[0][0]               
activation_7 (Activation)    |    (None, 100)     |      0         |  dense_2[0][0]                    
dense_3 (Dense)            |      (None, 50)      |     5050     |   activation_7[0][0]               
activation_8 (Activation)    |    (None, 50)   |         0        |   dense_3[0][0]                    
dense_4 (Dense)           |       (None, 10)      |      510     |    activation_8[0][0]               
activation_9 (Activation)      |  (None, 10)       |    0       |    dense_4[0][0]                    
dense_5 (Dense)               |   (None, 1)      |       11    |      activation_9[0][0]               
               
The model is with 2,116,983 params

## Training

For trainig the data used the following

 - *Adam optimizer*,   
 - *mean squared error* as loss metric and 
 -   *1e-4* learning rate.

The model is compiled and saves the architecture as a .json file *(model.json)*.
then trains the model over training data and save the model with weights as .hd file *(model.h5)* and weights as *weights.h5*

number_of_epochs = 8
number_of_samples_per_epoch = 20032
number_of_validation_samples = 6400

Trained the model with different epoch's but after twele epochs the accuracy starts decreasing, so fixed with eight epochs, this gives a sufficiently trained model for the given sample dataset. a kind of [early stopping]

## Results
Tried different model and also on the self generated dataset, but the results were not satisfactory. The Nvidia model with udacity class provided dataset gave the significant results.

*drive.py* script is given by udacity in class.
With the model.h5 as input this drive,py helps to run the simulator with the new camera image and predefined weights loaded from the trained model. the trottle value is fixed in the script.

while traing, the images are process as mentioned above so similarly while running the simulator in autonomous mode also the data is processed

below is the links to the final video generated using the video.py file:
https://www.youtube.com/watch?v=jOLK8DOkDHc&t=12s

## steps to run the code

**Just using the weights and running inference mode**

* `python drive.py model.h5 run1`         
this should use the pre trained weights and the model and predict the steering angle           
* then run the simulator in autonomous mode and the car should be sucessfully driven in the simulator and images hspould be saved in the folder run1

**Training and predicting**

* collect the data and modify the path accordingly in *processData.py*
* `python model.py` loads the pickled data and trains over it and should generate the model *model.json* and the model with weights *model.h5* and weights as *weight.h5*
change the epochs if required
* `python drive.py model.h5 run1`          
this should use the pre trained weights and the model and predict the steering angle             
* then run the simulator in autonomous mode and the car should be sucessfully driven in the simulator and images hspould be saved in the folder run1

