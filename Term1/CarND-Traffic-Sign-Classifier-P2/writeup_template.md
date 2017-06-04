#**Traffic Sign Recognition** 

#1. Dataset Exploration
##1.1 Dataset Summary
    The preprocessed data already has the test, train and validation set seperately, In this phase we import them into different variables.
    Later, we write a basic Summary of the Data Set Using Python, Numpy and/or Pandas
     
##1.2Exploratory Visualization
    Using the matlab interfaces ploting all the avalaible traffic sign classifiers.
    
#2.Design and Test a Model Architecture

##2.1Preprocessing
    
    ###Features processing technique:
    
    Based on this paper http://people.idsia.ch/~juergen/ijcnn2011.pdf , I have decided to preprocess the data by replacing the green channel with the CLAHE channel.and as the sign board done not have any green colour.
    
    In this preprocessing state the RGB image is converted to RCB
    
    After the image is converted to RCG, pixel values are normalized to values -1.0 to 1.0 from 0-255 to 
    
    ###Technique for Label processing
    One hot encoding is used for the label processing.


##2.2Model Architecture
    After trying many architectures I have decided for  VGG architecture.
    
    Archirtecture pipeline as below
    used the following architecture (derived from VGG):
    
    1: 1x1 convolution with relu
	2: 3x3 convolutionwith relu with 32 outputs
	3: 3x3 convolution with relu with 32 outputs
	4: 3x3 convolution with relu with 32 outputs
	5: 2x2 MaxPool (with stride 2) that reduces the image size to 16x16
	6: Dropout (0.5 during training, 1.0 during validation / testing)
	7: 3x3 convolution with relu with 64 outputs
	8: 3x3 convolution with relu with 64 outputs
	9: 3x3 convolution with relu with 64 outputs
	10: 2x2 MaxPool (with stride 2) that reduces the image size to 8x8
	11: Dropout (0.5 during training, 1.0 during validation / testing)
	12: Fully connected layer with relu with flattened inputs from step 6 and 11 with 12288 inputs and 512 outputs
	13: Dropout (0.5 during training, 1.0 during validation / testing)
	14: Fully connected layer with relu with 512 inputs and 43 (class) outputs
	
##2.3Model Training
    Adam optimizer was chosen after trying with various optimizers, as this has less parameter to tune.
    
    Batch size: 120 - After a lot of trail and error 120 seems to be a good batch size for the network.
    
    Numer of Epochs: Trained for 60 epochs with model saved at every 10 epochs, // update the results

##2.4Solution Approach
     Initally started with the lenet architecture and after some study and some research found that VVG (modified) version can help us in get some better results.
    
    As I want the validation set to be different for each epohs, so i have merged the train set and validation set and then spliting them again for epoch 
    
    After the training and then validating I have tested with the test data and reached an accuracy of  **0.954** 
    
#3.Test a Model on New Images

##3.1Acquiring New Images
    I have found some images from the german traffic signs database and selected five from them.
    
    Then Loaded 5 images from img folder with filenames test0.bmp - test5.bmp and stored them into an array.
    
    Used the above preprossing technique to preprocess the data i.e converting the RGB into RCB channels.
    
##3.2Performance on New Images
    The model has predicted 4 out of 5 signs correctly, it's 80% accurate on these new images.
    
    one image was not pridicted because its truncated i.e the image is not clearly seen.
    
##3.3Model Certainty - Softmax Probabilities
    tf.nn.top_k will return the values and indices (class ids) of the top k predictions. So if k=5, for each sign, it'll return the 5 largest probabilities (out of a possible 43) and the correspoding class ids.
    
    The tf.nn.top_k is used on the softmax output to get probability top 5 probabilities.
    
    The model has predicted 4 out of 5 signs correctly, it's 80% accurate on these new images.
    
    So incase of last image as the image is not clearly seen the model could not predict it properlly. 
