#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Hist_before.png "Histogram Before"
[image2]: ./examples/Hist_after.png "Histogram After"
[image3]: ./examples/R_recovery_pred.png "R Recovery Image with Prediction"
[image4]: ./examples/L_recovery_pred.png "L Recovery Image with Prediction"
[image5]: ./examples/Center_pred.png "Center Image with Prediction"
[image6]: ./examples/mse.png "MSE Loss vs Epoch"
[image7]: ./examples/nVidia_model.png "NVIDIA Model"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I started out with a model based on the NVIDIA model:

![alt text][image7]

However, my model uses ELU activations layers to introduce nonlinearity. The data is normalized in the model using a Keras lambda layer.

More model details are provided below.

####2. Attempts to reduce overfitting in the model

I had dropout layers in the model at first but they negatively affected performance so they were removed. L2 regularization was added to help reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 132). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 218).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving and recovering from the left and right sides of the road. In addition, the left and right camera angles were used along with a "correction factor" that aided the vehicle in recovering if it finds itself on the left/right side of the road. Since the track consist mostly of left turns, I also flipped the center camera image and steering angle to even out the distribution of left/right steering wheel data and create more data to train on.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

As suggested in the project introduction, the NVIDIA model was used as a basis for my model. As for collecting data, the biggest issue I had was poor performance in turns. To combat this, I looked at the distribution of the steering wheel angles.

![alt text][image1]

As you can see the distribution is centered around 0 and +-0.25, which is from the "correction factor" for the left and right camera images when the center camera has a steering wheel angle of 0 i.e. straight driving which dominates the course. To make this more of an even distribution, any histogram bins that contained more data than half the average were randomly down sampled to bring the number of samples in that bin down to half the average. Any bins that contained less than half the average number of samples were kept the same.

![alt text][image2]


Another helpful method is to visualize the data. This was done by imposing the steering wheel angle on the image. This idea was taken from comma.ai neo's display.

####2. Final Model Architecture

My model is based on the NVIDIA architecture and was trained on 15 epochs:

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 66, 200, 3)    0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 98, 24)    1824        lambda_1[0][0]                   
____________________________________________________________________________________________________
elu_1 (ELU)                      (None, 31, 98, 24)    0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 47, 36)    21636       elu_1[0][0]                      
____________________________________________________________________________________________________
elu_2 (ELU)                      (None, 14, 47, 36)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 22, 48)     43248       elu_2[0][0]                      
____________________________________________________________________________________________________
elu_3 (ELU)                      (None, 5, 22, 48)     0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 20, 64)     27712       elu_3[0][0]                      
____________________________________________________________________________________________________
elu_4 (ELU)                      (None, 3, 20, 64)     0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 18, 64)     36928       elu_4[0][0]                      
____________________________________________________________________________________________________
elu_5 (ELU)                      (None, 1, 18, 64)     0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1152)          0           elu_5[0][0]                      
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           115300      flatten_1[0][0]                  
____________________________________________________________________________________________________
elu_6 (ELU)                      (None, 100)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        elu_6[0][0]                      
____________________________________________________________________________________________________
elu_7 (ELU)                      (None, 50)            0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         elu_7[0][0]                      
____________________________________________________________________________________________________
elu_8 (ELU)                      (None, 10)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          elu_8[0][0]                      
====================================================================================================
Total params: 252,219
Trainable params: 252,219
Non-trainable params: 0



####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image5]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct itself if it was wondering off the center of the road. These images show what a recovery looks like:

![alt text][image3]
![alt text][image4]


After the collection process, I had 20k data points. However, after down sampling the over represented steering wheel angles, I was left with 5145 data points. I was very suprised that this model could perform with such a small data set. I'm sure, as usual, more "good" data would result in better performance.

Before: (20007,) (20007,)
After: (5145,) (5145,)

I then preprocessed this data by cropping the image to remove parts of the image that aren't used to make steering decisions. The bottom of the image that only contained the car was removed and the top of the image that contained background scenary was also removed. The images were also resized to fit the input of the CNN size of 200x66. I slight Guassian blur was added to the image to generalize the pixel input and per recommendations of the NVIDIA paper the image was converted from RGB to YUV. These same preprocessing techniques were applied to the real time steering wheel angle prediction in the drive.py

NOTE: CV2.imread reads the images in BGR so a BGR2YUV transform is needed in the model.py while the drive.py reads in RGB images from the simulator so a RGB2YUV is needed...this was a cause of a lot of wasted time and headache :(

I finally randomly shuffled the data set and put 10% of the data into a validation set. 

Train: (4630,) (4630,)
Validation: (515,) (515,)


I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 15 as evidenced by the MSE loss that can be visualized with a history object from Keras.

![alt text][image6]

I used an adam optimizer so that manually training the learning rate wasn't necessary.

In conclusion, the method propsed here performs well on the test track. After my struggles with this I did not have time to encorperate data from the challenge track. My model performes very poorly on the challenge track, which is no suprise given the different appaerance of the track.
