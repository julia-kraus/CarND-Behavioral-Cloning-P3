# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the data provided by Udacity for training the network
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

### Model Architecture and Training Strategy
#### 1. Data Collection
As I was very bad at driving the simulator, I used the training data that was supplied by Udacity in the workspace.

#### 2. Data Preprocessing
The data provided was then augmented by
- using left and right images in addition to center images by adding and subtracting a small steering correction to the steering angle 
- flipping the images
After preprocessing, the images were cropped to a size of 64x64 so that the training would be faster.

#### 1. An appropriate model architecture has been employed

I re-built the convolutional Neural Network that was developed by Nvidia and described in https://devblogs.nvidia.com/deep-learning-self-driving-cars/. It consists of 5 and fully connected layers, see the below graphic.
Instead of max-pooling, the model uses strided convolution.
The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 28). 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to train a convolutional neural network for regression on the steering angle.

My first step was to use a convolution neural network model similar to the LeNet model from the Traffic sign classifier. I thought this model might be appropriate because it worked well in the last challenge.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a high mean squared error on the training set and a high mean squared error on the validation set. This implied that the model was underfitting. 

To combat the overfitting, I implemented Nvidia's convolutional Neural network, as described in https://devblogs.nvidia.com/deep-learning-self-driving-cars/ in Keras. This time, the model performed a lot better. 

Then I increased the number of epochs from 8to 12, which further improved the training error.

The final step was to run the simulator to see how well the car was driving around track one. At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. You can see the result in the video file 

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

As I was very bad at steering the car in the simulator, I used the training data provided by Udacity in the work space.

To augment the data sat, I also flipped images and angles thinking that this would improve the training set, as the training track had mostly left turns. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Also, the training data provided center, left and right images. To further augment the data

After the collection process, I had X number of data points. I then preprocessed this data by cropping the images to the size


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The model gave good results after training 12 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
