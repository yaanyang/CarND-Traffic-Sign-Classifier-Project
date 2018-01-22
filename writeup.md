# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/visualization.png "Visualization"
[image2]: ./writeup_images/standarization.png "Standarization (converted to grayscale and normalization)"
[image3]: ./writeup_images/faked_image.png "Augmented Image"
[image4]: ./writeup_images/preprossing_pipeline.png "Preprossing Pipeline"
[image5]: ./writeup_images/data_augmentation.png "Data Augmentation"
[image6]: ./new_images/new_images.jpg "New Traffic Sign Images"
[image7]: ./new_images/2.jpg "Traffic Sign 2"
[image8]: ./new_images/3.jpg "Traffic Sign 3"
[image9]: ./new_images/4.jpg "Traffic Sign 4"
[image10]: ./new_images/5.jpg "Traffic Sign 5"
[image11]: ./writeup_images/new_image_bar.png "Visualization Top 5 Softmax"
[image12]: ./writeup_images/new_image_bar.png "Feature Maps"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/yaanyang/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 
It is a plot showing the distribution for all 43 classes in the training, validation and test datasets.
The plot is showing that among all 3 datasets, the distribution is roughly the same. This might imply that the occurrence for each class in the real world. So I am going to keep this relative ratio when training the model.
Also it is showing that some classes have much higher frequencies and some lower. To improve model accuracy, I will generate additional 50% of data for each class for both training and validation datasets.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it reduced the color channels from 3 to 1. This will significantly reduce computing time. Also, for the 43 classes in dataset, the color is not the decisive factor to recognize each of them. Since the shape/pattern are more important, it's good to conver them to grayscale images.

The second step is to normalize all the images because we want all our data to have small variance. Normalization can reduce brightness variance and improve contrast. This can force our model to focus on looking at features for each class instead of confused by their image quality.

I wrap these 2 operations into one function called standardize() and feed my datasets into it before sending to training. Note: I will do this after generated faked images for training and validation datasets.

Here is an example of Standarization for a Traffic Sign.

![alt text][image2]

I decided to generate additional data because some of the class has much less counts than others, this might degrade the model when trying to recognize them. I made a function called create_fake() that takes original image and do a random translation by +/- 3 pixels and random rotation by +/- 10 degrees.

I decided to not changing the image too much because some of the classes are differed only by a rotation.

Here is an example of an original image and an augmented image:

![alt text][image3]

Here is another example showing my preprossing pipeline. The normalized image would be kept as is and faked image would be added into the datasets.

![alt text][image4]

After the data augmentation, I finally have 52,198 images for training and 6,615 ones for validation. I didn't generate additional for test datasets.
Below charts showing before and after data augmentation.

![alt text][image5]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution1 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 		    		|
| Dropout   	      	| Keep Probability = 0.75       				|
| Convolution2 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 		    		|
| Dropout   	      	| Keep Probability = 0.75       				|
| Fully connected0		| Flatten. Input = 5x5x16. Output = 400			|
| Fully connected1		| Input = 400. Output = 120		             	|
| RELU					|												|
| Dropout   	      	| Keep Probability = 0.75       				|
| Fully connected2		| Input = 120. Output = 84		             	|
| RELU					|												|
| Dropout   	      	| Keep Probability = 0.75       				|
| Fully connected3		| Input = 84. Output = 43		             	|
| Dropout   	      	| Keep Probability = 0.75       				|
| Softmax				|            									|
| Cross Entropy			|												|
| Loss					|												|
| Regularization		| L2 loss with beta = 1e-5						|
| Learning Rate Decay   | 0.96 Exponential Decay, every 5 steps with 	|
|                       | starting rate = 1e-3  						|
| Optimizer     		| AdamOptimizer         						|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an 2 convelutional layers and 3 fully connected layers. This is largely referrence to LeNet structure as implemented in the class. However, I added dropout with 0.75 keep probability at each layer to prevent overfitting. Also there is a regularization applied at the loss, this is also to prevent overfitting. I choose to do 100 epochs with learning rate decay (starting rate is 1e-3), this is trying to avoid divergence and when we are close to the end, reduce the learning rate to to fine tune the model.

The batch size =128 and optimizer keep unchanged from the LeNet Model implemented in course.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 98.6%
* validation set accuracy of 93.2%
* test set accuracy of 95.0%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

I used LeNet Model which we implemented in course as my starting point. One reason to choose it is because that's my first implemented model and most familiar one :). Also because that it was used to recognize images that have letters, which I think to some extend would be similar to those patters on traffic signs. If the model is working good on geometric patterns of English letters, it might have a good chance to work on different traffic signs.

* What were some problems with the initial architecture?

The initial results actually was not too bad, I was able to achieve ~87% validation accuracy in 10 epochs with un-altered model from the course. But the accurarcy was not high enough, leaving some room for me to improve its performance.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

I decided to keep the 5-layer structure with 2 convelutional layers and 3 fully connected layers but with some modifications. 
(1) At first, since both validation and training sets were lower, the model was underfitting. Thus, I tried more epochs to try to get higher training accuracy.
(2) The validation would reach to some higher point but stop increasing, suggesting that the model was becoming overfitting. At this point, I tried several techniques to get even higher accuracy:
    (i) Adding Dropout layer: this is just to give up less important information when we tried to train the model so that we don't overfit the model to training dataset. I tried adding one dropout layer after last fully connected layer and got better results. Later, I applied dropout to every layer in the model to force it to learn the more useful features.
    (ii) Applied Regularization: another way to avoid overfitting is to applied regularization. I used TensorFlow's L2 regularization function to add the effects of all weights used in the model to the final loss.
    (iii) Learning Rate Decay: Bigger learning rate can save training time but could diverge the model, while smaller one would take more time to train. So it's a good idea to introduce learning rate decay. When we are more close to the minimum loss, decreasing the learning rate to try to reach the solution but not over it. This is particular helpful when I reached some plateau results from the validation accuracy.
(3) I didn't change the standard Relu activation function in my model.
(4) I generated additional training and validation data for my model to improve performance. 

* Which parameters were tuned? How were they adjusted and why?

There are some parameters for me to play around in my model:
(1) Epoch: I increased it from 10 to 100 for the model to become more accurate over time
(2) Batch size: I did not change batch size and keep it as 128, this is more related to the computer memory.
(3) keep_prob: the keep probability in dropout operation, I tried 0.75 and 0.5 as they are typically used. In my model, 0.75 works better as it retains just enough information without killing the learning curve.
(4) beta: this is the coefficient for L2 regulization, I tried several numbers from 1e-3 to 1e-5. It actually didn't change the results too much. For my final model I chose 1e-5.
(5) start_rate: this is the starting learning rate in my model, I used a exponential decay learning with 1e-3 starting. The decay was 0.96 over every 5 epochs. 1e-3 already works well when using fixed rate so I select it as starting rate. The decay was helpful to achieve even higher accuracy when the loss was close to minimum.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

As discussed above, I keep a 5-layer architecture with 2 convelutional layers and 3 fully connected layers. The convelutional layer increase the depth of the model but reduce the length and width. This significantly reduces number of parameters, and allowing more nodes to learn.
The dropout layers as mentioned above, can avoid the model overfiting by dumping less useful information and force the model to focus on more important ones.



### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6]

The first image might be difficult to classify because the color is a little bit faded and also the brackets around the sign might confuse the model.
The second image might have problem due to its distorted shape.
The third image might be easier to recognize but its scale is larger than other training images.
The forth image has a poorer lighting condtion which might be the difficulty.
The fifth image has a big dirt in the middle of the sign, which might be difficult to classfy.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road work      		| Road work   									| 
| Speed limit (50km/h)  | Speed limit (50km/h)							|
| Stop					| Stop											|
| Right-of-way at the   | Right-of-way at the                           |
| next intersection     | next intersection          	 				|
| Keep right			| Keep right        							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 95.0%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 27th cell of the Ipython notebook.

For the first image, the model is very sure that this is a Road work sign (probability of 0.99), and all other predictions have probabilities close to 0. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| Road work   									| 
| .000     				| Bicycles crossing								|
| .000					| Road narrows on the right						|
| .000	      			| Right-of-way at the next intersection			|
| .000				    | Children crossing    							|


For the second image, the model is relatively sure that this is a Speed limit (50km/h) sign (probability of 0.87). One interesting thing is that all top 5 predictions are speed limit sign, which gives us an idea that the model is good at differentiating speed limit signs over other kinds. This is only one out of 5 new images that does not have the correct prediction dominating others. I think the main reason for it is becuase speed limit signs are largely similar to each other. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .870         			| Speed limit (50km/h)							| 
| .117     				| Speed limit (120km/h)							|
| .005					| Speed limit (30km/h)  						|
| .003	      			| Speed limit (100km/h)             			|
| .003				    | Speed limit (70km/h) 							|


For the third image, the model is very sure that this is a Road work sign (probability of 0.99), and all other predictions have probabilities close to 0. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| Stop       									| 
| .000     				| Turn right ahead								|
| .000					| Speed limit (60km/h)  						|
| .000	      			| Roundabout mandatory               			|
| .000				    | Keep right        							|


For the forth image, the model is very sure that this is a Road work sign (probability of 1.00), and all other predictions have probabilities close to 0. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|1.000         			| Right-of-way at the next intersection			| 
| .000     				| Beware of ice/snow							|
| .000					| Children crossing      						|
| .000	      			| Pedestrians                       			|
| .000				    | General caution    							|


For the fifth image, the model is very sure that this is a Road work sign (probability of 1.00), and all other predictions have probabilities close to 0. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|1.000         			| Keep right                         			| 
| .000     				| Priority road     							|
| .000					| Yield                 						|
| .000	      			| Wild animals crossing                			|
| .000				    | Bumpy road        							|

Here is the bar chart for top 5 Softmax for each new image. As it implies, the predictions were pretty accurate.

![alt text][image11]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Here is the feature maps my model is generating for the test dataset. 

For the 1st convelutional layer, I can see the neural network is looking at the shape of the sign, perhaps the round ones. In which, there are some kinds of pattern or words. 

For the 2nd convelutional layer, since the map size was reduced to 5x5, it was harder to tell what the neural network acutally recognized. My guess is still probably the shape of the signs. But this gives me an idea how they work.

![alt text][image12]