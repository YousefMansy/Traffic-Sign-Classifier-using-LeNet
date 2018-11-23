
# Self-Driving Car Engineer Nanodegree

## Deep Learning

## Project: Build a Traffic Sign Recognition Classifier



[//]: # (Image References)

[image1]: ./writeup_images/random.png "random"
[image2]: ./writeup_images/hist.png "hist"
[image3]: ./writeup_images/internet.png "internet"

---
## Step 0: Load The Data

---

## Step 1: Dataset Summary & Exploration


### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas

- Number of training examples = 34799
- Number of testing examples = 12630
- Image data shape = (34799, 32, 32, 3)
- Number of classes = 43

![alt text][image1]

### Include an exploratory visualization of the dataset

![alt text][image2]

----

## Step 2: Design and Test a Model Architecture


### Pre-process the Data Set (normalization, grayscale, etc.)

#### The image data is converted into grayscale to help minimize training time.

#### Then the data is shuffled to randomize order of samples

#### Then the image data is normalized so that the data has mean zero and equal variance.

### Model Architecture

### This the LeNet Architecture, with the following layers:

#### - 5x5 Convolution Layer (Input = 32x32x1, Output = 28x28x6)
#### - ReLU Activation
#### - 2x2 Max Pool (Input = 28x28x6, Output = 14x14x6)
#### - 5x5 Convolution (Input = 14x14x6, Output = 10x10x16)
#### - ReLU Activation
#### - 2x2 Max Pool (Input = 10x10x16, Output = 5x5x16)
#### - Flatten Layer (Input = 5x5x16, Output = 400)
#### - Fully connected Layer (Input = 400, Output = 1024)
#### - Dropout Layer
#### - ReLU Activation
#### - Fully connected Layer (Input = 1024, Output = 1024)
#### - Dropout Layer
#### - ReLU Activation
#### - Fully connected Layer (Input = 1024, Output = 43)

### Train, Validate and Test the Model

A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

### For training, Adam optimizer was used.

### Hyperparametrs:
#### -batch size: 64
#### -epochs: 38
#### -learning rate: 0.0008
#### -mu: 0
#### -sigma: 0.1

# Validation set accuracy: 94.3%

# Test set accuracy: 93%

My approach was mostly trial and error. However, I had worked on a similar project using LeNet before prior to the ND, so I started from the parameters that I used then, However I was using 50 epochs initially, then I noticed that the accuracy doesn't get any better after the 38th epoch, so I stopped at that.

---

## Step 3: Test a Model on New Images

To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.

You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

I have download 6 random german traffic signs from the internet and labelled them

### Load and Output the Images

![alt text][image3]

### Predict the Sign Type for Each Image

- Actual Label:
[17, 4, 36, 25, 29, 38]
- Output:
[17  4 36 25 10 38]

### Analyze Performance

### Accuracy: 83.3 %

The model succeeded at predicting 5 out of the 6 images, the only one that failed was the bicycle one, which I find a bit odd since it's taken at a pretty straight angle and doesn't have a lot of noise, but I suppose maybe there's a chance this is not the standard sign for the german signs which the model has been trained on. The accuracy here does seem to be lower than the accuracy with the test data, but then again this is a very small sample (6 images) so it's expected to get an unreliable accuracy estimation for the model's performance.

### Output Top 5 Softmax Probabilities For Each Image Found on the Web

Top softmax probabilities:

Prediction: 17
1. Label: 17 - Probability: 100.0%
2. Label: 40 - Probability: 0.0%
3. Label: 33 - Probability: 0.0%
4. Label: 0 - Probability: 0.0%
5. Label: 1 - Probability: 0.0%

Prediction: 4
1. Label: 4 - Probability: 100.0%
2. Label: 0 - Probability: 0.0%
3. Label: 1 - Probability: 0.0%
4. Label: 6 - Probability: 0.0%
5. Label: 18 - Probability: 0.0%

Prediction: 36
1. Label: 36 - Probability: 100.0%
2. Label: 35 - Probability: 0.0%
3. Label: 13 - Probability: 0.0%
4. Label: 34 - Probability: 0.0%
5. Label: 12 - Probability: 0.0%

Prediction: 25
1. Label: 25 - Probability: 100.0%
2. Label: 30 - Probability: 0.0%
3. Label: 21 - Probability: 0.0%
4. Label: 24 - Probability: 0.0%
5. Label: 11 - Probability: 0.0%

Prediction: 29
1. Label: 10 - Probability: 99.4%
2. Label: 35 - Probability: 0.6%
3. Label: 9 - Probability: 0.0%
4. Label: 42 - Probability: 0.0%
5. Label: 23 - Probability: 0.0%

Prediction: 38
1. Label: 38 - Probability: 100.0%
2. Label: 0 - Probability: 0.0%
3. Label: 1 - Probability: 0.0%
4. Label: 2 - Probability: 0.0%
5. Label: 3 - Probability: 0.0%

The model seems to be pretty confident about most of the predictions, except for the one that failed, the bicycles one, where it didn't come near the correct guess at all, which bewilders me.
