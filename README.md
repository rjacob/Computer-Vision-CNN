# Computer-Vision-CNN
Motivation
The motivation for this assignment is to classify a dataset of images belonging to 10 distinct classes using
Convolutional Neural Networks. Convolutional Neural Networks (a.k.a ConvNet or CNN) is a powerful
machine learning model that has been increasingly gaining popularity and is being widely used to help
solve many complex problems, especially in the areas of computer vision. The ConvNet can be thought
of as a concatenation of two connected sub-networks, each made up of a series of interconnected
layers, the first half of the network, shown in green in Figure 1 below, is designed to extract features,
and the second half shown in orange is designed to classify them.
![alt text](https://github.com/rjacob/Computer-Vision-CNN/blob/master/img/Figure1.png) 
### Figure 1 - Concatenation of two subnetworks
The layers that make up the subnets are the building blocks of neural nets and are described in detail
below.
H
W
D
Figure 2 - Modified LeNet-5 (Input Layer)
To help illustrate each type of layer, the modified LeNet-5 ConvNet [1] architecture, defined in the
assignment, will be referred to throughout this section of the report. The original LeNet-5 ConvNet was
designed for MNIST, a dataset of handwritten digits. This version of the LeNet-5 ConvNet has been
modified to support the CIFAR-10 dataset.
In the image classification problem, the input into the ConvNet, is a WxHxD volume, refer to Figure 2. W
is the width, H is the height, and D is the depth or number of channels of the image. For example, for the
CIFAR-10 data set, the input color images have a volume size of 32x32x3.

Figure 3 Modified LeNet-5 (Output Layer)
The output, highlighted above in Figure 3, also considered the loss layer, is a 1-dimensional vector with a
size of C, the number of classes in the dataset. For the CIFAR-10 dataset, there are 10 distinct classes;
consequently, the output is a column vector with 10 elements. This vector contains the predictions
made by the network, the confidence score, where each element represents the probability an input
belongs to a particular class. The output is normalized using the softmax function. Besides the input and
output layers, what remain are the so called hidden layers, they are described in further detailed below.
Figure 4 Modified LeNet-5 (Convolutional Layer)
The fundamental building block of ConvNet is the convolution layer (CONV) and the intuition behind
this layer is to detect primitive shapes of an input image. The CONV layer is an order 3 tensor, where D is
the number of filters, W and H are the width and height or spatial dimensions of the activation map, and
are highlighted above in Figure 4. The activation maps are the response output of convolving or taking
the dot product of the filters scanned across the input volume spatially (WxHxD), refer to Figure 5
below. They are the energy response from the learned filter weights (Learning is discussed further
ahead) and are the neurons of the network. The hyper-parameters that define the convolution layer are;
the number of filters or depth, the filter window size, the stride, and the amount of zero padding. The
CONV layer is normally either the first (second to the input) or interlaced between subsequent layers of
the feature extraction subnetwork. The sizes of the filters, 𝐹𝑥𝐹 are normally odd centered, i.e. 3x3 or
5x5, and there are K number of filters. Therefore, there are 𝐹𝑥𝐹𝑥𝐷𝑥𝐾 number of weights and K number
of biases per CONV layer.

The stride defines how many pixels we slide the filter across the input by, for example, a stride of 1,
would slide the filter across the input 1 pixel at a time. Note the stride, controls the output volume
spatially. The larger the stride, the smaller, spatially, the output will be.
Input Volume
Filters
Activation Maps
Figure 5 - Convolution Layer
The activation function removes linearity from the network. If activation functions are not utilized in the
network, then the capacity of the network is that of a linear classifier. The activation functions have no
spatial volume. There are several activation functions available in practice, such as the Sigmoid, tanh,
ReLU, Leaky ReLU, and so forth. Because of its nice properties, the most common activation function
and the one utilized throughout this assignment is the ReLU (Rectified Linear Units). The ReLU activation
function takes a real-valued input and clamps values less than zero to zero, it does not saturate in the
positive region, it is computationally very efficient, and is known to converge much faster than sigmoid
and tanh in practice [2].
Figure 6 Modified LeNet-5 (Max Pooling Layer)
The max pooling layer (POOL) shrinks the input volume, to help reduce (1) the complexity or the
number of parameters of the network, and (2) the amount of computation of the network which helps
control overfitting [3]. The output resultant is subsampled spatially by some factor, while preserving
feature information. This is done by “pooling” features that are semantically similar. For example, for a
factor of, α=2, the result is achieved by taking the maximum of a 2𝑥2 sub window as seen in Figure 7

below. Although not commonly used in practice due to decreased performance, another method of
subsampling is taking the mean of the 2x2 windows.
Figure 7 Max Pooling example
Figure 8 Modified LeNet-5 (Fully Connected Layer)
The fully connected (FC) layers, highlighted in Figure 8 above, make up the second half of the network,
and are designed to utilize the features generated by the previous layers (first half of the network) to
help classify the input image. The second half is essentially the “classifier” of the network. Additionally,
FC layers are termed fully connected because their neurons are connected to every neuron of their
preceding layer. Fully connected layers are one dimensional vectors.
Learning is achieved through a backward propagation (BP) process where, weights assigned to each
dendrite (input to the neuron) of the perceptron are adjusted recursively to be optimal. This
optimization calculation is achieved through a method, such as the Stochastic Gradient Descent (SGD)
where the minima are realized after an appropriate number of iterations, epochs, through the entire
training data. To compute the gradients through the network, a combination of partial derivatives and
the chain-rule are used, see Figure 9 below. Yet another hyper parameter utilized during the BP, is the
learning rate which is normally selected empirically. Additionally, training is achieved through a random
mini-batch of samples, in order to limit variations between samples. In this experiment, mini-batch of 64
samples was used.
Figure 9 – BP, Gradient Descent

In Deep Learning models such as CovnNets, the network learns features over time through backward
propagation from a large training dataset, for example, of third ordered colored images. Once the
weights of the network are trained, the network has evolved into a hierarchy of extracted features that
drive the classifier. At the beginning of the network, are weights that help detect local features such as
edges and blobs, and towards the end of the network are higher level features or objects that are
targeted, for example towards the eyes and nose of a cat. The softmax classifier, applied at the end of
the classifier, uses the cross-entropy loss function where the influence of every single intermediary
value of the graph is propagated through the network.
Results & Analysis
The CIFAR-10 dataset, consisting of 60,000 32x32 color (RGB) images belonging to 10 different classes,
was used for this experiment.
First, a little about the setup of the machine used for this assignment; The CovnNet derived from LeNet-
5 network as well as the JacobNet in the next section were built, trained, and tested on a Virtual Guest
image hosted on Intel® Core™ 3.40GHz i7-4770 Central Processing Unit (CPU). Initially development was
begun on Google Cloud Computing Engine, however the limitations with the development tools and
browser shell speeds proved to be too costly, and the approach was abandoned. The Python language
and interpreter were used to develop and run the computations. Google’s Artificial Intelligence/Numeric
Computing library, TensorFlow[4] was utilized to develop the source code for this assignment.
Additionally, a Deep Learning Package built on top of TensorFlow, TfLearn [5] was used solely to
download and load images in Numpy arrays. The modified LeNet5 architecture is illustrated below in
Figure 10 in detail.
INPUT
3x32x32
CONV
6x28x28
POOL
6x14x14
CONV
16x10x10
POOL
16x5x5
FC
120
FC
84
10-Class
Softmax
ReLU ReLU ReLU ReLU
Figure 10 – Modified LeNet5 ConvNet
Here’s a program listing of the layers defined through TensorFlow for the network. The shape of each
layer validates the defined design.

Tensor("Input:0", shape=(?, 32, 32, 3), dtype=float32) Tensor("Model/Conv1/Relu:0", shape=(?, 28, 28, 6), dtype=float32) Tensor("Model/Conv1/MaxPool:0", shape=(?, 14, 14, 6), dtype=float32) Tensor("Model/Conv2/Relu:0", shape=(?, 10, 10, 16), dtype=float32) Tensor("Model/Conv2/MaxPool:0", shape=(?, 5, 5, 16), dtype=float32) Tensor("Model/FC/Relu:0", shape=(?, 120), dtype=float32) Tensor("Model/FC/Relu_1:0", shape=(?, 84), dtype=float32) Tensor("Model/add:0", shape=(?, 10), dtype=float32)
Figure 11 - Modified LeNet5 CovnNet (TensorFlow)
Preprocessing normalization and image augmentation was applied prior to training. To minimize out of range values encountered from loosely controlled data gathering, the images set were first zero-centered, where along every axis the global mean, was subtracted from test set. To normalize the data set, along every axis, the set was divided by the standard deviation, refer to Figure 12 below. This was accomplished using TensorFlow’s per_image_standardization() function.
Figure 12 - Data Preprocessing (Li, Karpathy, Johnson 2016)
The image set was also augmented, by enlarging each image by 10% and then cropping them to reduce dimensions down to the original 32x32.
Empirically, a learning rate of 0.001 was selected and defined in the regression layer of network. Initially a higher learning rate of 0.01 was attempted and through experiments, the rate was decayed over time to settle towards an optimal.
Furthermore, the Adam optimization method was specified when building the network for training. This optimization technique for the momentum helps the Gradient Descent converge to a minimum much faster, allows a velocity across the gradient to “build up” along shallow directions. The Adam gradient descent optimizer was applied to the regression layer to help minimize the provided loss function.
Finally, the test set of 50,000 images was further partitioned, where the network was trained with 40,000 images and the remaining 10,000 or 20% of this training set was utilized for validation. A mini-batch of 64 images was used, therefore for every epoch there were 40,000/64=625 training steps.

Summary of the training results is gathered using TensorBoard, and are presented in the plots below. An overall 52.52% accuracy with a cost of 0.4 was achieved with the modified LeNet-5 CovnNet. Training of 50 epochs took about 36 minutes.
As can been seen in the curves below, the network eventually converges to a very high accuracy and close to zero loss. However, as were observed with the validation and the test results, the network overfits. This is maybe due the lack of limited variations of the training data. Although all data is initially augmented, the dataset size is not expanded, i.e. with other random augmentations.
Figure 13 - Modified LeNet-5 Accuracy & Loss Curve
The figure below illustrates the confusion matrix of the predictions made by the trained modified LeNet5 network. As can be seen in the matrix, a bright dominant color is seen across the diagonal. One interesting observation that can be made, is that network has difficulty clearly distinguishing between cats and dogs.
Figure 14 - Modified LeNet-5 Confusion Matrix

Accuracy of the predictions are further demonstrated using rank-3 graphs below, where the top three resulting predictions are shown for a sample set of 4 images.
Figure 15 – Modified LeNet5 Predictions random sample (rank-3)
Note the network had no trouble classifying image #4510, but the same cannot be said for image #4625.
System Variation
To help improve results, a variant network named JacobNet was experimented with. The JacobNet differs in comparison to the modified LetNet-5 ConvNet as follows; The number of CONV layers are increased, and the kernel sizes are decreased to 3x3 for denser computation and to build thinner fibers. The number of filters in the earlier layers are increased as are the lengths of the fully connected layers. Here’s the program listing of the layers defined by TensorFlow for the JacobNet.
