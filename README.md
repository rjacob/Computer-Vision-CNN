# Computer-Vision-CNN
Motivation
The motivation for this assignment is to classify a dataset of images belonging to 10 distinct classes using
Convolutional Neural Networks. Convolutional Neural Networks (a.k.a ConvNet or CNN) is a powerful
machine learning model that has been increasingly gaining popularity and is being widely used to help
solve many complex problems, especially in the areas of computer vision. The ConvNet can be thought
of as a concatenation of two connected sub-networks, each made up of a series of interconnected
layers, the first half of the network, shown in green in Figure 1 below, is designed to extract features,
and the second half shown in orange is designed to classify them.
![alt text <>](https://github.com/rjacob/Computer-Vision-CNN/blob/master/img/Figure1.png) 
<div style="text-align: center">
### Figure 1 - Concatenation of two subnetworks
</div>
The layers that make up the subnets are the building blocks of neural nets and are described in detail
below.
<skipped>
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
