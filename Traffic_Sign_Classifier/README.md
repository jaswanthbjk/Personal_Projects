The aim of the project is to build a neural network which classifies the Traffic Signs.

The Neural netowrk used is a Convolutional Neural Network which takes Images of size 64x64x3 and outputs a 42 dimensional vector which gives the information about the category the image belongs (refer to the signnames.txt for information about categories)

The image dataset is downloaded from German Traffic Sign Recognition Benchmark (GTSRB) website. The link to the dataset used is https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip

The images available are of various sizes but are segregated into sub folders which represent the class of the images.

The images were loaded and reshaped (64 X 64 X 3) using skimage library.

Neural network that's planned for the classification purpose is taken from Lecture 5 | Convolutional Neural Networks of C231n course offered by Stanford university.

![Convolutional Neural Network](convnet.jpeg) 

The Training of the images is done for 15 epochs which lead to 99.44% accuracy and the testing is done on the test dataset split form the initial dataset acquired from the GTSRB.

The Results were plotted claiming the accurate classification of the images.
