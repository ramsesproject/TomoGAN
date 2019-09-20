# TomoGAN
An image quality enhancement model based on generative adversarial networks originally developed for synchrotron X-Ray tomography.

To give it a try:
* 
python ./main-gan.py -gpu 0 -expName test -xtrain dataset/noisy4train.h5 -ytrain dataset/clean4train.h5 -xtest dataset/noisy4test.h5 -ytest dataset/clean4test.h5
