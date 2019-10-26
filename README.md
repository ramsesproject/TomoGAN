# TomoGAN

Paper: https://arxiv.org/abs/1902.07582; Relevant [presentation](https://lzhengchun.github.io/file/pse-ai-townhall-TomoGAN-Zhengchun-Liu.pdf) 

An image quality enhancement model based on generative adversarial networks originally developed for synchrotron X-Ray tomography. It has also been used for other case like streaming tomography, the joint ptycho-tomography problem. We also developed a workflow to deploy TomoGAN on Coral edge TPU which can process an image with 1k x 1k pixels in 500ms. 

I, [Zhengchun Liu](https://github.com/lzhengchun), also implemented the generator model based on pure Intel DNNL (MKL-DNN) and NVIDIA cudnn seperately for inference, using C++, so that you can easily integrate it in your C++ project. I will open source them soon.

To give it a try:

* download sample dataset and the ImageNet pretrained vgg model from [Here](https://anl.box.com/s/h6koi0hhwqrj1c9tt82tldzo45tl3x15)

* install dependencies, Tensorflow(preferably 1.13)

* run with python ./main-gan.py -gpu 0 -expName test -xtrain dataset/noisy4train.h5 -ytrain dataset/clean4train.h5 -xtest dataset/noisy4test.h5 -ytest dataset/clean4test.h5

__Note__: with the sample dataset (128 images for training) provided here, you may not reproduce our results because we used a much larger dataset that has 2048 images. 
You may download the X-ray projection from [TomoBank](https://tomobank.readthedocs.io/en/latest/source/data/docs.data.spheres.html) and make a reconstruction using all the projections as ground truth and a subsampled projections (every 32) as (low dose) noisy input. The dataset we provided, both norml dose (all projections) and low-dose (1/32 subsampled), were reconstructed using SIRT algorithms with 100 iterastions. 

As an example / teaser, a pair of noisy input and its denoised output are shown as follows,

An example __input__ (a low-dose X-ray tomographic image ):

![Noisy Image](repo-image/ns-w016-i10-r25-s0364.png)

The corresponding __output__ (i.e., denoised using TomoGAN):

![Denoisied Image](repo-image/dn-w016-i10-r25-s0364.png)
