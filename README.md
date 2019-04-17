# CNN-based image stylization
### Overview
Our project aims to construct useful image stylization algorithms with neural networks. We seek to make this project as extensive and usable as possible.

### Goals
Goals for this project include 
* creating an algorithm that achieves a good balance between time and graphics performance, e.g. achieves a well-stylized image in a reasonable about of time
* supporting all kinds of images, and perhaps video (by deconstructing it into individual frames, stylizing, then restitching)
* providing an easy way for the end user to utilize our algorithm, namely through a command line or (optionally) a simple web server with upload/download functionality

Our goals may change as the project progresses.

### Algorithm
##### Style transfer
For style transfer, we will load a pre-trained network (likely VGG16 or VGG19) due to the hardware limitations of training our own style transfer network. Our task then becomes an optimization problem, where we seek to minimize two types of loss, namely
1. _content loss_—the difference between our content image (the image we wish to stylize) and (stylized) output images. This can be computed by passing the input and output through some network and calculating the Euclidean distance between the internal layers of the  network.
2. _style loss_—the difference between our style image (the image that we used to style our content image) and (stylized) output images. This can be computed by passing the input and output through some network and comparing the Gram matrices of the outputs.

We can then compute the gradients based on these losses and optimize, likely with Tensorflow. The algorithm that seems to be of choice for image stylization is [L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS), but SGD would also work.

##### Image colorization (maybe)
For image colorization, we will use Richard Zhang's CNN-based approach, as outlined in his [2016 paper](https://arxiv.org/pdf/1603.08511.pdf). 

### Requirements
* Pre-trained VGG network (VGG16 or VGG19)
* Python 3 with all dependencies (outlined in requirements.txt; run `pip install -r requirements.txt`)