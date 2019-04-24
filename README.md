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
#### Style transfer
For style transfer, we will load a pre-trained network (tf.keras's VGG19 from the paper [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)) due to the hardware limitations of training our own image recognition network. Our task then becomes an optimization problem, where we seek to minimize two types of loss, namely
1. _content loss_—the difference between our content image (the image we wish to stylize) and (stylized) output images. This can be computed by passing the input and output through some network and calculating the Euclidean distance between the internal layers of the  network.
2. _style loss_—the difference between our style image (the image that we used to style our content image) and (stylized) output images. This can be computed by passing the input and output through some network and comparing the Gram matrices of the outputs.

We can then compute the gradients based on these losses and optimize, likely with Tensorflow. The algorithm that seems to be of choice for image stylization is [L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS), but SGD would also work.

#### Video style transfer
For video style transfer, we plan to use Manuel Ruder's algorithm as outlined in [Artistic style transfer for videos](https://arxiv.org/abs/1604.08610). The main difference between this and simply applying our original style transfer algorithm to each frame of the video lies in the consistency of objects/backgrounds between frames; Ruder's algorithm introduces the idea of temporal loss when training and a focus on optical flow, meaning that each frame takes into account the previous frame when being stylized.

#### Image colorization (maybe)
For image colorization, we will use Richard Zhang's CNN-based approach, as outlined in his [2016 paper](https://arxiv.org/pdf/1603.08511.pdf). 

### Requirements
* Python 3 with all dependencies (outlined in requirements.txt; run `pip install -r requirements.txt --user`)

### Running
Once all requirements have been installed, run `python main.py` from the `src/` folder. This will run and save the resulting stylized image as a bitmap into the `src/` folder; if running on Google Compute Engine, you can download the file by clicking the settings gear in the top right corner, clicking ''Download file'', and putting the full filepath (something like `/home/<username>/styletransfer/src/<img_name>`).