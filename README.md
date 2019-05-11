# CNN-based image stylization
### Overview
Our project aims to construct useful image and video stylization algorithms with neural networks. We seek to make this project as extensive and usable as possible. We adapted our project using cysmith's implementation as a starting point: https://github.com/cysmith/neural-style-tf

### Goals
Goals for this project include 
* creating an algorithm that achieves a good balance between time and graphics performance, e.g. achieves a well-stylized image in a reasonable about of time
* supporting all kinds of images and video (by deconstructing it into individual frames, stylizing, then restitching)
* providing an easy way for the end user to utilize our algorithm through a command line 

### Algorithm
#### Style transfer
For style transfer, we load a pre-trained network (VGG19 from the paper [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)) due to the hardware limitations of training our own image recognition network. Our task then becomes an optimization problem, where we seek to minimize two types of loss, namely
1. _content loss_—the difference between our content image (the image we wish to stylize) and (stylized) output images. This can be computed by passing the input and output through some network and calculating the Euclidean distance between the internal layers of the  network.
2. _style loss_—the difference between our style image (the image that we used to style our content image) and (stylized) output images. This can be computed by passing the input and output through some network and comparing the Gram matrices of the outputs.

We then compute the gradients based on these losses and optimize with Tensorflow. The algorithm that we chose for image stylization is [L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS) , but Adam would also work.

#### Video style transfer
For video style transfer, we use Manuel Ruder's algorithm as outlined in [Artistic style transfer for videos](https://arxiv.org/abs/1604.08610). The main difference between this and simply applying our original style transfer algorithm to each frame of the video lies in the consistency of objects/backgrounds between frames; Ruder's algorithm introduces the idea of temporal loss when training and a focus on optical flow, meaning that each frame takes into account the previous frame when being stylized.

### Requirements
* Python 3 with all dependencies (outlined in requirements.txt; run `pip install -r requirements.txt --user`)
* Download and copy the [VGG-19 model weights file](http://www.vlfeat.org/matconvnet/pretrained/) (see the "VGG-VD models from the Very Deep Convolutional Networks for Large-Scale Visual Recognition project" section) `imagenet-vgg-verydeep-19.mat` and move it to the project directory

### Running
Once all requirements have been installed, run `bash stylize.sh <path to content image or video> <path to style image>` in `/styletransfer`. This will prompt you to specify `y` or `n` to whether the content is a video. It will then run and save the resulting stylized image or video into the respective `/image_output` or `/video_output` folder.

If running on Google Compute Engine, first activate the virtual environment by running

`source /home/ruizhao_zhu/.bashrc && source activate tf_gpu`

and then running `bash stylize.sh <path to content image or video> <path to style image>`. After it finishes, you can download the file by clicking the settings gear in the top right corner, clicking ''Download file'', and putting the full filepath (something like `/home/<username>/styletransfer/image_output/<img_name>`).
