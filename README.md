
# Action Detection 
Action Detection is ipython notebook contains all steps of training action recognition model with optical flow and single person tracking 

****Requirements****
 - [Python](https://www.python.org/) 3.*
 - [Imutils](https://pypi.org/project/imutils/)
 - [Numpy](http://www.numpy.org/)
 - [OpenCV](https://opencv.org/)
 - [Pytorch](https://pytorch.org/)
 - [Caffe2](https://caffe2.ai)
 - [Scikit-learn](https://scikit-learn.org/stable/)
 - [SciPy](https://www.scipy.org)
 - [Matplotlib](https://matplotlib.org)
   
## Dataset
*  [MERL_Shopping_Dataset](ftp://ftp.merl.com/pub/tmarks/MERL_Shopping_Dataset)

# About 
the implementation of the model was based on on [A Multi-Stream Bi-Directional Recurrent Neural Network for Fine-Grained Action Detection](http://www.merl.com/publications/docs/TR2016-080.pdf) paper

the model consist of three stages 

 1. person detection and tracking 
 2. optical flow 
 3. action detection
 
 ## Person Detection & Tracking 
for this part we peruse [pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd) repo.
we faced some problems cause of the view angle of the camera, some time model didn't notice the person as person or mistake the dimensions of him  but we solved by those steps:
 1. take highest object's score as a person, and ignoring the detection label of it
 2. make fixed box instead of dynamic one
 3. for the missing person frames, keep the previous one as the current 

## Optical Flow 
for this part we peruse [flowiz](https://github.com/georgegach/flowiz) repo.
the problem w faced at this point was the output of the repo wasn't good enough when we use two consecutive frames so that we decided to take `frame[n]` and `frame[n-6]` to calculate the optical flow in frame n


## Action Detection
for this part we peruse [Action Recognition](https://github.com/eriklindernoren/Action-Recognition) repo.
model consist of:

 - vgg16 net as features extractor (encoder)
 - lstm net with attention mechanism (decoder)
 
 trained by make prediction every 6 frames


# How to run
after download this repo and it's requirements, you have to download [pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd) repo inside action recognition directory , and put [crop.py ](https://github.com/DiaaZiada/action-recognition/blob/master/crop.py) file into pytorch-ssd repo directory and enjoy with playing by the notebook 
