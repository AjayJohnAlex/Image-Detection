import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'svg'
# creating a dict specifying the model ,using gpu ,weights etc

options = {
    'model':'cfg/yolo.cfg',
    'load':'bin/yolov2.weights',
    'threshold':0.3, #confidence factor more than 0.3 
    'gpu':1.0
}
# object to train on the diff weights and models
tfnet = TFNet(options)
# testing it on an image
# the image is of a guy sitting on a horse
img = cv2.imread('2.jpg')
# predicting the image
result = tfnet.return_predict(img)
result