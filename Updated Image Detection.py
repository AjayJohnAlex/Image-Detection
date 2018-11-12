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
# testing on another image (of an elephant but image named cat)
img = cv2.imread('cat.jpg',cv2.IMREAD_COLOR)
# redifine the image 
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
result = tfnet.return_predict(img)
result
# making a box around the object and giving it a label
tl = (result[0]['topleft']['x'],result[0]['topleft']['y'])
br = (result[0]['bottomright']['x'],result[0]['bottomright']['y'])
label =  result[0]['label']
# plotting the box
img = cv2.rectangle(img,tl,br,(0,255,0),7)
# adding the label
img = cv2.putText(img,label,tl,cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)

plt.imshow(img)