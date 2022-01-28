README : ELEN0016 - Computer Vision - Student project 2021

Description of all files :
- best_640.pt
  This file contains all the information about our weights of our YOLOv5s model.

- create_annotations.py :
  Creates the training, validation and test set. To be able to run the code, be sure that you have a folder called train_data who
  contains two folders called images and labels. Your structure should look like this:
	train_data
	    ¦_ images
	    ¦_ labels
  To run the code : python create_annotations.py

- custom :
  This file is needed to train the YOLOv5 model. It contains information about the classes we want to detect and where the model can 
  find the images and labels.

- customtest :
  This file is needed to test the YOLOv5 model. It contains information about the classes we want to detect and where the model can 
  find the images and labels.

- object_detection.py :
  This class contains all the code that we need to detect the objects. 
  To run the code you need to be connected to the internet in order to load YOLOv5s from GitHub.
  The command line to run the code is : python object_detection.py path_to_video/sequence.mov

- yolov5_learning.ipynb :
  The code which we used in order to learn our model.
