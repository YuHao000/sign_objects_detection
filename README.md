C++ Road signs detector by color and textural features
=========================

## Usage
OpenCV and alglib are used in this detector.

## Description
Detection algorithm consists of stages:
1. Auto correction handler process each image of input video stream.
2. Mode 0: user can choose object of interest for formed vector of the color features.
By clicking on the "n" button, the user switch forming between the available road signs. 
3. Mode 1: by clicking on the "t" button, the user switch the algorithm to the training mode.
In this mode, detector by the color features process each image of input video stream.
After that, user can choose object for formed vector of the textural features.
4. Mode 2: by clicking on the "l" button, the user switch the algorithm to the mode for neural network training.
5. Mode 3: work mode of detector by color and textural features.
In this mode, detector by the color features process each image of input video stream, 
and after that, neural network classifies the recognized objects.
By clicking on the "g" button, the user switch detection between detection with garbage objects and without them.