# Deep-Neural-Network

## Requirements:

Numpy -- fundamental package for scientific computing with Python  
matplotlib -- library to plot graphs with Python  
h5py -- package to interact with a dataset that is stored on an H5 file  
PIL -- Python Imaging Library to read image  

## How to use
Open command line at the folder, put your favourite cat picture in the folder, enter the command below  
```bash
python main.py cat.jpg T
```  
replace 'cat.jpg' with your cat pic name  

Note that since the training data set uses pictures in square, so the program will resize your cat picture to square, so if your picture is a rectangle, it might squeeze the picture so the result will be inaccurate. Recommand to resize the picture manually to get the best result.

The last letter 'T' indicates two layer model, 'L' indicates L layer model

You can try running 2 different model to see the differece in learning curves!
