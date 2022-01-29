import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from dnn_app_utils_v3 import *
import sys


#process argument
image = "test.png"
mode = 'A'
try:
    image = str(sys.argv[1])
    if len(sys.argv ) == 3:
        mode = str(sys.argv[2])
except IndexError as e:
    print("You did not enter picture name, program will use test.png as default")


plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def plot_costs(costs, learning_rate=0.0075):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

np.random.seed(1)

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# Explore your dataset 
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))

# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255
test_x = test_x_flatten/255

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))

def calculate(image):
    mode = 'T'
    if mode == 'L':
        ### CONSTANTS ###
        layers_dims = [12288, 20, 7, 5, 1] #  4-layer model 

        parameters, costs = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)

        plot_costs(costs)
    elif mode == 'T': 
        ### CONSTANTS DEFINING THE MODEL ####
        n_x = 12288     # num_px * num_px * 3
        n_h = 7
        n_y = 1
        layers_dims = (n_x, n_h, n_y)
        
        parameters, costs = two_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)

        plot_costs(costs)
    else:
        print("You did not enter L or T as argument, it will run L layer by default")

        ### CONSTANTS ###
        layers_dims = [12288, 20, 7, 5, 1] #  4-layer model 

        parameters, costs = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)

        plot_costs(costs)

    pred_train = predict(train_x, train_y, parameters)

    pred_test = predict(test_x, test_y, parameters)

    fileImage = image.convert("RGB").resize([num_px,num_px],Image.ANTIALIAS)
    my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)

    image = np.array(fileImage)
    my_image = image.reshape(num_px*num_px*3,1)
    my_image = my_image/255.
    my_predicted_image = predict(my_image, my_label_y, parameters)

    plt.imshow(image)
    return  ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")

if __name__ == '__main__':
    image = Image.open(image)
    print(calculate(image))
    