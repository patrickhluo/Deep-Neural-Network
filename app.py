import streamlit as st
from PIL import Image
import main
import numpy as np


def show_predict_page():
    st.title('Cat scanner')

    st.write('### we need some cat pic to predict')
    
    box = ['Two layers', 'L layer']

    layer = st.selectbox('Layers',box)
    
    image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
    if image_file is not None:

		# To See details
        file_details = {"filename":image_file.name, "filetype":image_file.type,
                        "filesize":image_file.size}
        
        st.write(file_details)

        # To View Uploaded Image
        num_px = 64
        st.image(load_image(image_file))
        image = load_image(image_file).convert("RGB").resize([num_px,num_px],Image.ANTIALIAS)
        my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)

        image = np.array(image)
        my_image = image.reshape(num_px*num_px*3,1)
        my_image = my_image/255.
        
        if layer == 'Two layers':
            parameters = np.load('Tparameters.npy', allow_pickle=True).item()
        else:
            parameters = np.load('Lparameters.npy', allow_pickle=True).item()

        my_predicted_image = main.predict(my_image, my_label_y, parameters)
        result = ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + main.classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")


        st.write(result)


def load_image(image_file):
	img = Image.open(image_file)
	return img

if __name__ == '__main__':
    show_predict_page()



