import streamlit as st
from PIL import Image
import main
import numpy as np
from streamlit_cropper import st_cropper
st.set_option('deprecation.showfileUploaderEncoding', False)

def crop_image(image, realtime_update,box_color,aspect_ratio):
    if image:
        img = Image.open(image)
        if not realtime_update:
            st.write("Double click to save crop")
        # Get a cropped image from the frontend
        cropped_img = st_cropper(img, realtime_update=realtime_update, box_color=box_color,
                                    aspect_ratio=aspect_ratio)

        # Manipulate cropped image at will
        st.write("Preview")
        _ = cropped_img.thumbnail((150,150))
        st.image(cropped_img)
        return cropped_img

def show_predict_page():
    st.title('Cat scanner')

    st.write('### we need some cat pic to predict')
    
    box = ['Two layers', 'L layer']

    layer = st.selectbox('Layers',box)
    realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=False)
    box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
    aspect_ratio = (1, 1)
    image_file = crop_image(st.file_uploader("Upload Images", type=["png","jpg","jpeg"]),realtime_update , box_color, aspect_ratio)
    if image_file is not None:

        # To View Uploaded Image
        num_px = 64
        st.image(image_file)
        image = image_file.convert("RGB").resize([num_px,num_px],Image.ANTIALIAS)
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



