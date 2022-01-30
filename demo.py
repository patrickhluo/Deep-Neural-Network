import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
st.set_option('deprecation.showfileUploaderEncoding', False)

# Upload an image and set some options for demo purposes
st.header("Cropper Demo")
img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg'])
realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=False)
box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
aspect_ratio = (1, 1)

def crop_image(image):
    if img_file:
        img = Image.open(img_file)
        if not realtime_update:
            st.write("Double click to save crop")
        # Get a cropped image from the frontend
        cropped_img = st_cropper(img, realtime_update=realtime_update, box_color=box_color,
                                    aspect_ratio=aspect_ratio)

        # Manipulate cropped image at will
        st.write("Preview")
        _ = cropped_img.thumbnail((64,64))
        st.image(cropped_img)
        return cropped_img