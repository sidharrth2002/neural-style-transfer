import streamlit as st
from segment import segment_image
from grabcut import get_foreground
from styletransfer import load_magenta_model, style_img_magenta, masked_stylize
# Importing the functions to load magenta model, stylize each image in magenta, and to perform masked stylization
from PIL import Image
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
from download_image import get_image_download_link
from tensorflow.compat.v1.keras import backend as K  # Used for managing model session


# If you are using GPU, you should uncomment tensorflow v1 compatibility
# However, there may be errors that arise due to version issues with tensorflow and CuDNN
# We developed this application fully on CPU, without using GPU, due to compatibility issues among our computers

# Forcing tensorflow v1 compatibility
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# from tensorflow.compat.v1 import Session
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# sess = Session(config=config)

# Function to load magenta model
@st.cache(allow_output_mutation=True)
def load_model():
    model = load_magenta_model()
    # model._make_predict_function()
    # model.summary()  # included to make it visible when model is reloaded
    session = K.get_session()
    return model, session

st.set_option('deprecation.showfileUploaderEncoding', False)

DATA_DIR = './'
IMG_DIR = './images'
class_dict = pd.read_csv(os.path.join(DATA_DIR, 'labels_class_dict.csv'))
class_names = class_dict['class_names'].tolist()  # Get class names
class_rgb_values = class_dict[['r','g','b']].values.tolist()  # Get class RGB values
select_classes = ['sky', 'tree', 'road', 'grass', 'water', 'building', 'mountain', 'foreground', 'unknown']

# Get RGB values of required classes
select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
select_class_rgb_values =  np.array(class_rgb_values)[select_class_indices]
label_style_mapping = dict(zip(select_classes, select_class_indices))

st.title("Semantically Segmented Neural Style Transfer")
st.markdown('''
    ### Good artists copy; great artists steal. -*Vincent Van Gogh*
''')

st.markdown('#### The Greatest Art The World Has Every Seen!')
STYLE_IMG_NAMES = ['None', 'Kandinsky', 'Seated Nude', 'Shipwreck', 'Starry Night', 'The Scream', 'Woman With Hat Matisse']
style_images = [f'./images/styles/{img}.jpg' for img in STYLE_IMG_NAMES if img != 'None']
st.image(style_images, width=200, caption=[img for img in STYLE_IMG_NAMES if img != 'None'])

st.markdown('''
Upload an image -> Wait for the segmentation model -> Choose a style for each segment -> *Voila!*
''')

uploaded_file = st.file_uploader("Choose an image to perform style transfer...")

segment_styles = {}  # Dict that stores the class label encoded value to style image mappings
submit_button = 0  # Declaring the submit button outside of the form below in order for it to be accessed later on in the web app
content_image = 0  # Stores the content image in an array format

# Load model based on current session
hub_module, session = load_model()  # Load magenta model and session
K.set_session(session)

if uploaded_file:
    content_image = Image.open(uploaded_file)
    content_image = np.array(content_image)

# Declare radio buttons for getting user feature choice
genres = ['Single style transfer', 'Custom foreground and background mask', 'Foreground extraction using the GrabCut algorithm', 'Semantic style transfer using UNET']
genre_radio = st.radio('Style transfer genre', genres)

if genre_radio == genres[0]:
    # Single style transfer
    col1, col2 = st.columns(2)
    if uploaded_file is not None:
        content_image = cv2.resize(content_image, (320, 320))
        with col1:
            st.image(uploaded_file, use_column_width=True)
        with col2:
            # Declare a form and call methods directly on the returned object
            single_form = st.form(key='single_style_form')
            single_form_response = single_form.selectbox('Style to transfer', STYLE_IMG_NAMES)

            # Single style transfer submit_button
            single_submit_button = single_form.form_submit_button(label='Single Stylize')

        if single_submit_button:
            st.write('Output')
            with st.spinner("Processing..."):
                # Stylize image based on segments and display output to user
                single_stylized = 0  # Temp variable for storing single stylized image
                if single_form_response != 'None':
                    single_style_image = plt.imread(os.path.join(IMG_DIR, 'styles', f'{single_form_response}.jpg'))
                    single_stylized = style_img_magenta(content_image, single_style_image, hub_module)
                    single_stylized = np.squeeze(single_stylized)  # Convert EagerTensor instance to a typical image dimension
                else:
                    single_stylized = content_image.copy()
                st.image(single_stylized, clamp=True)
                st.markdown(get_image_download_link(single_stylized), unsafe_allow_html=True)

elif genre_radio == genres[1]:
    # Upload custom mask
    col1, col2 = st.columns(2)
    if uploaded_file is not None:
        with col1:
            st.image(uploaded_file, use_column_width=True)
        with col2:
            custom_mask = st.file_uploader("Choose an input image mask...")

            if custom_mask:
                # If the user did upload a custom mask to be used
                # Display custom mask, for each class, ask for the style of transfer
                st.image(custom_mask, use_column_width=True)

                custom_mask = Image.open(custom_mask)
                custom_mask = np.array(custom_mask)

                # Assign 2 as foreground, 1 as background
                custom_mask[custom_mask == 255] = 2
                custom_mask[custom_mask == 0] = 1
                layer_labels = {2: 'Foreground', 1: 'Background'}
                layer_styles = []

                layered_form = st.form(key='custom_mask_form')
                lkeys = layer_labels.keys()
                for lk in lkeys:
                    lf = layered_form.selectbox(layer_labels[lk], STYLE_IMG_NAMES)
                    layer_styles.append(lf)
                layered_button = layered_form.form_submit_button(label='Stylize Segments')

                segment_styles = {}  # Reset the segment styles dictionary
                for i, f in enumerate(lkeys):
                    if layer_styles[i] != 'None':
                        segment_styles[f] = layer_styles[i] + '.jpg'
                    else:
                        segment_styles[f] = None

                if layered_button:
                    st.write('Output')
                    with st.spinner("Processing..."):
                        # Stylize image based on segments and display output to user
                        layered_segmented_img = masked_stylize(content_image, custom_mask, segment_styles, hub_module)
                        st.image(layered_segmented_img, clamp=True)
                        st.markdown(get_image_download_link(layered_segmented_img), unsafe_allow_html=True)

elif genre_radio == genres[2]:
    # GrabCut for FGBG extraction, then style transfer
    st1, st2 = st.columns(2)
    if uploaded_file is not None:
        gc_form = st.form(key='single_style_form_foreground')
        st1.image(content_image, use_column_width=True)
        foreground = get_foreground(content_image)

        fg_display = foreground.copy()
        fg_display[fg_display == 1] = 255
        st2.image(fg_display, clamp=True, channels='Grayscale')
        plt.imshow(fg_display, cmap='gray')

        # Assign 2 as foreground, 1 as background
        gc_layer_labels = {2: 'Foreground', 1: 'Background'}
        gc_layer_styles = []
        gc_segment_styles = {}  # Reset the segment styles dictionary

        gc_layered_form = st.form(key='grabcut_mask_form')
        gc_lkeys = gc_layer_labels.keys()
        for lk in gc_lkeys:
            gc_lf = gc_layered_form.selectbox(gc_layer_labels[lk], STYLE_IMG_NAMES)
            gc_layer_styles.append(gc_lf)
        gc_layered_button = gc_layered_form.form_submit_button(label='Stylize Grabcut Segments')

        if gc_layered_button:
            st.write('Generated output')
            with st.spinner("Processing..."):
                # Stylize image based on segments and display output to user
                single_stylized = 0  # Temp variable for storing single stylized image

                num_styles_used = 0
                for i in gc_layer_styles:
                    if i != 'None':
                        num_styles_used += 1

                if num_styles_used > 0:
                    foreground[foreground == 1] = 2
                    foreground[foreground == 0] = 1

                    for i, f in enumerate(gc_lkeys):
                        if gc_layer_styles[i] != 'None':
                            gc_segment_styles[f] = gc_layer_styles[i] + '.jpg'
                        else:
                            gc_segment_styles[f] = None
                    segmented_img = masked_stylize(content_image, foreground, gc_segment_styles, hub_module, True)
                else:
                    segmented_img = content_image.copy()
                st.image(segmented_img, clamp=True)
                st.markdown(get_image_download_link(segmented_img), unsafe_allow_html=True)

elif genre_radio == genres[3]:
    # UNET semantic segmentation, then style transfer
    col1, col2 = st.columns(2)
    if uploaded_file is not None:
        with col1:
            st.image(uploaded_file, use_column_width=True)
        with col2:
            with st.spinner("Processing..."):
                le_mask, rgb_mask, foreground_heatmap, objects = segment_image(content_image)
                st.image(rgb_mask, use_column_width=True)
                st.image(foreground_heatmap, use_column_width=True)

            filtered_objects = []
            for i in objects:
                if(objects[i] > 0):
                    filtered_objects.append(i)

            # Declare a form and call methods directly on the returned object
            form = st.form(key='my_form')
            styles = []
            for idx, obj in enumerate(filtered_objects):
                sf = form.selectbox(obj.title(), STYLE_IMG_NAMES)
                styles.append(sf)

            # global submit_button
            submit_button = form.form_submit_button(label='Stylize Segments')

            for i, f in enumerate(filtered_objects):
                cur_class = label_style_mapping.get(f.lower())
                cur_class = int(cur_class)
                if styles[i] != 'None':
                    segment_styles[cur_class] = styles[i] + '.jpg'
                else:
                    segment_styles[cur_class] = None
            print(segment_styles)

        if submit_button:
            st.write('Output')
            with st.spinner("Processing..."):
                segmented_img = masked_stylize(content_image, le_mask, segment_styles, hub_module, True)
                st.image(segmented_img, clamp=True)
                st.markdown(get_image_download_link(segmented_img), unsafe_allow_html=True)