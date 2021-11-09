import streamlit as st
from segment import segment_image
from styletransfer import load_magenta_model, style_img_magenta, masked_stylize
# Importing the functions to load magenta model, stylize each image in magenta, and to perform masked stylization
from PIL import Image
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tensorflow.compat.v1.keras import backend as K  # Used for managing model session

# Forcing tensorflow v1 compatibility
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import Session

config = ConfigProto()
config.gpu_options.allow_growth = True
sess = Session(config=config)

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
class_dict = pd.read_csv(os.path.join(DATA_DIR, 'labels_class_dict.csv'))
# Get class names
class_names = class_dict['class_names'].tolist()
# Get class RGB values
class_rgb_values = class_dict[['r','g','b']].values.tolist()

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

uploaded_file = st.file_uploader("Choose an image...")

col1, col2 = st.columns(2)
segment_styles = {}  # Dict that stores the class label encoded value to style image mappings
submit_button = 0  # Declaring the submit button outside of the form below in order for it to be accessed later on in the web app
content_image = 0  # Stores the content image in an array format

# Load model based on current session
hub_module, session = load_model()  # Load magenta model and session
if uploaded_file:
    K.set_session(session)

if uploaded_file is not None:
    with col1:
        st.image(uploaded_file, use_column_width=True)
    with col2:
        with st.spinner("Processing..."):
            content_image = Image.open(uploaded_file)
            content_image = np.array(content_image)
            le_mask, rgb_mask, foreground_heatmap, objects = segment_image(content_image)
            st.image(rgb_mask, use_column_width=True)
            st.image(foreground_heatmap, use_column_width=True)

        # print(objects)
        # only get objects that appear in at least 1 pixel
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
        # print(segment_styles)

    if submit_button:
        st.write('Output')
        with st.spinner("Processing..."):

            # Stylize image based on segments and display output to user
            # content_image = Image.open(uploaded_file)
            # content_image = np.array(content_image)
            segmented_img = masked_stylize(content_image, le_mask, segment_styles, hub_module)
            st.image(segmented_img)