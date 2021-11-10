import os

# from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2

# Forcing tensorflow v1 compatibility
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# from tensorflow.compat.v1 import Session


SEGMENT_STYLES = {}  # Global constant
IMG_DIR = './images'


style_images = os.listdir(os.path.join(IMG_DIR, 'styles'))
style_images = sorted(style_images)
print(f'The style images available by default are: {style_images}\n')

for i in style_images:
    SEGMENT_STYLES[i] = plt.imread(os.path.join(IMG_DIR, 'styles', i))

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# sess = Session(config=config)

##### Utility functions related to loading the magenta model and stylizing image using magenta model

def load_magenta_model():
    cwd = os.getcwd()

    # Load image stylization module.
    hub_handle = os.path.join(cwd, 'models', 'magenta')
    hub_module = hub.load(hub_handle)
    return hub_module

def style_img_magenta(content_image, style_image, hub_module):
    # Convert to float32 numpy array, add batch dimension, and normalize to range [0, 1]. Example using numpy:
    ctm = content_image.copy()
    stm = style_image.copy()
    ctm = ctm.astype(np.float32)[np.newaxis, ...] / 255.
    stm = stm.astype(np.float32)[np.newaxis, ...] / 255.

    # Optionally resize the images. It is recommended that the style image is about
    # 256 pixels (this size was used when training the style transfer network).
    # The content image can be any size.
    dim = 256
    stm = tf.image.resize(stm, (dim, dim))

    # Stylize image
    outputs = hub_module(tf.constant(ctm), tf.constant(stm))
    stylized_image = outputs[0]
    return stylized_image

def masked_stylize(content_image, mask, segment_styles, hub_module, resize_dim = False):
    # `styles` parameter MUST BE a list of styles, if no style for the current class index, specify as `None`
    styles_to_segment = list(segment_styles.keys())
    n_sty = len(styles_to_segment)
    print(f'styles_to_segment: {styles_to_segment}')
    mask_classes = list(np.unique(mask))
    n_classes = len(mask_classes)
    # print(f'MASK CLASSES: {mask_classes}')
    print(f'mask_classes: {mask_classes}')
    if ((n_sty > n_classes) or n_sty == 0):
        raise Exception('Error: number of styles does not match the number of segmented regions in the mask or no style is passed in')

    content_image = np.array(content_image)
    norm_ctm = content_image.copy().astype(np.float32) / 255.
    stylized_image = content_image.copy().astype(np.float32)
    stylized_norm_image = norm_ctm.copy()
    cur_mask = 0  # Temporary variable that stores the mask for the current class involved
    cur_layer = 0  # Temporary variable that stores the processed layer for the current class involved
    cur_style = 0  # Temp variable to store current style image for style transfer

    if resize_dim:
        dim = 320
        stylized_image = cv2.resize(stylized_image, (dim, dim))  # TODO: we are resizing the content image to be 320 by 320, perhaps we should resize the segmentation mask instead
        stylized_norm_image = cv2.resize(stylized_norm_image, (dim, dim))  # TODO: we are resizing the content image to be 320 by 320, perhaps we should resize the segmentation mask instead

    # print(segment_styles.get(mask_classes[-1]))
    for i, val in enumerate(styles_to_segment):
        # `val` indicates the value of the current class within the image mask
        # print(val, segment_styles.get(val))
        if val not in styles_to_segment or segment_styles.get(val) == None:
            print('I am here')
            continue
        cur_layer = stylized_norm_image.copy()
        cur_mask = mask.copy()
        cur_mask = (cur_mask == val).astype(np.uint8)  # Getting only the current class as the active mask
        print(cur_mask)
        cur_style_name = segment_styles.get(val)  # TODO: ont hahrdcode here
        cur_style = plt.imread(os.path.join(IMG_DIR, 'styles', cur_style_name))
        cur_layer = style_img_magenta(cur_layer, cur_style, hub_module)  # Get style of current layer
        cur_layer = np.squeeze(cur_layer)  # Convert EagerTensor instance to a typical image dimension
        print(cur_layer)
        for j in range(stylized_image.shape[0]):
            for k in range(stylized_image.shape[1]):
                if cur_mask[j][k] == 1 and segment_styles.get(val) != None:
                    stylized_norm_image[j][k] = cur_layer[j][k].copy()
    return stylized_norm_image