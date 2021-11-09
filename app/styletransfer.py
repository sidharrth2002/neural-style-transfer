import os

# from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2

SEGMENT_STYLES = {}  # Global constant
IMG_DIR = './images'


style_images = os.listdir(os.path.join(IMG_DIR, 'styles'))
style_images = sorted(style_images)
print(f'The style images available by default are: {style_images}\n')

for i in style_images:
    SEGMENT_STYLES[i] = plt.imread(os.path.join(IMG_DIR, 'styles', i))

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

hub_module = load_magenta_model()

def masked_stylize(content_image, mask, segment_styles, hub_module):
    # `styles` parameter MUST BE a list of styles, if no style for the current class index, specify as `None`
    styles_to_segment = set(segment_styles.keys())
    n_sty = len(styles_to_segment)

    mask_classes = list(np.unique(mask))
    n_classes = len(mask_classes)

    if ((n_sty > n_classes) or n_sty == 0):
        raise Exception('Error: number of styles does not match the number of segmented regions in the mask or no style is passed in')

    content_image = np.array(content_image)
    print('CONTENT IMAGE')
    print(type(content_image))
    norm_ctm = content_image.copy().astype(np.float32) / 255.
    stylized_image = norm_ctm.copy()
    cur_mask = 0  # Temporary variable that stores the mask for the current class involved
    cur_layer = 0  # Temporary variable that stores the processed layer for the current class involved
    cur_style = 0  # Temp variable to store current style image for style transfer

    dim = 320
    stylized_image = cv2.resize(stylized_image, (dim, dim))  # TODO: we are resizing the content image to be 320 by 320, perhaps we should resize the segmentation mask instead

    for i, val in enumerate(mask_classes):
        # `val` indicates the value of the current class within the image mask
        if i not in styles_to_segment or segment_styles.get(i) is None:
            continue

        print(segment_styles.get(i))

        cur_layer = norm_ctm.copy()
        cur_mask = mask.copy()
        cur_mask = (cur_mask == val).astype(np.uint8)  # Getting only the current class as the active mask
        cur_style_name = segment_styles.get(i)  # TODO: ont hahrdcode here
        cur_style = plt.imread(os.path.join(IMG_DIR, 'styles', cur_style_name))
        print(cur_style)
        cur_layer = style_img_magenta(cur_layer, cur_style, hub_module)  # Get style of current layer
        cur_layer = np.squeeze(cur_layer)  # Convert EagerTensor instance to a typical image dimension
        print(stylized_image.shape)
        for j in range(stylized_image.shape[0]):
            for k in range(stylized_image.shape[1]):
                print(j, k)
                if cur_mask[j][k] == 1:
                    stylized_image[j][k] = cur_layer[j][k].copy()

    return stylized_image