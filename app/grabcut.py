import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# Import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Import some common detectron2 utilities
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.modeling import build_model

import torch
print('Are we using GPU?')
print(torch.cuda.is_available())

import os

cfg = get_cfg()
cfg.merge_from_file("./detectron2-main/detectron2/model_zoo/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
model = build_model(cfg)

cfg.MODEL.WEIGHTS = "./models/model_final_a3ec72.pkl"

from detectron2.checkpoint import DetectionCheckpointer
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS) # Making sure that it's 'Detectron2 Model Zoo'

def grabcut(img, bbox, final_mask):

    mask = final_mask[:,:,0].copy()

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    x, y, w, h = bbox  # Which coordinate it represents with respect to the foreground -> x: left most, y: top most, w: right most, h: bottom most
    rect = (int(x), int(y), int(w)+1, int(h)+1) # ROI coordinates

    temp_mask = mask[int(y):int(y)+int(h)+1, int(x):int(x)+int(w)+1] == 1
    mask[int(y):int(y)+int(h)+1, int(x):int(x)+int(w)+1] = (temp_mask*3) # Mask region = probable foreground (represented by 3, 1 represents definite foreground)

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')

    return mask2

def get_mask(image, mask, bbox, content):

    final_mask = np.zeros_like(image)

    for c in range(final_mask.shape[-1]):
        final_mask[:,:,c] = mask.copy()

    mask2 = grabcut(image, bbox, final_mask)

    mask2 = np.repeat(mask2[:, :, np.newaxis], 3, axis=2)
    mask2 = cv2.GaussianBlur(mask2, (3, 3), 0)

    foreground = np.copy(image).astype(float)

    foreground[mask2 == 0] = 0
    foreground[mask2 == 1] = 1

    background = np.copy(image).astype(float)
    background[mask2 != 0] = 0

    merged = cv2.add(foreground, background)

    if content == '1':
      output = foreground
    elif content == '2':
      output = background
    else:
      output = merged

    return output

pred = DefaultPredictor(cfg)

# Perform reverse one-hot-encoding on labels / preds
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified
        class key.
    """
    x = np.argmax(image, axis = -1)
    return x

def get_foreground(image):
  outputs = pred(image)

  bbox = outputs['instances'].get('pred_boxes').tensor.cpu().tolist()[0]
  mask = outputs["instances"].get("pred_masks").cpu().numpy()[0]
  # mask = outputs['instance'].get('pred_masks').cpu().numpy()[0]

  content = '1'
  final_image = get_mask(image.copy(), mask.copy(), bbox, content)
  twodimage = np.zeros((final_image.shape[0], final_image.shape[1]))
  twodimage = final_image[:,:,0]
  twodimage = twodimage.astype(np.uint8)
  return twodimage