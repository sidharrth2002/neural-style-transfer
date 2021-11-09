from tensorflow.python.ops.gen_array_ops import reverse
import torch
import cv2
import albumentations as album
import numpy as np
import pandas as pd
import os
import segmentation_models_pytorch as smp
from collections import Counter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Force pytorch to use CPU
DEVICE = torch.device("cpu")
ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'

best_model = torch.load('./models/unet/trained-model.pth', map_location=DEVICE)
print('Loaded UNet model.')

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

class BackgroundDataset(torch.utils.data.Dataset):
    """Stanford Background Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        df (str): DataFrame containing images / labels paths
        class_rgb_values (list): RGB values of select classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """
    def __init__(
            self,
            df,
            class_rgb_values=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.image_paths = df['image_path'].tolist()
        self.mask_paths = df['label_colored_path'].tolist()

        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read images and masks
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)

        # one-hot-encode the mask
        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        # return length of
        return len(self.image_paths)

def get_validation_augmentation():
    # Add sufficient padding to ensure image is divisible by 32
    test_transform = [
        album.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
    ]
    return album.Compose(test_transform)

def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callable): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))

    return album.Compose(_transform)

# helper function for data visualization
def visualize(**images):
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize=(20,8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([]);
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_',' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()

# Perform one hot encoding on label
def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label
        label_values

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map

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

# Perform colour coding on the reverse-one-hot outputs
def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x

test_image = cv2.imread('9003250.jpg')

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

# Center crop padded image / mask to original image dims
def crop_image(image, true_dimensions):
    return album.CenterCrop(p=1, height=true_dimensions[0], width=true_dimensions[1])(image=image)

'''
Input: Image
Output: Mask, Foreground Heatmap and Dictionary of Objects in Image
'''
def segment_image(image):
    # print(type(image))
    true_dimensions = image.shape
    image = preprocessing_fn(image)
    image = cv2.resize(image, (320, 320))
    x_tensor = torch.from_numpy(image).double().to(DEVICE).unsqueeze(0)
    x_tensor = x_tensor.permute(0, 3, 1, 2)
    x_tensor = x_tensor.double()
    # change to double
    best_model.double()
    # Predict test image
    pred_mask = best_model(x_tensor.double())

    pred_mask = pred_mask.detach().squeeze().cpu().numpy()
    # Convert pred_mask from `CHW` format to `HWC` format
    pred_mask = np.transpose(pred_mask,(1,2,0))

    # Only label encoded classes, no RGB
    label_encoded_mask = reverse_one_hot(pred_mask)

    # Map label encoded mask to RGB values
    # Get prediction channel corresponding to foreground
    pred_foreground_heatmap = crop_image(pred_mask[:,:,select_classes.index('foreground')], (320, 320))['image']
    rgb_mask = crop_image(colour_code_segmentation(reverse_one_hot(pred_mask), select_class_rgb_values), (320, 320))['image']

    # What objects are in the image and how many pixels for each object?
    objects = {}
    for i in range(len(select_classes)):
        # print(select_classes[i])
        segment = select_class_rgb_values[i]
        count = np.count_nonzero(np.all(rgb_mask == segment, axis=2))
        objects[select_classes[i]] = count

    return label_encoded_mask, rgb_mask, pred_foreground_heatmap, objects