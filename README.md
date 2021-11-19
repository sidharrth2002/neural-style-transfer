# Semantically Segmented Neural Style Transfer
### Combined semantic segmentation with multiple style transfer

#### Abstract

This work proposes a web application that demonstrates the power of neural style transfer when coupled with foreground-background and automated semantic segmentation. Pioneered training on the Stanford Background dataset, the style transfer app is capable of generating remarkably aesthetic results albeit unconventional ones at times. The main motivation of this application is to enable users to convert their memorable moments captured in images into wonderful works of art from various well-known styles. Reference code for this project is available at: https://github.com/sidharrth2002/neural-style-transfer

#### Introduction

Pablo Picasso once said “good artists copy, great artists steal”. In art, there have been countless attempts shown by artists to compose unique visual experiences by combining the style of an image and the content of another image. Recent advances in Convolutional Neural Networks have enabled researchers to create artificial systems that generate artistic images with high perceptual accuracy. The problem that we propose to study is style transfer - a process of transferring the semantic content of an image to other images of different styles.

Style transfer algorithms have shown remarkable performance in recent years. In this work, we propose a semantically segmented style transfer algorithm that combines the benefits of scene parsing with style transfer to generate aesthetically interesting graphics. A combination of several style transfer algorithms, segmentation models and image processing techniques will be fused into a pipeline for what we term semantically segmented artistic style transfer.

#### Datasets

Stanford Backgrounds

##### Segmentation

The project will employ a Stanford Background dataset that features images taken in different outdoor scenes with the  occasional cameo from passersby. This dataset will be used to assess the quality of outputs on the app.

There are several reasons why this dataset was chosen to provide training images for our model. For one, the dataset sports a total of 715 images while being merely 14 MB in size, making it particularly lightweight. This will allow space to be conserved, and reduce computation time needed to train the model as there are not that many images to begin with. Furthermore, the dataset aligns with the concept and idea of style transfer, where the goal is not to style everything, but rather only certain objects of note. The minimalist nature of the class labels in this dataset support this idea, making it easier for us to manipulate them accordingly.


#### Running the app

Ensure all requirements are downloaded. If not, create a virtual environment and:

```
pip3 install -r requirements.txt
```

After this, you can run the app:

```
streamlit run app.py
```

You have to go into /app/models and create a new folder called unet. Afterwhich, download the model from Google Drive and add it to the folder.
