# Unet keras model to color black and white images

Autoencoder-decoder with skip connections

dataset with 3000 images in Lab format used

Implemented on AWS sagemaker

Didn't deploy as performance is expectedly low

This repository is concerned with the first version
## version 2

convert it into tensorflow model for distributed computing
increase dataset size
Improve unet with inception resnet for global features
Add in an adversial part to improve performance
Deploy on a web-server

## version 3

Add Gui option to choose colours based on which the model the model will produce the image
Craft the error term accordingly (balance between opted colour error and adversial error the later is trying to keep the image real while the former is trying to get the color pallette to match that of user inputs)
