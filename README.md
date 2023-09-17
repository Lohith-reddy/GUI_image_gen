# Aim

To create a pipeline of models that generate pictures from user input. Unlike traditional models this pipeline allows the user more control over the output. It is long known that humans do not excel at giving verbal instructions. An art director knows where a particular object should be but would face a hard time describing the same exactly in words. They can better express their demands on a canvas as rough images or points. Keeping that in mind, this project creates a pipeline of models that take GUI based inputs from the user and creates images.

A perfect pipeline would give many controls to the user.
     color,
     position,
     size,
     layer,
     lighting,
     orientation etc.

We will tackle the problem incrementally.

Current focus of audience - 2d animations. - cartoons, anime, explainer videos etc.
aim: Decrease the time and resources spent to create animation videos for youtube or other platforms.

## Step 1 - Color

Unet keras model to color black and white images

Autoencoder-decoder with skip connections

dataset with 3000 images in Lab format used - <https://www.kaggle.com/datasets/shravankumar9892/image-colorization>

Implemented on AWS sagemaker

Didn't deploy as performance is expectedly low

### V2

convert it into tensorflow model for distributed computing

increase dataset size - Used a different dataset - <https://huggingface.co/datasets/skytnt/fbanimehq>

Processed the data to fetch major colours in each image (by clustering). Used these colors data to find out where the colour is present in each image. These coordinates are then used to create an image with the color blots at specified coordinates.

The model is expected to generate the right colours using the BW image and the color blots image.

The following is the result after training on 2.8k images.
<img width="818" alt="2800ol" src="https://github.com/Lohith-reddy/recolourise/assets/26896217/21f8157e-5d68-4c8e-8cdb-66a3944e0a17">
<img width="814" alt="2800ol2" src="https://github.com/Lohith-reddy/recolourise/assets/26896217/e01f79c2-fe15-495c-8e3b-784d1f74dfe5">
<img width="829" alt="2800ol3" src="https://github.com/Lohith-reddy/recolourise/assets/26896217/c994f45d-d703-4ec3-9685-c39a65a6a242">


Though the model does a good job in certain aspects, it generates bunch of colours in some areas to decrease the overall loss. To counter this I have added a new loss function that punishes the model for using too many colours. (the new loss function is a combination of mae and number of colours used. the balance between mae and number of colours can be changed by modifying a coefficient which provides higher control)

*insert output here
<img width="727" alt="4200nl" src="https://github.com/Lohith-reddy/recolourise/assets/26896217/049a53f0-c7d0-43bf-ae90-1f5c81113df0">
![4200nl3](https://github.com/Lohith-reddy/recolourise/assets/26896217/d00048a4-d872-468c-95e5-8746abc34bf9)
![4200nl2](https://github.com/Lohith-reddy/recolourise/assets/26896217/5822aeba-21d2-4870-8b1b-810b1b3a109a)
![4200nl4](https://github.com/Lohith-reddy/recolourise/assets/26896217/d8749f38-e61b-4736-a818-a19c4fbcd2b6)


Improve unet with inception resnet for global features - Did not seem necessary as the model did okay. Will add globlal models to the sketching model.

Add in an adversial part to improve performance - Did not seem necessary as the model was doing pretty okay without adversial part. Might add adversial part to the sketching model.

### further experimentation

MSE instead of MAE

Use different coefficients

Retrain model from the start with the new loss function (Initially, I have added the new loss function to the model after training with 2.8k images)

### problems

Can only train 1000 images per batch on colab due to RAM and GPU restriction
This might be hindering performance as the model is looking at only a subset of images each training round.

## Step 2 Position and Size

Add a model to the pipeline that creates images of specific size and in specific locations of the picture based on user input (boxes and name tags (specifying the name of the object) on a blank canvas).

Dataset: image classification dataset outputs (for any image the data set will have a description of detected objects and a box to determine the size and location)

the image generated will have colour which the user can readily accept or input their choice of colors as blots on the B/W image.

## Step 3

## Step N

Deploy on a web-server - Pending. Requires learning Django or similar framework. Hence postponing.

Add canvas to allow the user to input boxes and tags.

Add Gui option to choose colours based on which the model the model will produce the image

Craft the error term accordingly (balance between opted colour error and adversial error the later is trying to keep the image real while the former is trying to get the color pallette to match that of user inputs)
