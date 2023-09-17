#from pickle import load
import numpy as np
#import cv2
import pandas as pd
#import matplotlib.pyplot as plt
#import gc
#import tensorflow as tf
import keras.layers as L
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D
from keras.layers import UpSampling2D
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import RepeatVector
from keras.layers import Reshape
from keras.layers import concatenate
from keras.utils import plot_model
from sklearn.externals import joblib
import json
import argparse
#from s3fs.core import S3FileSystem

def unet():
    Xinpt = L.Input([None, None, 1])
    X0 = L.Conv2D(64, (3, 3), padding='same')(Xinpt)
    X0 = L.BatchNormalization()(X0)
    X0 = L.LeakyReLU(alpha=0.2)(X0)    #l,b,64
    X0 = L.Conv2D(64, (3, 3), strides=1, padding='same')(X0)
    X0 = L.BatchNormalization()(X0)
    X0 = L.LeakyReLU(alpha=0.2)(X0)    #l,b,64
    
    X1 = L.MaxPool2D((2, 2), strides=2)(X0)    #l/2,b/2,64
    X1 = L.Conv2D(128, (3, 3), padding='same')(X1)
    X1 = L.BatchNormalization()(X1)
    X1 = L.LeakyReLU(alpha=0.2)(X1)
    X1 = L.Conv2D(128, (3, 3), padding='same')(X1)
    X1 = L.BatchNormalization()(X1)
    X1 = L.LeakyReLU(alpha=0.2)(X1)    #l/2,b/2,128
    
    X2 = L.MaxPool2D((2, 2), strides=2)(X1)    #l/4,b/4,128
    X2 = L.Conv2D(256, (3, 3), padding='same')(X2)
    X2 = L.BatchNormalization()(X2)
    X2 = L.LeakyReLU(alpha=0.2)(X2)
    X2 = L.Conv2D(256, (3, 3), padding='same')(X2)
    X2 = L.BatchNormalization()(X2)
    X2 = L.LeakyReLU(alpha=0.2)(X2)    #l/4,b/4,256
    
    X3 = L.MaxPool2D((2, 2), strides=2)(X2)    #l/8,b/8,256
    X3 = L.Conv2D(512, (3, 3), padding='same')(X3)
    X3 = L.BatchNormalization()(X3)
    X3 = L.LeakyReLU(alpha=0.2)(X3)
    X3 = L.Conv2D(512, (3, 3), padding='same')(X3)
    X3 = L.BatchNormalization()(X3)
    X3 = L.LeakyReLU(alpha=0.2)(X3)    #l/8,b/8,512
    
    X4 = L.MaxPool2D((2, 2), strides=2)(X3)    #l/16,b/16,512
    X4 = L.Conv2D(1024, (3, 3), padding='same')(X4)
    X4 = L.BatchNormalization()(X4)
    X4 = L.LeakyReLU(alpha=0.2)(X4)
    X4 = L.Conv2D(1024, (3, 3), padding='same')(X4)
    X4 = L.BatchNormalization()(X4)
    X4 = L.LeakyReLU(alpha=0.2)(X4)    #l/16,b/16,1024
    
    X4 = L.Conv2DTranspose(512, (2, 2), strides=2)(X4)    #l/8,b/8,512
    X4 = L.Concatenate()([X3, X4])     #l/8,b/8,1024
    X4 = L.Conv2D(512, (3, 3), padding='same')(X4)
    X4 = L.BatchNormalization()(X4)
    X4 = L.Activation('relu')(X4)
    X4 = L.Conv2D(512, (3, 3), padding='same')(X4)
    X4 = L.BatchNormalization()(X4)
    X4 = L.Activation('relu')(X4)    #l/8,b/8,512
    
    X3 = L.Conv2DTranspose(256, (2, 2), strides=2)(X4)    #l/4,b.4,256
    X3 = L.Concatenate()([X2, X3])     #l/4,b/4,512
    X3 = L.Conv2D(256, (3, 3), padding='same')(X3)
    X3 = L.BatchNormalization()(X3)
    X3 = L.Activation('relu')(X3)
    X3 = L.Conv2D(256, (3, 3), padding='same')(X3)
    X3 = L.BatchNormalization()(X3)
    X3 = L.Activation('relu')(X3)    #l/4,b/4,256
    
    X2 = L.Conv2DTranspose(128, (2, 2), strides=2)(X3)    #l/2,b/2,128
    X2 = L.Concatenate()([X1, X2])     #l/2,b/2,256
    X2 = L.Conv2D(128, (3, 3), padding='same')(X2)
    X2 = L.BatchNormalization()(X2)
    X2 = L.Activation('relu')(X2)
    X2 = L.Conv2D(128, (3, 3), padding='same')(X2)
    X2 = L.BatchNormalization()(X2)
    X2 = L.Activation('relu')(X2)   #l/2,b/2,128
    
    X1 = L.Conv2DTranspose(64, (2, 2), strides=2)(X2)    #l,b,64
    X1 = L.Concatenate()([X0, X1])    #l,b,128
    X1 = L.Conv2D(64, (3, 3), padding='same')(X1)
    X1 = L.BatchNormalization()(X1)
    X1 = L.Activation('relu')(X1)
    X1 = L.Conv2D(64, (3, 3), padding='same')(X1)
    X1 = L.BatchNormalization()(X1)
    X1 = L.Activation('relu')(X1)    #l,b,64
    
    X0 = L.Conv2D(3, (1, 1), strides=1)(X1)     #l,b,3 
    model = Model(inputs=Xinpt, outputs=X0)
    return model

def model(gray_train, train, gray_test, test):
    """Generate a simple model"""
    model = unet()
    model.compile('adam', loss='mean_squared_error', metrics=['mae', 'acc'])
    for epoch in range(1, args.epochs):
        model.fit(gray_train, train, batch_size= args.batch-size, validation_data=(gray_test, test))
    score = model.evaluate(gray_test, test)
    print('Test loss:', score[0]) 
    print('Test accuracy:', score[1])
    return model


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.05)

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    return parser.parse_known_args()


def load_data(base_dir):
    bucket = 'recolorising'
    train = np.load(os.path.join(base_dir,'train.npy'))
    test = np.load(os.path.join(base_dir,'test.npy'))
    gray_test = np.load(os.path.join(base_dir,'gray_test.npy'))
    gray_train = np.load(os.path.join(base_dir,'gray_train.npy'))
    return train, test, gray_test, gray_train

if __name__ == "__main__":
    args, unknown = _parse_args()

    train, test, gray_test, gray_train = load_data(args.train)

    recolor = model(gray_train, train, gray_test, test)

    if args.current_host == args.hosts[0]:
        # save model to an S3 directory with version number '00000001' 
        # To export the model as h5 format use model.save('my_model.h5')
        joblib.dump(recolor, os.path.join(args.model_dir, "model.joblib"))
