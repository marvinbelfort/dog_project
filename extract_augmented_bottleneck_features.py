# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 10:27:42 2019

@author: Marvin
"""




import numpy as np
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
import math
from PIL import ImageFile
from keras.applications.xception import Xception


# address a bug when using ImageDataGenerator and the InceptionV3 model
ImageFile.LOAD_TRUNCATED_IMAGES = True



model = Xception(include_top=False)
batch_size = 32

imagespath='dogImages'
imageset={
        "train": '{}/train'.format(imagespath),
        "valid": '{}/valid'.format(imagespath),
        "test" :'{}/test'.format(imagespath)
        }


augmentation_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    
non_augmentation_datagen = ImageDataGenerator(rescale=1./255) 
  
def create_augmented_bottleneck():


    
    train_generator = augmentation_datagen.flow_from_directory(
            imageset["train"],
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False)
    
    number_of_train_samples = len(train_generator.filenames)
    number_of_train_steps = int(math.ceil(number_of_train_samples / float(batch_size)))
    
    train_bottleneck_features = model.predict_generator(train_generator, number_of_train_steps, verbose=1)
    
    valid_generator = augmentation_datagen.flow_from_directory(
            imageset["valid"],
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False)
    
    number_of_valid_samples = len(valid_generator.filenames)
    number_of_valid_steps = int(math.ceil(number_of_valid_samples / float(batch_size)))
    
    valid_bottleneck_features = model.predict_generator(valid_generator, number_of_valid_steps, verbose=1)
    
    test_generator = non_augmentation_datagen.flow_from_directory(
            imageset["test"],
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False)
    
    number_of_test_samples = len(test_generator.filenames)
    number_of_test_steps = int(math.ceil(number_of_test_samples / float(batch_size)))
    
    test_bottleneck_features = model.predict_generator(test_generator, number_of_test_steps, verbose=1)
    
    np.savez(
            'bottleneck_features/xception_augmentend_bottleneck_features.npz',
            train=train_bottleneck_features,
            valid=valid_bottleneck_features,
            test=test_bottleneck_features)

def train():
    # load the bottleneck features
    bottleneck_features = np.load('bottleneck_features/xception_augmentend_bottleneck_features.npz')
    
    # configure training data generator
    train_generator = augmentation_datagen.flow_from_directory(
            imageset["train"],
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False)
    
    #number_of_train_samples = len(train_generator.filenames)
    number_of_train_classes = len(train_generator.class_indices)
    #number_of_train_steps = int(math.ceil(number_of_train_samples / float(batch_size)))
    
    # get the training bottleneck features and class labels
    train_data = bottleneck_features['train']
    train_labels = train_generator.classes
    train_labels = to_categorical(train_labels, num_classes=number_of_train_classes)
    
    # configure validation data generator
    valid_generator = augmentation_datagen.flow_from_directory(
            imageset["valid"],
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False)
    
    #number_of_valid_samples = len(valid_generator.filenames)
    number_of_valid_classes = len(valid_generator.class_indices)
    #number_of_valid_steps = int(math.ceil(number_of_valid_samples / float(batch_size)))
    
    # get the validation bottleneck features and class labels
    valid_data = bottleneck_features['valid']
    valid_labels = valid_generator.classes
    valid_labels = to_categorical(valid_labels, num_classes=number_of_valid_classes)
    
    # configure testing data generator
    test_generator = non_augmentation_datagen.flow_from_directory(
            imageset["test"],
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False)
    
    #number_of_test_samples = len(test_generator.filenames)
    number_of_test_classes = len(test_generator.class_indices)
    #number_of_test_steps = int(math.ceil(number_of_test_samples / float(batch_size)))
    
    # get the testing bottleneck features and class labels
    #test_data = bottleneck_features['test']
    test_labels = test_generator.classes
    test_labels = to_categorical(test_labels, num_classes=number_of_test_classes)
    
        # define the a new model
    new_model = Sequential()
    new_model.add(GlobalAveragePooling2D(input_shape=train_data.shape[1:]))
    
    # add fully connected layer
    new_model.add(Dense(number_of_train_classes, activation='relu'))
    new_model.add(Dropout(0.20))
    # add fully connected layer
    new_model.add(Dense(number_of_train_classes, activation='softmax'))
    
    # compile the model
    new_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    # train the new model:
    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.xception.augmented.hdf5', verbose=1, save_best_only=True)
    
    
    new_model.fit(train_data, train_labels, 
              validation_data=(valid_data, valid_labels),
              epochs=50, batch_size=batch_size, callbacks=[checkpointer], verbose=1)
    

train()
