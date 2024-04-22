import pickle
import random
import re
from PIL import Image
import tensorflow as tf
import numpy as np
import collections
from tqdm import tqdm

def get_image_features(image_names, data_folder, vis_subset=100):
    '''
    Method used to extract the features from the images in the dataset using ResNet50
    '''
    image_features = []
    vis_images = []
    resnet = tf.keras.applications.ResNet50(False)  ## Produces Bx7x7x2048
    gap = tf.keras.layers.GlobalAveragePooling2D()  ## Produces Bx2048
    pbar = tqdm(image_names)
    for i, image_name in enumerate(pbar):
        img_path = f'{data_folder}/Images/{image_name}'
        pbar.set_description(f"[({i+1}/{len(image_names)})] Processing '{img_path}' into 2048-D ResNet GAP Vector")
        with Image.open(img_path) as img:
            img_array = np.array(img.resize((224,224)))
        img_in = tf.keras.applications.resnet50.preprocess_input(img_array)[np.newaxis, :]
        image_features += [gap(resnet(img_in))]
        if i < vis_subset:
            vis_images += [img_array]
    print()
    return image_features, vis_images

def load_data(data_folder):
    '''
    Method that was used to preprocess the data in the data.p file. You do not need 
    to use this method, nor is this used anywhere in the assignment. This is the method
    that the TAs used to pre-process the Flickr 8k dataset and create the data.p file 
    that is in your assignment folder. 

    Feel free to ignore this, but please read over this if you want a little more clairity 
    on how the images and captions were pre-processed 
    '''
    files = []
    images_file_path = f'{data_folder}/dataset5'

    for letter in files:
        letter_path = images_file_path + letter
    with open(images_file_path) as file:
        for 
    

    #randomly split examples into training and testing sets
    shuffled_images = list(image_names_to_captions.keys())
    random.seed(0)
    random.shuffle(shuffled_images)
    test_image_names = shuffled_images[:1000]
    train_image_names = shuffled_images[1000:]

    def get_all_captions(image_names):
        to_return = []
        for image in image_names:
            captions = image_names_to_captions[image]
            for caption in captions:
                to_return.append(caption)
        return to_return


    # get lists of all the captions in the train and testing set
    train_captions = get_all_captions(train_image_names)
    test_captions = get_all_captions(test_image_names)

    #remove special charachters and other nessesary preprocessing
    window_size = 20
    preprocess_captions(train_captions, window_size)
    preprocess_captions(test_captions, window_size)

    # count word frequencies and replace rare words with '<unk>'
    word_count = collections.Counter()
    for caption in train_captions:
        word_count.update(caption)

    def unk_captions(captions, minimum_frequency):
        for caption in captions:
            for index, word in enumerate(caption):
                if word_count[word] <= minimum_frequency:
                    caption[index] = '<unk>'

    unk_captions(train_captions, 50)
    unk_captions(test_captions, 50)

    
    # use ResNet50 to extract image features
    print("Getting training embeddings")
    train_image_features, train_images = get_image_features(train_image_names, data_folder)
    print("Getting testing embeddings")
    test_image_features,  test_images  = get_image_features(test_image_names, data_folder)

    return dict(
        train_captions          = np.array(train_captions),
        test_captions           = np.array(test_captions),
        train_image_features    = np.array(train_image_features),
        test_image_features     = np.array(test_image_features),
        train_images            = np.array(train_images),
        test_images             = np.array(test_images),
        word2idx                = word2idx,
        idx2word                = {v:k for k,v in word2idx.items()},
    )


