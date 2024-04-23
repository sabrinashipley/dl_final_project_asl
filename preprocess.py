from PIL import Image
import numpy as np
import pandas as pd
import random
import os

def load_data_to_csv(data_folder, image_file='preprocessed_images.csv', label_file='preprocessed_label.csv'):
    image_data = []
    label_data = []

    files = ['A', 'B', 'C', 'D', 'E']
    folders_file_path = f'{data_folder}/dataset5/'

    for person in files:
        file_path = folders_file_path + person
        for letter in os.listdir(file_path):
            full_path = file_path + '/' + letter
            for filename in os.listdir(full_path):
                img_path = os.path.join(full_path, filename)
                img = Image.open(img_path)
                img = img.resize((100, 100), Image.ANTIALIAS)
                img_array = np.array(img)
                img_array = img_array.astype(np.float32) / 255.0
                img_array_flat = img_array.flatten()
                image_data.append(img_array_flat)
                label_data.append(letter)
    
    image_frame = pd.DataFrame(image_data)
    label_frame = pd.DataFrame(label_data)
    image_frame.to_csv(image_file, index=False)
    label_frame.to_csv(label_file, index=False)
    print(f"Preprocessed images saved to {image_file}")
    print(f"Preprocessed labels saved to {label_file}")
    
def load_images_from_csv(image_file='preprocessed_images.csv', label_file='preprocessed_label.csv', test_size=1000):
    image_frame = pd.read_csv(image_file)
    label_frame = pd.read_csv(label_file)
    images_array = image_frame.to_numpy()
    labels_array = label_frame.to_numpy()
    
    num_images = images_array.shape[0]
    image_height, image_width = int(np.sqrt(images_array.shape[1])), int(np.sqrt(images_array.shape[1]))
    images_array = images_array.reshape(num_images, image_height, image_width)

    random.seed(0)
    indices = np.arange(images_array.shape[0])
    random.shuffle(indices)
    
    shuffled_images = images_array[indices]
    shuffled_labels = labels_array[indices]
    
    test_images = shuffled_images[:test_size]
    test_labels = shuffled_labels[:test_size]
    train_images = shuffled_images[test_size:]
    train_labels = shuffled_labels[test_size:]

    return (np.array(train_images), np.array(test_images), np.array(train_labels), np.array(test_labels))