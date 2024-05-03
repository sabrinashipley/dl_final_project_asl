import argparse
import csv
from PIL import Image
import numpy as np
import pandas as pd
import random
import os

def write_batch_to_csv(image_batch, label_batch, image_file, label_file, mode='a'):
    print("writing")
    i = 0
    with open(image_file, mode, newline='') as f_image, open(label_file, mode, newline='') as f_label:
        image_writer = csv.writer(f_image)
        label_writer = csv.writer(f_label)
        if mode == 'w':  # Only write header for the first batch
            image_writer.writerow(['pixel_{:04d}'.format(i) for i in range(len(image_batch[0]))])
            label_writer.writerow(['label'])
        for img, lbl in zip(image_batch, label_batch):
            i += 1
            image_writer.writerow(img)
            label_writer.writerow([lbl])
        print("done")
        print(i)

# Main function to load data to CSV
def load_data_to_csv(data_folder, image_file='preprocessed_images1.csv', label_file='preprocessed_labels1.csv', batch_size=1000):
    image_data = []
    label_data = []

    folders_file_path = os.path.join(data_folder, 'dataset5')

    # Determine the mode for the CSV files, 'w' to write header, then 'a' for subsequent appends
    write_mode = 'w'

    for root, dirs, files in os.walk(folders_file_path):
        files = [f for f in files if not f.startswith('.')]
        print(root)
        for filename in files:
            if filename.endswith('.png'):
                label = os.path.basename(root)
                img_path = os.path.join(root, filename)
                img = Image.open(img_path)
                img = img.resize((100, 100), Image.Resampling.LANCZOS)
                img_array = np.array(img)
                img_array = img_array.astype(np.float32) / 255.0
                img_array_flat = img_array.flatten().tolist()  # Flatten and convert to list
                image_data.append(img_array_flat)
                label_data.append(label)

                # When batch size is reached, write to CSV and reset lists
                if len(image_data) >= batch_size:
                    write_batch_to_csv(image_data, label_data, image_file, label_file, mode=write_mode)
                    image_data, label_data = [], []
                    write_mode = 'a'  # Subsequent batches will append to the CSV files

    # Write the final batch if there's any data left
    if image_data and label_data:
        write_batch_to_csv(image_data, label_data, image_file, label_file, mode=write_mode)

    print(f"Preprocessed images saved to {image_file}")
    print(f"Preprocessed labels saved to {label_file}")


# def load_data_to_csv(data_folder, image_file='preprocessed_images.csv', label_file='preprocessed_label.csv'):
#     image_data = []
#     label_data = []

#     files = ['A', 'B', 'C', 'D', 'E']
#     folders_file_path = f'{data_folder}/dataset5/'

#     for person in files:
#         file_path = folders_file_path + person + "/"
#         if os.path.isdir(file_path):
#             for letter in os.listdir(file_path):
#                 full_path = os.path.join(file_path, letter)
#                 if os.path.isdir(full_path):
#                     for filename in os.listdir(full_path):
#                         img_path = os.path.join(full_path, filename)
#                         if os.path.isfile(img_path):
#                             img = Image.open(img_path)
#                             img = img.resize((100, 100), Image.Resampling.LANCZOS)
#                             img_array = np.array(img)
#                             img_array = img_array.astype(np.float32) / 255.0
#                             img_array_flat = img_array.flatten()
#                             image_data.append(img_array_flat)
#                             label_data.append(letter)
    
#     image_frame = pd.DataFrame(image_data)
#     label_frame = pd.DataFrame(label_data)
#     image_frame.to_csv(image_file, index=False)
#     label_frame.to_csv(label_file, index=False)
#     print(f"Preprocessed images saved to {image_file}")
#     print(f"Preprocessed labels saved to {label_file}")
    
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str, help='The folder where the data is stored')

    args = parser.parse_args()

    load_data_to_csv(args.data_folder)

if __name__ == '__main__':
    main()