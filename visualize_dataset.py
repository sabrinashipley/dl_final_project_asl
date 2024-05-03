import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

class DisplayPreprocessedData:
    def __init__(self, image_file, label_file, num_to_display=5):
        self.image_file = image_file
        self.label_file = label_file
        self.num_to_display = num_to_display

    def display_images(self):
        # Determine total lines excluding header
        total_lines = sum(1 for line in open(self.image_file)) - 1
        random_indices = sorted(random.sample(range(1, total_lines + 1), self.num_to_display))  # Skip header by starting from 1

        # Load the sampled lines for images and labels
        images = pd.read_csv(self.image_file, skiprows=lambda i: i not in random_indices, header=None)
        labels = pd.read_csv(self.label_file, skiprows=lambda i: i not in random_indices, header=None)

        # Display the images and labels
        for i, (img, lbl) in enumerate(zip(images.values, labels[0])):
            plt.subplot(1, self.num_to_display, i + 1)
            plt.imshow(img.reshape(100, 100, 3))  # Assuming RGB images
            plt.title(f'Label: {lbl}')
            plt.axis('off')
        plt.show()

# Command line interface setup
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Display preprocessed images with labels.')
    parser.add_argument('--image_file', type=str, default='preprocessed_images.csv',
                        help='CSV file containing image data.')
    parser.add_argument('--label_file', type=str, default='preprocessed_labels.csv',
                        help='CSV file containing label data.')
    parser.add_argument('--num_to_display', type=int, default=5,
                        help='Number of images to display.')

    args = parser.parse_args()

    display_data = DisplayPreprocessedData(args.image_file, args.label_file, args.num_to_display)
    display_data.display_images()
