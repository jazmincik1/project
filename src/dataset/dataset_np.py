import os
import numpy as np
from PIL import Image

class DatasetNP:
    def __init__(self, root_dir, target_size=(224, 224)):
        self.root_dir = root_dir
        self.target_size = target_size
        self.images = []
        self.labels = []
        self.class_names = []
        self.load_images()

    def load_images(self):
        self.class_names = os.listdir(self.root_dir)
        self.class_names.sort()
        if '.DS_Store' in self.class_names:
            self.class_names.remove('.DS_Store')

        for label, class_name in enumerate(self.class_names):
            class_path = os.path.join(self.root_dir, class_name)
            image_names = os.listdir(class_path)
            if '.DS_Store' in image_names:
                image_names.remove('.DS_Store')

            for image_name in image_names:
                image_path = os.path.join(class_path, image_name)
                try:
                    image = Image.open(image_path).convert('RGB')
                    image = image.resize(self.target_size)
                    # Flatten the image to a 1D array
                    flattened_image = np.array(image).flatten()
                    self.images.append(flattened_image)
                    self.labels.append(label)
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")

        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

    def get_data(self):
        return self.images, self.labels
