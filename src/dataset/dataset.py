from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image


class AnimalDataset(Dataset):
    def __init__(self, root_dir, transform=None, use_all_images=False):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = sorted(os.listdir(root_dir))
        self.class_names = [name for name in self.class_names if os.path.isdir(os.path.join(root_dir, name))]

        self.class_image_count = {}
        for class_name in self.class_names:
            class_dir = os.path.join(root_dir, class_name)
            self.class_image_count[class_name] = len(os.listdir(class_dir))

        min_image_count = min(self.class_image_count, key=self.class_image_count.get)
        print(f"Minimum image count: {min_image_count} - {self.class_image_count[min_image_count]}")

        for idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(root_dir, class_name)
            class_dir_list = os.listdir(class_dir) if use_all_images else os.listdir(class_dir)[:min_image_count]
            for img_name in class_dir_list:
                if os.path.isfile(os.path.join(class_dir, img_name)):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
