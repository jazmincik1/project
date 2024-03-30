from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image


class AnimalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = sorted(os.listdir(root_dir))
        self.class_names = [name for name in self.class_names if os.path.isdir(os.path.join(root_dir, name))]

        for idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
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
