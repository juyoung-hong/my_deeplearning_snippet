import os

import cv2
import pandas as pd
from torch.utils.data import Dataset


class TinyImageNetTrainDataset(Dataset):
    def __init__(self, root, transform=None):
        self.labels = {l: c for c, l in enumerate(os.listdir(root))}
        self.imgs = []
        for path, dir, files in os.walk(root):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext == ".JPEG":
                    self.imgs.append(os.path.join(path, filename))
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.imread(self.imgs[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = os.path.basename(self.imgs[idx]).split("_")[0]
        label = self.labels[label]  # map class -> integer
        if self.transform:
            img = self.transform(img)
        return (img, label)


class TinyImageNetValDataset(Dataset):
    def __init__(self, labels, root, transform=None):
        self.labels = labels
        anns = pd.read_csv(
            os.path.join(root, "val_annotations.txt"), sep="\t", header=None
        )
        self.class_mapper = {anns[0][row]: anns[1][row] for row in range(anns.shape[0])}
        self.imgs = []
        for path, dir, files in os.walk(root):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext == ".JPEG":
                    self.imgs.append(os.path.join(path, filename))
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.imread(self.imgs[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = os.path.basename(self.imgs[idx])
        label = self.class_mapper[label]  # map filename -> class
        label = self.labels[label]  # map class -> integer
        if self.transform:
            img = self.transform(img)
        return (img, label)
