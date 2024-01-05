import os
from PIL import Image
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.transforms as transforms


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.pairs = self._find_pairs(image_dir, mask_dir)

    def _find_pairs(self, image_dir, mask_dir):
        pairs = []
        for img_file in os.listdir(image_dir):
            base_name = os.path.splitext(img_file)[0]
            mask_file = os.path.join(mask_dir, base_name + '.png')
            if os.path.exists(mask_file):
                pairs.append(img_file)
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_file = self.pairs[idx]
        image_path = os.path.join(self.image_dir, img_file)
        base_name = os.path.splitext(img_file)[0]
        mask_path = os.path.join(self.mask_dir, base_name + '.png')

        image = Image.open(image_path)
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


def get_dataset(batch_size = 8, train_size = 0.8):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop((448, 448)),
    ]) 

    # Paths to your image and mask directories
    image_dir = "../data/images/"
    mask_dir = '../data/labels/'

    # Create the dataset
    dataset = SegmentationDataset(image_dir, mask_dir, transform)


    dataset_size = len(dataset)
    train_size = int(dataset_size * train_size)  
    val_size = dataset_size - train_size

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def get_dataloader(batch_size = 8):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop((448, 448)),
    ]) 

    # Paths to your image and mask directories
    image_dir = "../data/images/"
    mask_dir = '../data/labels/'

    # Create the dataset
    dataset = SegmentationDataset(image_dir, mask_dir, transform)


    return DataLoader(dataset, batch_size=batch_size, shuffle=True)