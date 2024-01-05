import os
from PIL import Image
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import random

class VessMapDataset(Dataset):
    def __init__(self, image_dir, mask_dir, skeleton_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.skeleton_dir = skeleton_dir
        self.transform = transform
        self.pairs = self._find_pairs(image_dir, mask_dir, skeleton_dir)

    def _find_pairs(self, image_dir, mask_dir, skeleton_dir):
        pairs = []
        for img_file in os.listdir(image_dir):
            base_name = os.path.splitext(img_file)[0]
            mask_file = os.path.join(mask_dir, base_name + '.png')
            skeleton_file = os.path.join(skeleton_dir, base_name + '.png')
            if os.path.exists(mask_file) and os.path.exists(skeleton_file):
                pairs.append(img_file)
        return pairs

    def __len__(self):
        return len(self.pairs)

    def apply_transform(self, image, mask, skeleton, seed):
        random.seed(seed)  # Synchronize transformations
        if random.random() > 0.5:
            image, mask, skeleton = [TF.hflip(x) for x in [image, mask, skeleton]]
            
        if random.random() > 0.5:
            image, mask, skeleton = [TF.vflip(x) for x in [image, mask, skeleton]]

        i, j, h, w = transforms.RandomResizedCrop.get_params(image,
                                                             scale=(0.8, 1.0), 
                                                             ratio=(0.75, 1.33))
        image, mask, skeleton = [
            TF.resized_crop(x, i, j, h, w, (224, 224)) for x in [image, mask, skeleton]]

        image, mask, skeleton = [TF.to_tensor(x) for x in [image, mask, skeleton]]
        return image, mask, skeleton
    
    def __getitem__(self, idx):
        img_file = self.pairs[idx]
        image_path = os.path.join(self.image_dir, img_file)
        base_name = os.path.splitext(img_file)[0]
        mask_path = os.path.join(self.mask_dir, base_name + '.png')
        skeleton_path = os.path.join(self.skeleton_dir, base_name + '.png')

        image = Image.open(image_path)
        mask = Image.open(mask_path)
        skeleton = Image.open(skeleton_path)

        if self.transform:
            seed = random.randint(0, 2**32)
            image, mask, skeleton = self.apply_transform(image, mask, skeleton, seed)

        return image, mask, skeleton

from torch.utils.data import DataLoader

class OversamplingDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, num_epochs_oversample, shuffle=True):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)
        self.dataset = dataset
        self.num_epochs_oversample = num_epochs_oversample

    def __iter__(self):
        for _ in range(self.num_epochs_oversample):
            for batch in super().__iter__():
                yield batch

    def __len__(self):
        return len(self.dataset) * self.num_epochs_oversample


def vess_map_dataloader(image_dir, mask_dir, skeleton_dir, batch_size, train_size, shuffle=True):
    """ transform = TF.Compose([
        TF.RandomHorizontalFlip(),
        TF.RandomVerticalFlip(),
        TF.RandomRotation(30),
        TF.RandomResizedCrop(224, scale=(0.8, 1.0)),
        TF.ToTensor(),
    ]) """

    dataset = VessMapDataset(image_dir, mask_dir, skeleton_dir, transform = True)
    
    dataset_size = len(dataset)
    train_size = int(train_size * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
