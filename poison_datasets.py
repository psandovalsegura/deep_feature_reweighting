import os
import pickle
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms, datasets

from constants import CIFAR10_ROOT, POISON_ROOTS, TAP_FORMAT_POISONS

class NTGA(torchvision.datasets.VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        x_file = 'x_train_cifar10_ntga_fnn_best.npy'
        y_file = 'y_train_cifar10.npy' # in one-hot encoding
        x = np.load(os.path.join(root, x_file))
        y = np.load(os.path.join(root, y_file))
        x = (x * 255).astype(np.uint8)
        y = np.argmax(y, axis=1)       # convert to index label
        self.data = x
        self.targets = y
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target 
    
    def __len__(self):
        return len(self.data)

class TAPFormatPoison(torch.utils.data.Dataset):
    """
    A poison format consisting of a data/ folder with images named
    {base_idx}.png, where base_idx is the index of the image in the
    base dataset.
    """
    def __init__(self, root, baseset):
        self.baseset = baseset
        self.transform = self.baseset.transform
        self.samples = sorted(os.listdir(os.path.join(root, 'data')))
        self.root = root

    def __len__(self):
        return len(self.baseset)

    def __getitem__(self, idx):
        base_idx = int(self.samples[idx].split('.')[0])
        _, label = self.baseset[base_idx]
        return self.transform(Image.open(os.path.join(self.root, 'data',
                                            self.samples[idx]))), label

class UnlearnablePoison(torch.utils.data.Dataset):
    def __init__(self, root, baseset):
        self.baseset = baseset
        self.root = root
        self.classwise = 'classwise' in root
        noise = torch.load(os.path.join(root, 'perturbation.pt'))

        # Load images into memory to prevent IO from disk
        self._perturb_images(noise)

    def __len__(self):
        return len(self.baseset)

    def __getitem__(self, idx):
        return self.baseset[idx]

    def _perturb_images(self, noise):
        if 'stl' in self.root.lower() or 'svhn' in self.root.lower():
            perturb_noise = noise.mul(255).clamp_(-255, 255).to('cpu').numpy()
        else:
            perturb_noise = noise.mul(255).clamp_(-255, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        self.baseset.data = self.baseset.data.astype(np.float32)
        for i in range(len(self.baseset)):
            if self.classwise:
                self.baseset.data[i] += perturb_noise[self.baseset.targets[i]]
            else: # samplewise
                self.baseset.data[i] += perturb_noise[i]
            self.baseset.data[i] = np.clip(self.baseset.data[i], a_min=0, a_max=255)
        self.baseset.data = self.baseset.data.astype(np.uint8)

class RobustErrorMin(torch.utils.data.Dataset):
    def __init__(self, root, baseset):
        self.baseset = baseset
        self.root = root
        
        with open(root, 'rb') as f:
            raw_noise = pickle.load(f)
            self._perturb_images(raw_noise)

    def __len__(self):
        return len(self.baseset)

    def __getitem__(self, idx):
        return self.baseset[idx]
    
    def _perturb_images(self, raw_noise):
        assert isinstance(raw_noise, np.ndarray)
        assert raw_noise.dtype == np.int8

        raw_noise = raw_noise.astype(np.int16)

        noise = np.zeros_like(raw_noise)
        indices = np.random.permutation(len(noise))
        noise[indices] += raw_noise[indices]

        ''' restore noise (NCWH) for raw images (NHWC) '''
        noise = np.transpose(noise, [0,2,3,1])

        ''' add noise to images (uint8, 0~255) '''
        imgs = self.baseset.data.astype(np.int16) + noise
        imgs = imgs.clip(0,255).astype(np.uint8)
        self.baseset.data = imgs

def get_train_dataset(dataset_name, batch_size, num_workers, percent_train):    
    transform_list = [transforms.RandomCrop(32, padding=4),
                      transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(), 
                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]
    transform = transforms.Compose(transform_list)

    if dataset_name == 'clean':
        ds = datasets.CIFAR10(root=CIFAR10_ROOT, train=True, download=False, transform=transform)
    else:
        raise ValueError(f'Dataset {dataset_name} not supported!')

    # Take a subset of the training data
    indices = torch.randperm(len(ds)).tolist()
    train_indices = indices[:int(len(indices)*percent_train)]
    train_subset = torch.utils.data.Subset(ds, train_indices)

    loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader

def get_test_dataset(batch_size, num_workers, normalize=True):
    transform_list = [transforms.ToTensor()]
    if normalize:
        transform_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)))
    transform = transforms.Compose(transform_list)
    ds = datasets.CIFAR10(root=CIFAR10_ROOT, train=False, download=False, transform=transform)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader