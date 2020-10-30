from torch.utils.data.dataset import Dataset
import numpy as np

class BaseDataset(Dataset):
    def __init__(self, split):
        self.split = split
        if split == 'train':
            self.transform = self.training_preprocess
        elif split == 'val':
            self.transform = self.validation_preprocess
        elif split == 'test':
            self.transform = self.validation_preprocess
        else:
            raise (RuntimeError("Invalid dataset type: " + split + "\nSupported dataset types are: train, val, test"))

    def training_preprocess(self, rgb, depth):
        raise NotImplementedError()

    def validation_preprocess(self, rgb, depth):
        raise NotImplementedError()

    def get_raw(self, index):
        raise NotImplementedError()

    def __getitem__(self, index):
        rgb, depth = self.get_raw(index)
        return self.transform(rgb, depth)

    def __len__(self):
        return len(self.images)

class ConcatDataset(Dataset):
    def __init__(self, datasets):
        self.transform = None
        self.datasets = datasets
        self.indices = np.array([[dataset_index] * len(d) for dataset_index, d in enumerate(self.datasets)]).flatten()
        np.random.shuffle(self.indices)

    def __getitem__(self, i):
        if not self.transform is None:
            for dataset in self.datasets:
                dataset.transform = lambda x: x
        item_index = (self.indices[0:i] == self.indices[i]).sum()
        item = self.datasets[self.indices[i]][item_index]
        if self.transform is None:
            return item
        else:
            return self.transform(item)

    def __len__(self):
        return sum(len(d) for d in self.datasets)