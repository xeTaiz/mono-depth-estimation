from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, split):
        self.split = split
        if split == 'train':
            self.transform = self.preprocess_training
        else:
            self.transform = self.preprocess_validation

    def preprocess_training(self, raw_data):
        raise NotImplementedError

    def preprocess_validation(self, raw_data):
        raise NotImplementedError

    def get_raw(self, index):
        raise NotImplementedError
    
    def __getitem__(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True # for older python versions
        raw_data = self.get_raw(index)
        return self.transform(raw_data)

    def __len__(self):
        return len(self.images)