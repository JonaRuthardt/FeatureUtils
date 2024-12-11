import torch
from torch.utils.data import Dataset

class TorchFeatureDataset(Dataset):
    def __init__(self, feature_io, keys=None, features=None):
        self.feature_io = feature_io
        self.features = features
        keys = keys if keys is not None else feature_io.list_keys()
        self.idx2keys = {idx: features_id for idx, features_id in enumerate(keys)}

    def __len__(self):
        return len(self.idx2keys)

    def __getitem__(self, idx):
        key = self.idx2keys[idx]
        features = self.features if self.features is not None else self.feature_io.list_features(key)
        return self.feature_io.load_feature(key, features)