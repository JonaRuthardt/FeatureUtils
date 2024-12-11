import os
import json
import torch
from pathlib import Path
from typing import Any, Dict, Optional, List
from featureutils.io import FeatureIO, ZIPFeatureIO
from featureutils.dataset import TorchFeatureDataset


class FeatureUtils:
    def __init__(self, base_dir: str, staging_dir: str = None, feature_num: int = 1,
                 storage_backend: str = "ZIP", shard_size: int = 10000):
        """
        Initializes the FeatureUtils library core class.

        Args:
            base_dir (str): Directory for storing feature shards and metadata.
            staging_dir (str): Directory for temporary files and staging area.
            feature_num (int): Number of features to manage for a given instance.
            storage_backend (str): Storage backend for feature data [one of "HDF5", "ZIP"].
            shard_size (int): Maximum number of features per shard.
            allow_overwrite (bool): Whether to allow overwriting existing features with same key.
        """
        
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir = Path(staging_dir) if staging_dir is not None else None
        self.staging_dir.mkdir(parents=True, exist_ok=True) if self.staging_dir is not None else None
        self.feature_num = feature_num

        # Load or initialize metadata and feature io
        assert storage_backend in ["ZIP"], "Invalid storage backend. Must be one of 'HDF5' or 'ZIP'."
        if storage_backend == "ZIP":
            self.feature_io = ZIPFeatureIO(self.base_dir, self.staging_dir, shard_size)
    
    def convert_key(self, key: Any) -> str:
        """
        Converts a key to a string.

        Args:
            key (Any): Key to convert to string.

        Returns:
            str: String representation of the key.
        """
        
        key = str(key) if key is not None else None
        
        return key
            
    def save_feature(self, key: str, **features: torch.Tensor) -> None:
        """
        Saves a feature to the appropriate shard.

        Args:
            key (str): Unique key for the feature.
            features (Dict[str, torch.Tensor]): Dictionary of feature names and tensors.
        """
        assert len(features) == self.feature_num, f"Expected {self.feature_num} features, got {len(features)}."
        
        self.feature_io.save_feature(self.convert_key(key), **features)
        
    def save_features(self, keys: List[str], features: Dict[str, torch.Tensor]) -> None:
        """
        Saves features of multiple keys to the appropriate shard.
        
        Args:
            keys (List[str]): List of unique keys for the features.
            features (Dict[str, Dict[str, torch.Tensor]]): Dictionary of keys and feature names and tensors.
        """
        
        for key_idx, key in enumerate(keys):
            self.feature_io.save_feature(self.convert_key(key), **{feature_name: features[feature_name][key_idx] for feature_name in features})

    def load_feature(self, key: str, feature_names: List[str]) -> Dict[str, torch.Tensor]:
        """
        Loads a feature by its key.

        Args:
            key (str): Unique key for the feature.
            feature_name (List[str]): List of feature names to load.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of feature names and tensors.
        """

        return self.feature_io.load_feature(self.convert_key(key), feature_names)

    def delete_feature(self, key: str) -> None:
        """
        Deletes a feature by its key.

        Args:
            key (str): Unique key for the feature.
        """
        
        self.feature_io.delete_feature(self.convert_key(key))
    
    def list_keys(self) -> List[str]:
        """
        List all unique keys available in the feature store.
        
        Returns:
            List[str]: List of unique keys.
        """
        
        return self.feature_io.list_keys()

    def list_features(self, key: str = None) -> List[str]:
        """
        List all types of features avaialble for a given key.
        
        Args:
            key (str): Unique key for the feature.
        
        Returns:
            List[str]: List of feature names.
        """
        
        return self.feature_io.list_features(self.convert_key(key))
    
    def stage_data(self) -> None:
        """
        Stages data to disk.
        """
        
        self.save()
        self.feature_io.stage_data()
    
    def save(self) -> None:
        """
        Saves state to main disk (relevant when utilizing faster staging storage).
        """
            
        self.feature_io.save_metadata()
        self.feature_io.save_shard()
        
    def get_dataset(self, keys: List[str] = None, features: List[str] = None) -> TorchFeatureDataset:
        """
        Returns a Torch dataset for the given keys.
        
        Args:
            keys (List[str]): List of keys to include in the dataset.
            features (List[str]): List of features to include in the dataset.
            
        Returns:
            TorchFeatureDataset: Torch dataset for the given keys.
        """
        
        self.save()
        if keys is not None:
            keys = [self.convert_key(key) for key in keys]
        return TorchFeatureDataset(self.feature_io, keys, features)
    
    def __del__(self):
        """
        Destructor to ensure data is saved when the object is deleted.
        """
        self.save()