import os
import io
import datetime
import torch
from tqdm import tqdm
from pathlib import Path
import threading
import portalocker
import json
from typing import Any, Dict, List
from abc import ABC, abstractmethod
import zipfile
import shutil

class FeatureIO(ABC):
    """
    Abstract base class for feature storage backends.
    """
    
    def __init__(self, base_dir: Path, staging_dir: Path, shard_size: int, require_features_exist=False):
        """
        Initializes the feature I/O base class that handles basic feature storage operations.
        All feature I/O implementations should inherit from this class.
        
        Args:
            base_dir (str): Directory for storing feature shards.
            staging_dir (str): Directory for temporary files and staging area.
            shard_size (int): Maximum number of features per shard.
            require_features_exist (bool): Whether to raise an error if features do not exist.
        """
        
        self.base_dir = base_dir
        self.staging_dir = staging_dir
        self.shard_size = shard_size
        self.staged = False
        
        self.thread_lock = threading.Lock()
        
        self.metadata_file = self.base_dir / "metadata.json"
        
        self._load_metadata(require_features_exist)
        self.tmp_metadata = self._get_default_metadata()

    @abstractmethod
    def _get_default_metadata(self) -> Dict[str, Any]:
        """
        Returns the default metadata for feature management.
        
        Returns:
            Dict[str, Any]: Default metadata dictionary.
        """
        raise NotImplementedError

    def _load_metadata(self, require_features_exist: bool) -> None:
        """
        Loads metadata for feature management.
        
        Args:
            require_features_exist (bool): Whether to raise an error if features do not exist.
        """
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                portalocker.lock(f, portalocker.LockFlags.SHARED)
                self.metadata = json.load(f)
                portalocker.lock(f, portalocker.LockFlags.UNBLOCK)
        else:
            if require_features_exist:
                raise ValueError(f"No metadata for features found at {self.metadata_file}. Ensure path is correct or store features first.")
            self.metadata = self._get_default_metadata()
            with open(self.metadata_file, 'w') as f:
                portalocker.lock(f, portalocker.LockFlags.EXCLUSIVE)
                json.dump(self.metadata, f)
                portalocker.lock(f, portalocker.LockFlags.UNBLOCK)
                
    def save_metadata(self) -> None:
        """
        Updates the metadata with new entries and saves to disk.
        """
        if self.tmp_metadata["feature_count"] > 0:
            with self.thread_lock:
                
                with open(self.metadata_file, 'r+') as f:
                    portalocker.lock(f, portalocker.LockFlags.EXCLUSIVE)
                    # Load most recent metadata
                    self.metadata = json.load(f)
                    f.seek(0)

                    # Update metadata
                    for key, value in self.tmp_metadata.items():
                        if type(value) == dict:
                            self.metadata[key].update(value)
                        else:
                            self.metadata[key] += value
                            
                    # Save metadata
                    json.dump(self.metadata, f, indent=4)
                    portalocker.lock(f, portalocker.LockFlags.UNBLOCK)
                
                self.tmp_metadata = self._get_default_metadata()
    
    @abstractmethod
    def save_shard(self) -> None:
        """
        Save the current shard to main base directory.
        """
        raise NotImplementedError
    
    @abstractmethod
    def save_feature(self, key: str, **features: torch.Tensor) -> None:
        """
        Save a feature by its key.
        
        Args:
            key (str): Unique key for the feature.
            features (Dict[str, torch.Tensor]): Dictionary of feature names and tensors.
        """
        raise NotImplementedError
    
    @abstractmethod
    def delete_feature(self, key: str) -> None:
        """
        Delete a feature by its key.
        
        Args:
            key (str): Unique key for the feature.
        """
        raise NotImplementedError
    
    @abstractmethod
    def load_feature(self, key: str, feature_names: List[str]) -> Dict[str, torch.Tensor]:
        """
        Load features by their key and names.
        
        Args:
            key (str): Unique key for the feature.
            feature_name (List[str]): List of feature names to load.
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary of feature names and tensors.
        """
        raise NotImplementedError

    @abstractmethod
    def list_keys(self) -> List[str]:
        """
        List all feature keys (IDs).
        
        Returns:
            List[str]: List of unique keys.
        """
        raise NotImplementedError
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if a feature exists by its key.
        
        Args:
            key (str): Unique key for the feature.
            
        Returns:
            bool: True if the feature exists, False otherwise.
        """
        raise NotImplementedError
    
    @abstractmethod
    def list_features(self, key: str = None) -> List[str]:
        """
        List all types of features avaialble for a given key.
        
        Args:
            key (str): Unique key for the feature.
            
        Returns:
            List[str]: List of available feature names.
        """
        raise NotImplementedError
    
    @abstractmethod
    def stage_data(self, features: List[str] = None) -> None:
        """
        Stage data to the final storage location.
        
        Args:
            features (List[str]): List of features to stage
        """
        raise NotImplementedError
    
    
class ZIPFeatureIO(FeatureIO):
    def __init__(self, base_dir: Path, staging_dir: Path, shard_size: int, require_features_exist: bool = False):
        """
        Initializes the ZIP feature I/O implementation.
        """
        super().__init__(base_dir, staging_dir, shard_size)
        
        self.current_shard = None
        self.shard_id = 0
        
    def _get_default_metadata(self):
        return {"key2shard": {}, "shards": [], "feature_count": 0}
    
    def list_keys(self):
        return list(self.metadata["key2shard"].keys())
    
    def exists(self, key):
        return key in self.metadata["key2shard"]
    
    def list_features(self, key):
        if key is not None:
            shard = self.metadata["key2shard"][key]
            with zipfile.ZipFile(self.base_dir / (shard + ".zip"), 'r') as zf:
                return ["_".join(name.split("_")[1:]).split(".")[0] for name in zf.namelist() if name.startswith(key)]
        else:   
            shard = self.metadata["shards"][0]
            with zipfile.ZipFile(self.base_dir / (shard + ".zip"), 'r') as zf:
                return list(set(["_".join(name.split("_")[1:]).split(".")[0] for name in zf.namelist()]))
    
    def save_shard(self):
        if self.current_shard is None:
            return
        # Create a ZIP archive for the current shard
        shard_base_path = self.staging_dir if self.staging_dir is not None else self.base_dir
        shard_base_path = shard_base_path / self.current_shard
        
        shard = self.current_shard
        with zipfile.ZipFile(self.base_dir / f"{shard}.zip", 'w') as zf:
            for file in shard_base_path.iterdir():
                zf.write(file, file.name)
            shutil.rmtree(shard_base_path)
        
        self.current_shard = None
    
    def save_feature(self, key, **features):
        shard_base_path = self.staging_dir if self.staging_dir is not None else self.base_dir
        
        if self.tmp_metadata["feature_count"] == 0:
            # Create a new shard
            self.current_shard = f"shard_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{os.getpid()}_{self.shard_id}"
            (shard_base_path / self.current_shard).mkdir(exist_ok=False)
            self.tmp_metadata["shards"].append(self.current_shard)
            self.shard_id += 1
            
        # Save features to the current shard
        for feature_name, feature_value in features.items():
            with open(shard_base_path / self.current_shard / f"{key}_{feature_name}.pt", 'wb') as f:
                buffer = io.BytesIO()
                torch.save(feature_value.cpu(), buffer)
                f.write(buffer.getvalue())
                
        self.tmp_metadata["key2shard"][key] = self.current_shard
        self.tmp_metadata["feature_count"] += 1
        
        if self.tmp_metadata["feature_count"] >= self.shard_size:
            # Save the current shard
            self.save_shard()
            self.save_metadata()
            
    def load_feature(self, key, feature_names):
        if self.tmp_metadata["feature_count"] > 0:
            self.save_metadata()
        
        if key not in self.metadata["key2shard"]:
            raise ValueError(f"Key {key} not found.")
        
        shard = self.metadata["key2shard"][key]
        features = {}
        
        if self.staged:
            # Load from the final storage location (individual extracted files)
            for feature_name in feature_names:
                with open(self.staging_dir / shard / f"{key}_{feature_name}.pt", 'rb') as f:
                    buffer = io.BytesIO(f.read())
                    features[feature_name] = torch.load(buffer, weights_only=True)
        else:
            # Load from the ZIP archive
            with zipfile.ZipFile(self.base_dir / (shard  + ".zip"), "r") as zf:
                for feature_name in feature_names:
                    with zf.open(f"{key}_{feature_name}.pt") as f:
                        buffer = io.BytesIO(f.read())
                        features[feature_name] = torch.load(buffer, weights_only=True)
                    
        return features
    
    def delete_feature(self, key):
        with self.thread_lock, open(self.metadata_file, 'r+') as f:
                portalocker.lock(f, portalocker.LockFlags.EXCLUSIVE)
                self.metadata = json.load(f)
                del self.metadata["key2shard"][key]
                self.metadata["feature_count"] -= 1
                f.seek(0)
                json.dump(self.metadata, f, indent=4)
                f.truncate()
                portalocker.lock(f, portalocker.LockFlags.UNBLOCK)
    
    def stage_data(self, features=None):
        if self.staging_dir is None:
            raise ValueError("Temporary directory for staging not provided.")
        if not self.staged:
            # Extract all features to the final storage location
            for shard in tqdm(self.metadata["shards"], desc="Staging data", total=len(self.metadata["shards"])):
                if (self.staging_dir / shard).exists():
                    continue
                with zipfile.ZipFile(self.base_dir / (shard + ".zip"), "r") as zf:
                    if features is None:
                        files = None
                    else:
                        files = [f for f in zf.namelist() if any([f.endswith(f"_{feature}.pt") for feature in features])]
                    zf.extractall(self.staging_dir / shard, files)
                        
            self.staged = True
        else:
            raise ValueError("Data has already been staged.")