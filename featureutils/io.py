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
    
    def __init__(self, base_dir: Path, staging_dir: Path, shard_size: int):
        """
        Initializes the feature I/O base class that handles basic feature storage operations.
        All feature I/O implementations should inherit from this class.
        
        Args:
            base_dir (str): Directory for storing feature shards.
            staging_dir (str): Directory for temporary files and staging area.
            shard_size (int): Maximum number of features per shard.
        """
        
        self.base_dir = base_dir
        self.staging_dir = staging_dir
        self.shard_size = shard_size
        self.staged = False
        
        self.thread_lock = threading.Lock()
        
        self.metadata_file = self.base_dir / "metadata.json"
        
        self._load_metadata()
        self.tmp_metadata = self._get_default_metadata()

    @abstractmethod
    def _get_default_metadata(self) -> Dict[str, Any]:
        """
        Returns the default metadata for feature management.
        
        Returns:
            Dict[str, Any]: Default metadata dictionary.
        """
        raise NotImplementedError

    def _load_metadata(self) -> None:
        """
        Loads metadata for feature management.
        
        Args:
            lock (bool): Whether to acquire an additional lock on the metadata file.
            file (Any): File object to load metadata from.
        """
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                portalocker.lock(f, portalocker.LockFlags.SHARED)
                self.metadata = json.load(f)
                portalocker.lock(f, portalocker.LockFlags.UNBLOCK)
        else:
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
    def stage_data(self) -> None:
        """
        Stage data to the final storage location.
        """
        raise NotImplementedError
    
    
class ZIPFeatureIO(FeatureIO):
    def __init__(self, base_dir: Path, staging_dir: Path, shard_size: int):
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
    
    def list_features(self, key):
        if key is not None:
            shard = self.metadata["key2shard"][key]
            with zipfile.ZipFile(self.base_dir / shard, 'r') as zf:
                return ["_".join(name.split("_")[1:]).split(".")[0] for name in zf.namelist() if name.startswith(key)]
        else:   
            shard = self.metadata["shards"][0]
            with zipfile.ZipFile(self.base_dir / shard, 'r') as zf:
                return list(set(["_".join(name.split("_")[1:]).split(".")[0] for name in zf.namelist()]))
    
    def save_shard(self):
        if self.staging_dir is None or self.current_shard is None:
            return
        # Move the current shard from the staging directory to the final storage location
        def move_shard():
            shutil.move(self.staging_dir / self.current_shard, self.base_dir / self.current_shard)
        
        move_thread = threading.Thread(target=move_shard)
        move_thread.start()
        move_thread.join()
        self.current_shard = None
    
    def save_feature(self, key, **features):
        
        if self.tmp_metadata["feature_count"] == 0:
            # Create a new shard
            self.current_shard = f"shard_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{os.getpid()}_{self.shard_id}.zip"
            self.shard_id += 1
            self.tmp_metadata["shards"].append(self.current_shard)
            
        # Save features to the current shard
        shard_base_path = self.staging_dir if self.staging_dir is not None else self.base_dir
        with self.thread_lock:
            with zipfile.ZipFile(shard_base_path / self.current_shard, 'a') as zf:
                for feature_name, feature_value in features.items():
                    buffer = io.BytesIO()
                    torch.save(feature_value.cpu(), buffer)
                    buffer.seek(0)
                    zf.writestr(f"{key}_{feature_name}.pt", buffer.read())
                
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
                with open(self.staging_dir / shard.replace(".zip", "") / f"{key}_{feature_name}.pt", 'rb') as f:
                    buffer = io.BytesIO(f.read())
                    features[feature_name] = torch.load(buffer, weights_only=True)
        else:
            # Load from the ZIP archive
            with zipfile.ZipFile(self.base_dir / shard, 'r') as zf:
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
    
    def stage_data(self):
        if self.staging_dir is None:
            raise ValueError("Temporary directory for staging not provided.")
        if not self.staged:
            # Extract all features to the final storage location
            for shard in tqdm(self.metadata["shards"], desc="Staging data"):
                with zipfile.ZipFile(self.base_dir / shard, 'r') as zf:
                    zf.extractall(self.staging_dir / shard.replace(".zip", ""))
            self.staged = True
        else:
            raise ValueError("Data has already been staged.")