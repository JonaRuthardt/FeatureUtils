# FeatureUtils: Simplified Feature Management for AI Model Training

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

**FeatureUtils** is a lightweight Python library that simplifies handling (precomputed) features in AI model training pipelines. It focuses on IO and storage management for feature datasets, making it easy to scale feature storage while maintaining flexibility and performance.

Whether you're dealing with large-scale machine learning workflows or managing features across multiple experiments, FeatureUtils provides intuitive APIs for saving, loading, staging, and querying features. It is designed to integrate seamlessly with PyTorch and supports efficient storage backends that limit inode usage when working with large-scale datasets.

---

## Key Features

- **Flexible Feature Management**: Save, load, and manage multiple features for training instances with minimal effort.
- **Efficient Storage**: Supports optimized storage backends that aggregate  shard-based data partitioning.
- **Staging for Speed**: Leverage fast storage (e.g., SSDs) to stage data for rapid random access.
- **PyTorch Integration**: Directly create PyTorch datasets from your features.
- **Scalable Metadata Handling**: Efficiently list, query, and manage metadata for large feature collections.

---

## Installation

To install the latest version of FeatureUtils from the repository, use pip to install the package and its dependencies:

```bash
pip install git+https://github.com/JonaRuthardt/featureutils.git
```

--- 

## Getting Started

Here’s a quick example to demonstrate how to use FeatureUtils in your AI training pipeline.

### Initialization
```python
from featureutils.core import FeatureUtils

# Initialize FeatureUtils with a ZIP storage backend
feature_utils = FeatureUtils(
    base_dir="path/to/storage", # directory where features will be saved to
    staging_dir="path/to/staging", # (optional) directory for data staging
    feature_num=1, # number of individual features for a given key
    shard_size=10000 # maximum size of individual shards
)
```

### Save Features
```python
import torch

# Save features for a specific key
key = "example_id"
feature_utils.save_feature(key, feature1=torch.rand(256), feature2=torch.rand(512))
```

### Load Features
```python
# Load features for a specific key
features = feature_utils.load_feature("example_id", feature_names=["feature1", "feature2"])
```

### Create a PyTorch Dataset
```python
# Get a Torch dataset for selected keys and features
keys = ["example_id1", "example_id2"] # (optional) specifications of keys to include
features = ["feature1", "feature2"] # (optional) specification of features to include
dataset = feature_utils.get_dataset(keys=keys, features=features)
```

### Query and Manage Features
```python
# List all keys
keys = feature_utils.list_keys()

# Check if a key exists
exists = feature_utils.exists("example_id")

# Delete a feature
feature_utils.delete_feature("example_id")
```

## Citation

If you use FeatureUtils in your research, please consider citing it with the following BibTeX entry:
```bibtex
@misc{featureutils,
  author = Jona Ruthardt,
  title = FeatureUtils: Simplified Feature Management for AI Training,
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/featureutils}},
}
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact and Contributing
For questions or support, open an issue on the GitHub repository. Contributions and feedback are always welcome. 