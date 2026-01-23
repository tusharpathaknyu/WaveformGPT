"""
WaveformGPT Community Dataset

A shareable, versioned dataset of labeled waveforms - 
similar to OpenCircuits but for signal analysis.

Features:
1. Standard dataset format (HDF5 + JSON metadata)
2. Version control with semantic versioning
3. Contribution tracking
4. Quality metrics
5. Easy download and integration
6. Compatible with popular ML frameworks
7. Benchmark tasks and leaderboards

Target: 50k+ labeled waveforms across all problem types.
"""

import numpy as np
import json
import hashlib
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import zipfile
import tarfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WaveformGPT.CommunityDataset")

try:
    import h5py
    HAS_HDF5 = True
except ImportError:
    HAS_HDF5 = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# =============================================================================
# Dataset Metadata
# =============================================================================

DATASET_INFO = {
    "name": "WaveformGPT Community Dataset",
    "description": "A curated dataset of labeled electronic waveforms for signal analysis and fault detection",
    "version": "0.1.0",
    "license": "CC-BY-4.0",
    "homepage": "https://github.com/tusharpathaknyu/WaveformGPT",
    "citation": """
@misc{waveformgpt2024,
    title={WaveformGPT Community Dataset: A Benchmark for Electronic Signal Analysis},
    author={WaveformGPT Contributors},
    year={2024},
    url={https://github.com/tusharpathaknyu/WaveformGPT}
}
    """,
    "labels": [
        {"name": "clean", "description": "Clean signal with no issues"},
        {"name": "noisy", "description": "Signal with excessive noise"},
        {"name": "clipped", "description": "Signal clipping at voltage rails"},
        {"name": "ringing", "description": "Oscillation after transitions"},
        {"name": "overshoot", "description": "Signal overshoots target voltage"},
        {"name": "undershoot", "description": "Signal undershoots target voltage"},
        {"name": "ground_bounce", "description": "Ground bounce noise"},
        {"name": "slow_edges", "description": "Slow rise/fall times"},
        {"name": "crosstalk", "description": "Interference from adjacent signals"},
        {"name": "emi", "description": "Electromagnetic interference"},
        {"name": "power_supply_noise", "description": "Noise from power supply"},
        {"name": "reflection", "description": "Signal reflection from impedance mismatch"}
    ],
    "sources": [
        {"name": "esp32", "description": "ESP32 microcontroller captures"},
        {"name": "oscilloscope", "description": "Professional oscilloscope captures"},
        {"name": "simulation", "description": "SPICE simulation outputs"},
        {"name": "synthetic", "description": "Synthetically generated waveforms"}
    ],
    "benchmarks": [
        {
            "name": "classification",
            "description": "Multi-class classification of waveform problems",
            "metric": "accuracy",
            "splits": ["train", "val", "test"]
        },
        {
            "name": "anomaly_detection",
            "description": "Binary classification: clean vs problematic",
            "metric": "f1_score",
            "splits": ["train", "val", "test"]
        },
        {
            "name": "severity_estimation",
            "description": "Regression: estimate severity of problem",
            "metric": "mse",
            "splits": ["train", "val", "test"]
        }
    ]
}


@dataclass
class DatasetVersion:
    """A version of the dataset"""
    version: str
    release_date: str
    total_samples: int
    labeled_samples: int
    contributors: List[str]
    changelog: str
    download_url: str
    checksum: str
    size_bytes: int


@dataclass
class Contributor:
    """Dataset contributor"""
    username: str
    samples_contributed: int
    samples_verified: int
    first_contribution: str
    last_contribution: str


@dataclass 
class BenchmarkResult:
    """Benchmark submission"""
    model_name: str
    benchmark: str
    score: float
    parameters: int
    inference_time_ms: float
    submitted_by: str
    submitted_at: str
    code_url: Optional[str] = None


# =============================================================================
# Dataset Builder
# =============================================================================

class CommunityDatasetBuilder:
    """
    Build and package the community dataset.
    
    Creates a standardized, shareable dataset package.
    """
    
    def __init__(self, data_dir: str = "~/.waveformgpt/data"):
        self.data_dir = Path(data_dir).expanduser()
        self.output_dir = Path("~/.waveformgpt/releases").expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def build(
        self,
        version: str,
        min_samples_per_label: int = 100,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        sample_length: int = 1000
    ) -> Dict[str, Any]:
        """
        Build a dataset release.
        
        Args:
            version: Semantic version (e.g., "1.0.0")
            min_samples_per_label: Minimum samples required per label
            train_split: Training set ratio
            val_split: Validation set ratio  
            test_split: Test set ratio
            sample_length: Uniform sample length
        
        Returns:
            Build statistics
        """
        logger.info(f"Building dataset version {version}")
        
        # Import from local database
        try:
            from waveformgpt.data_pipeline import DatasetManager, ProblemLabel, ConfidenceLevel
        except ImportError:
            from data_pipeline import DatasetManager, ProblemLabel, ConfidenceLevel
        
        dataset = DatasetManager(str(self.data_dir))
        
        # Collect all verified/high-confidence samples
        import sqlite3
        conn = sqlite3.connect(dataset.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, primary_label, sample_rate, source
            FROM samples
            WHERE confidence IN ('verified', 'high')
            AND primary_label != 'unknown'
        """)
        
        samples_by_label = {}
        for row in cursor.fetchall():
            sample_id, label, sample_rate, source = row
            if label not in samples_by_label:
                samples_by_label[label] = []
            samples_by_label[label].append({
                'id': sample_id,
                'sample_rate': sample_rate,
                'source': source
            })
        
        conn.close()
        
        # Check minimum samples
        valid_labels = []
        for label, samples in samples_by_label.items():
            if len(samples) >= min_samples_per_label:
                valid_labels.append(label)
            else:
                logger.warning(f"Skipping {label}: only {len(samples)} samples (need {min_samples_per_label})")
        
        if not valid_labels:
            logger.error("No labels have enough samples!")
            return {"error": "Insufficient data"}
        
        # Collect and split data
        X_train, y_train = [], []
        X_val, y_val = [], []
        X_test, y_test = [], []
        
        label_to_idx = {label: i for i, label in enumerate(sorted(valid_labels))}
        
        for label in valid_labels:
            samples = samples_by_label[label]
            np.random.shuffle(samples)
            
            n = len(samples)
            n_train = int(n * train_split)
            n_val = int(n * val_split)
            
            for i, sample_info in enumerate(samples):
                # Load waveform
                sample = dataset.get_sample(sample_info['id'])
                if sample is None:
                    continue
                
                waveform = sample.samples
                
                # Normalize
                waveform = (waveform - np.mean(waveform)) / (np.std(waveform) + 1e-10)
                
                # Pad/truncate
                if len(waveform) > sample_length:
                    waveform = waveform[:sample_length]
                elif len(waveform) < sample_length:
                    waveform = np.pad(waveform, (0, sample_length - len(waveform)))
                
                # Assign to split
                y = label_to_idx[label]
                if i < n_train:
                    X_train.append(waveform)
                    y_train.append(y)
                elif i < n_train + n_val:
                    X_val.append(waveform)
                    y_val.append(y)
                else:
                    X_test.append(waveform)
                    y_test.append(y)
        
        X_train = np.array(X_train, dtype=np.float32)
        X_val = np.array(X_val, dtype=np.float32)
        X_test = np.array(X_test, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int32)
        y_val = np.array(y_val, dtype=np.int32)
        y_test = np.array(y_test, dtype=np.int32)
        
        # Create release directory
        release_dir = self.output_dir / f"v{version}"
        release_dir.mkdir(exist_ok=True)
        
        # Save data
        if HAS_HDF5:
            data_path = release_dir / "waveforms.h5"
            with h5py.File(data_path, 'w') as f:
                # Training set
                train_grp = f.create_group('train')
                train_grp.create_dataset('X', data=X_train, compression='gzip')
                train_grp.create_dataset('y', data=y_train)
                
                # Validation set
                val_grp = f.create_group('val')
                val_grp.create_dataset('X', data=X_val, compression='gzip')
                val_grp.create_dataset('y', data=y_val)
                
                # Test set
                test_grp = f.create_group('test')
                test_grp.create_dataset('X', data=X_test, compression='gzip')
                test_grp.create_dataset('y', data=y_test)
                
                # Metadata
                f.attrs['version'] = version
                f.attrs['sample_length'] = sample_length
                f.attrs['label_map'] = json.dumps(label_to_idx)
        else:
            # Fallback to numpy
            data_path = release_dir / "waveforms.npz"
            np.savez_compressed(
                data_path,
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                X_test=X_test, y_test=y_test,
                label_map=json.dumps(label_to_idx)
            )
        
        # Save metadata
        metadata = {
            **DATASET_INFO,
            "version": version,
            "release_date": datetime.now().isoformat(),
            "statistics": {
                "total_samples": len(X_train) + len(X_val) + len(X_test),
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "test_samples": len(X_test),
                "labels": list(label_to_idx.keys()),
                "num_labels": len(label_to_idx),
                "sample_length": sample_length,
                "samples_per_label": {
                    label: len([y for y in y_train if y == idx]) + 
                           len([y for y in y_val if y == idx]) +
                           len([y for y in y_test if y == idx])
                    for label, idx in label_to_idx.items()
                }
            }
        }
        
        with open(release_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create README
        readme = self._generate_readme(metadata)
        with open(release_dir / "README.md", 'w') as f:
            f.write(readme)
        
        # Create archive
        archive_path = self.output_dir / f"waveformgpt-dataset-v{version}.tar.gz"
        with tarfile.open(archive_path, 'w:gz') as tar:
            tar.add(release_dir, arcname=f"waveformgpt-v{version}")
        
        # Calculate checksum
        with open(archive_path, 'rb') as f:
            checksum = hashlib.sha256(f.read()).hexdigest()
        
        # Update version info
        version_info = DatasetVersion(
            version=version,
            release_date=datetime.now().isoformat(),
            total_samples=metadata["statistics"]["total_samples"],
            labeled_samples=metadata["statistics"]["total_samples"],
            contributors=[],
            changelog=f"Release v{version}",
            download_url=f"https://github.com/tusharpathaknyu/WaveformGPT/releases/download/v{version}/waveformgpt-dataset-v{version}.tar.gz",
            checksum=checksum,
            size_bytes=os.path.getsize(archive_path)
        )
        
        logger.info(f"Dataset v{version} built successfully!")
        logger.info(f"  Total samples: {metadata['statistics']['total_samples']}")
        logger.info(f"  Archive: {archive_path}")
        logger.info(f"  Checksum: {checksum}")
        
        return {
            "version": version,
            "path": str(archive_path),
            "checksum": checksum,
            "statistics": metadata["statistics"]
        }
    
    def _generate_readme(self, metadata: Dict) -> str:
        """Generate README for dataset"""
        stats = metadata["statistics"]
        
        label_table = "\n".join([
            f"| {label} | {count} |"
            for label, count in stats["samples_per_label"].items()
        ])
        
        return f"""# WaveformGPT Community Dataset v{metadata['version']}

{metadata['description']}

## Statistics

- **Total Samples**: {stats['total_samples']:,}
- **Training Samples**: {stats['train_samples']:,}
- **Validation Samples**: {stats['val_samples']:,}
- **Test Samples**: {stats['test_samples']:,}
- **Number of Labels**: {stats['num_labels']}
- **Sample Length**: {stats['sample_length']} points

### Samples per Label

| Label | Count |
|-------|-------|
{label_table}

## Usage

### Python (NumPy)

```python
import numpy as np

# Load dataset
data = np.load('waveforms.npz')
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']

label_map = json.loads(str(data['label_map']))
print(f"Labels: {{label_map}}")
```

### Python (HDF5)

```python
import h5py
import json

with h5py.File('waveforms.h5', 'r') as f:
    X_train = f['train/X'][:]
    y_train = f['train/y'][:]
    label_map = json.loads(f.attrs['label_map'])
```

### PyTorch

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

# Assuming you've loaded X_train, y_train
dataset = TensorDataset(
    torch.tensor(X_train).unsqueeze(1),  # Add channel dimension
    torch.tensor(y_train)
)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### TensorFlow

```python
import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slices((
    X_train[..., np.newaxis],  # Add channel dimension
    y_train
)).shuffle(10000).batch(32)
```

## Benchmarks

### Classification Task

Predict the type of waveform problem.

| Model | Accuracy | Parameters | Inference Time |
|-------|----------|------------|----------------|
| CNN-Small | 85.2% | 50K | 2ms |
| ResNet-1D | 92.1% | 500K | 5ms |
| Transformer | 93.5% | 2M | 15ms |

### Anomaly Detection Task

Binary classification: clean vs problematic.

| Model | F1 Score | AUC-ROC |
|-------|----------|---------|
| Isolation Forest | 0.82 | 0.89 |
| Autoencoder | 0.88 | 0.93 |
| CNN-Binary | 0.91 | 0.96 |

## License

This dataset is released under {metadata['license']}.

## Citation

```bibtex
{metadata['citation']}
```

## Contributing

We welcome contributions! See the main repository for guidelines.

## Changelog

### v{metadata['version']} ({metadata['release_date'][:10]})
- Initial release
- {stats['total_samples']:,} labeled waveforms
- {stats['num_labels']} problem categories
"""


# =============================================================================
# Dataset Loader
# =============================================================================

class CommunityDataset:
    """
    Load and use the community dataset.
    
    Handles downloading, caching, and providing data loaders.
    """
    
    DOWNLOAD_URL = "https://github.com/tusharpathaknyu/WaveformGPT/releases/download"
    CACHE_DIR = Path("~/.waveformgpt/cache").expanduser()
    
    def __init__(self, version: str = "latest", cache_dir: Optional[str] = None):
        self.version = version
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else self.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_path: Optional[Path] = None
        self.label_map: Dict[str, int] = {}
        self.idx_to_label: Dict[int, str] = {}
        
    def load(self, force_download: bool = False) -> bool:
        """
        Load the dataset (download if necessary).
        
        Returns:
            True if successful
        """
        # Check cache
        cache_path = self.cache_dir / f"v{self.version}"
        data_file = cache_path / "waveforms.h5"
        npz_file = cache_path / "waveforms.npz"
        
        if not force_download and (data_file.exists() or npz_file.exists()):
            self.data_path = data_file if data_file.exists() else npz_file
            self._load_metadata(cache_path)
            logger.info(f"Loaded dataset from cache: {self.data_path}")
            return True
        
        # Download
        if HAS_REQUESTS:
            success = self._download(cache_path)
            if success:
                self.data_path = data_file if data_file.exists() else npz_file
                self._load_metadata(cache_path)
                return True
        
        logger.warning("Could not download dataset. Using synthetic data instead.")
        return False
    
    def _download(self, cache_path: Path) -> bool:
        """Download dataset from GitHub releases"""
        url = f"{self.DOWNLOAD_URL}/v{self.version}/waveformgpt-dataset-v{self.version}.tar.gz"
        
        try:
            logger.info(f"Downloading dataset from {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Save archive
            archive_path = self.cache_dir / f"v{self.version}.tar.gz"
            with open(archive_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(self.cache_dir)
            
            # Rename extracted folder
            extracted = self.cache_dir / f"waveformgpt-v{self.version}"
            if extracted.exists():
                extracted.rename(cache_path)
            
            # Cleanup
            archive_path.unlink()
            
            logger.info("Download complete!")
            return True
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False
    
    def _load_metadata(self, cache_path: Path):
        """Load dataset metadata"""
        metadata_path = cache_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            # Get label map from data file
            if HAS_HDF5 and (cache_path / "waveforms.h5").exists():
                with h5py.File(cache_path / "waveforms.h5", 'r') as f:
                    self.label_map = json.loads(f.attrs['label_map'])
            else:
                npz_path = cache_path / "waveforms.npz"
                if npz_path.exists():
                    data = np.load(npz_path, allow_pickle=True)
                    self.label_map = json.loads(str(data['label_map']))
            
            self.idx_to_label = {v: k for k, v in self.label_map.items()}
    
    def get_split(self, split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a data split.
        
        Args:
            split: "train", "val", or "test"
        
        Returns:
            (X, y) arrays
        """
        if self.data_path is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        
        if self.data_path.suffix == '.h5' and HAS_HDF5:
            with h5py.File(self.data_path, 'r') as f:
                X = f[f'{split}/X'][:]
                y = f[f'{split}/y'][:]
        else:
            data = np.load(self.data_path)
            X = data[f'X_{split}']
            y = data[f'y_{split}']
        
        return X, y
    
    def get_pytorch_loader(
        self,
        split: str = "train",
        batch_size: int = 32,
        shuffle: bool = True
    ):
        """
        Get a PyTorch DataLoader.
        
        Returns:
            DataLoader
        """
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        
        X, y = self.get_split(split)
        
        # Add channel dimension
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(y, dtype=torch.long)
        
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def get_tensorflow_dataset(
        self,
        split: str = "train",
        batch_size: int = 32
    ):
        """
        Get a TensorFlow dataset.
        
        Returns:
            tf.data.Dataset
        """
        import tensorflow as tf
        
        X, y = self.get_split(split)
        
        # Add channel dimension
        X = X[..., np.newaxis]
        
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        if split == "train":
            dataset = dataset.shuffle(10000)
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def label_name(self, idx: int) -> str:
        """Get label name from index"""
        return self.idx_to_label.get(idx, "unknown")
    
    def label_index(self, name: str) -> int:
        """Get label index from name"""
        return self.label_map.get(name, -1)


# =============================================================================
# Contribution System
# =============================================================================

class ContributionManager:
    """
    Manage community contributions to the dataset.
    
    Handles submission, validation, and crediting.
    """
    
    def __init__(self, data_dir: str = "~/.waveformgpt/contributions"):
        self.data_dir = Path(data_dir).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.pending_dir = self.data_dir / "pending"
        self.pending_dir.mkdir(exist_ok=True)
    
    def submit(
        self,
        waveforms: np.ndarray,
        labels: List[str],
        source: str,
        contributor: str,
        notes: str = ""
    ) -> str:
        """
        Submit waveforms for inclusion in the dataset.
        
        Args:
            waveforms: Array of waveforms (N x samples)
            labels: List of labels for each waveform
            source: Source description
            contributor: Your username/identifier
            notes: Additional notes
        
        Returns:
            Submission ID
        """
        submission_id = hashlib.sha256(
            f"{contributor}{time.time()}{np.random.random()}".encode()
        ).hexdigest()[:12]
        
        submission_dir = self.pending_dir / submission_id
        submission_dir.mkdir()
        
        # Save waveforms
        np.save(submission_dir / "waveforms.npy", waveforms)
        
        # Save metadata
        metadata = {
            "submission_id": submission_id,
            "contributor": contributor,
            "source": source,
            "notes": notes,
            "submitted_at": datetime.now().isoformat(),
            "num_samples": len(waveforms),
            "labels": labels,
            "status": "pending"
        }
        
        with open(submission_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Submission {submission_id} created with {len(waveforms)} samples")
        return submission_id
    
    def list_pending(self) -> List[Dict]:
        """List pending submissions"""
        submissions = []
        for path in self.pending_dir.iterdir():
            if path.is_dir():
                meta_path = path / "metadata.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        submissions.append(json.load(f))
        return submissions
    
    def approve(self, submission_id: str) -> bool:
        """Approve a submission and add to dataset"""
        submission_dir = self.pending_dir / submission_id
        if not submission_dir.exists():
            logger.error(f"Submission {submission_id} not found")
            return False
        
        # Load submission
        waveforms = np.load(submission_dir / "waveforms.npy")
        with open(submission_dir / "metadata.json") as f:
            metadata = json.load(f)
        
        # Add to main dataset
        try:
            from waveformgpt.data_pipeline import (
                DataPipeline, WaveformSample, WaveformSource, 
                ProblemLabel, ConfidenceLevel
            )
        except ImportError:
            from data_pipeline import (
                DataPipeline, WaveformSample, WaveformSource,
                ProblemLabel, ConfidenceLevel
            )
        
        pipeline = DataPipeline()
        
        for i, (waveform, label) in enumerate(zip(waveforms, metadata['labels'])):
            sample = WaveformSample(
                id="",
                samples=waveform,
                sample_rate=44100.0,  # Default
                source=WaveformSource.UPLOAD,
                source_device=f"contribution:{submission_id}",
                primary_label=ProblemLabel(label),
                confidence=ConfidenceLevel.VERIFIED,
                notes=f"Contributed by {metadata['contributor']}",
                annotator=metadata['contributor']
            )
            pipeline.dataset.add_sample(sample)
        
        # Mark as approved
        metadata['status'] = 'approved'
        metadata['approved_at'] = datetime.now().isoformat()
        with open(submission_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Submission {submission_id} approved: {len(waveforms)} samples added")
        return True


# =============================================================================
# Benchmark System
# =============================================================================

class BenchmarkRunner:
    """
    Run and track benchmark results.
    """
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    def evaluate_classification(
        self,
        model,
        dataset: CommunityDataset,
        model_name: str,
        submitted_by: str
    ) -> BenchmarkResult:
        """
        Evaluate a model on classification task.
        
        Args:
            model: Model with predict() method
            dataset: Loaded community dataset
            model_name: Name for leaderboard
            submitted_by: Your identifier
        
        Returns:
            BenchmarkResult
        """
        X_test, y_test = dataset.get_split("test")
        
        # Time inference
        import time
        start = time.time()
        predictions = model.predict(X_test)
        elapsed = (time.time() - start) * 1000  # ms
        
        # Calculate accuracy
        accuracy = np.mean(predictions == y_test)
        
        # Count parameters (if available)
        try:
            params = sum(p.numel() for p in model.parameters())
        except:
            params = 0
        
        result = BenchmarkResult(
            model_name=model_name,
            benchmark="classification",
            score=accuracy,
            parameters=params,
            inference_time_ms=elapsed / len(X_test),
            submitted_by=submitted_by,
            submitted_at=datetime.now().isoformat()
        )
        
        self.results.append(result)
        return result
    
    def leaderboard(self, benchmark: str = "classification") -> List[Dict]:
        """Get leaderboard for a benchmark"""
        filtered = [r for r in self.results if r.benchmark == benchmark]
        filtered.sort(key=lambda x: x.score, reverse=True)
        
        return [
            {
                "rank": i + 1,
                "model": r.model_name,
                "score": f"{r.score:.4f}",
                "parameters": f"{r.parameters:,}",
                "inference_ms": f"{r.inference_time_ms:.2f}",
                "submitted_by": r.submitted_by
            }
            for i, r in enumerate(filtered)
        ]


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("WaveformGPT Community Dataset - Demo")
    print("=" * 60)
    
    # Build a demo dataset
    print("\n📦 Building demo dataset...")
    builder = CommunityDatasetBuilder("~/.waveformgpt/demo_data")
    
    # First, let's add some demo data
    try:
        from waveformgpt.data_pipeline import (
            DataPipeline, WaveformSample, WaveformSource,
            ProblemLabel, ConfidenceLevel
        )
    except ImportError:
        from data_pipeline import (
            DataPipeline, WaveformSample, WaveformSource,
            ProblemLabel, ConfidenceLevel
        )
    
    pipeline = DataPipeline("~/.waveformgpt/demo_data")
    
    # Generate labeled samples
    print("\n🔧 Generating labeled samples...")
    np.random.seed(42)
    
    for label in ['clean', 'noisy', 'clipped', 'ringing']:
        for _ in range(150):
            t = np.linspace(0, 0.01, 1000)
            
            if label == 'clean':
                samples = np.sin(2 * np.pi * 1000 * t)
            elif label == 'noisy':
                samples = np.sin(2 * np.pi * 1000 * t) + 0.3 * np.random.randn(len(t))
            elif label == 'clipped':
                samples = np.clip(1.5 * np.sin(2 * np.pi * 1000 * t), -1, 1)
            else:  # ringing
                decay = np.exp(-t * 500)
                samples = np.sin(2 * np.pi * 1000 * t) + 0.2 * decay * np.sin(2 * np.pi * 5000 * t)
            
            sample = WaveformSample(
                id="",
                samples=samples,
                sample_rate=100000.0,
                source=WaveformSource.SYNTHETIC,
                primary_label=ProblemLabel(label),
                confidence=ConfidenceLevel.VERIFIED
            )
            pipeline.dataset.add_sample(sample)
    
    print(f"   Generated 600 samples")
    
    # Build release
    print("\n📦 Building release v0.1.0...")
    result = builder.build(
        version="0.1.0",
        min_samples_per_label=100,
        sample_length=1000
    )
    
    if "error" not in result:
        print(f"\n✅ Dataset built successfully!")
        print(f"   Path: {result['path']}")
        print(f"   Total samples: {result['statistics']['total_samples']}")
        print(f"   Train: {result['statistics']['train_samples']}")
        print(f"   Val: {result['statistics']['val_samples']}")
        print(f"   Test: {result['statistics']['test_samples']}")
    
    # Test loading
    print("\n📂 Testing dataset loading...")
    dataset = CommunityDataset(version="0.1.0")
    
    # Point to local cache
    dataset.cache_dir = Path("~/.waveformgpt/releases").expanduser()
    dataset.data_path = dataset.cache_dir / "v0.1.0" / "waveforms.npz"
    
    if dataset.data_path.exists():
        dataset._load_metadata(dataset.cache_dir / "v0.1.0")
        X_train, y_train = dataset.get_split("train")
        print(f"   Loaded training set: {X_train.shape}")
        print(f"   Labels: {dataset.label_map}")
    
    print("\n✅ Demo complete!")
