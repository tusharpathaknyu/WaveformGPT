"""
Waveform CNN Classifier

Lightweight CNN for classifying waveform problems:
- Overshoot
- Ringing  
- Noise
- Clipping
- Slow rise
- Distortion
- DC offset
- Normal (healthy)

Can be trained on synthetic data (generated from SPICE simulations)
or real oscilloscope captures.

No GPU required - runs on CPU with PyTorch or even numpy-only.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import json
import os

# Try to import PyTorch (preferred)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not found. Using numpy-only classifier.")


class WaveformClass(Enum):
    """Waveform problem classes"""
    NORMAL = 0
    OVERSHOOT = 1
    RINGING = 2
    NOISE = 3
    CLIPPING = 4
    SLOW_RISE = 5
    DISTORTION = 6
    DC_OFFSET = 7


@dataclass
class ClassificationResult:
    """Result of waveform classification"""
    predicted_class: WaveformClass
    confidence: float
    all_probabilities: Dict[str, float]
    features_used: Optional[np.ndarray] = None


# =============================================================================
# PyTorch CNN Model
# =============================================================================

if HAS_TORCH:
    
    class WaveformCNN(nn.Module):
        """
        1D CNN for waveform classification.
        
        Architecture:
        - 3 conv layers with batch norm and pooling
        - 2 fully connected layers
        - Softmax output
        
        Input: (batch, 1, 1000) - 1000 sample points
        Output: (batch, 8) - 8 classes
        """
        
        def __init__(self, input_length: int = 1000, num_classes: int = 8):
            super().__init__()
            
            self.input_length = input_length
            self.num_classes = num_classes
            
            # Convolutional layers
            self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
            self.bn1 = nn.BatchNorm1d(32)
            self.pool1 = nn.MaxPool1d(4)
            
            self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
            self.bn2 = nn.BatchNorm1d(64)
            self.pool2 = nn.MaxPool1d(4)
            
            self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm1d(128)
            self.pool3 = nn.AdaptiveAvgPool1d(1)  # Global average pooling
            
            # Fully connected layers
            self.fc1 = nn.Linear(128, 64)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(64, num_classes)
        
        def forward(self, x):
            # Conv block 1
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            
            # Conv block 2
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            
            # Conv block 3
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            
            # Flatten
            x = x.view(x.size(0), -1)
            
            # FC layers
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            
            return x
        
        def predict(self, waveform: np.ndarray) -> ClassificationResult:
            """
            Classify a single waveform.
            
            Args:
                waveform: 1D numpy array of samples
            
            Returns:
                ClassificationResult
            """
            self.eval()
            
            # Preprocess
            x = self._preprocess(waveform)
            
            with torch.no_grad():
                logits = self.forward(x)
                probs = F.softmax(logits, dim=1).squeeze().numpy()
            
            predicted_idx = int(np.argmax(probs))
            confidence = float(probs[predicted_idx])
            
            all_probs = {
                WaveformClass(i).name: float(probs[i])
                for i in range(len(probs))
            }
            
            return ClassificationResult(
                predicted_class=WaveformClass(predicted_idx),
                confidence=confidence,
                all_probabilities=all_probs
            )
        
        def _preprocess(self, waveform: np.ndarray) -> torch.Tensor:
            """Normalize and reshape waveform for CNN input"""
            # Resample to fixed length
            if len(waveform) != self.input_length:
                x_old = np.linspace(0, 1, len(waveform))
                x_new = np.linspace(0, 1, self.input_length)
                waveform = np.interp(x_new, x_old, waveform)
            
            # Normalize to [-1, 1]
            waveform = waveform.astype(np.float32)
            vmin, vmax = waveform.min(), waveform.max()
            if vmax - vmin > 0:
                waveform = 2 * (waveform - vmin) / (vmax - vmin) - 1
            
            # Reshape: (1, 1, length)
            x = torch.tensor(waveform).unsqueeze(0).unsqueeze(0)
            
            return x


    class WaveformDataset(Dataset):
        """Dataset for training waveform classifier"""
        
        def __init__(self, waveforms: List[np.ndarray], labels: List[int], 
                     input_length: int = 1000):
            self.waveforms = waveforms
            self.labels = labels
            self.input_length = input_length
        
        def __len__(self):
            return len(self.waveforms)
        
        def __getitem__(self, idx):
            waveform = self.waveforms[idx]
            label = self.labels[idx]
            
            # Resample
            if len(waveform) != self.input_length:
                x_old = np.linspace(0, 1, len(waveform))
                x_new = np.linspace(0, 1, self.input_length)
                waveform = np.interp(x_new, x_old, waveform)
            
            # Normalize
            waveform = waveform.astype(np.float32)
            vmin, vmax = waveform.min(), waveform.max()
            if vmax - vmin > 0:
                waveform = 2 * (waveform - vmin) / (vmax - vmin) - 1
            
            x = torch.tensor(waveform).unsqueeze(0)  # (1, length)
            y = torch.tensor(label, dtype=torch.long)
            
            return x, y


# =============================================================================
# Numpy-only Classifier (No PyTorch required)
# =============================================================================

class NumpyWaveformClassifier:
    """
    Simple feature-based classifier using numpy only.
    Uses hand-crafted features + decision tree logic.
    
    Not as accurate as CNN, but works without any ML libraries.
    """
    
    def __init__(self):
        self.thresholds = {
            'overshoot': 15.0,      # percent
            'ringing_min_freq': 1000,  # Hz
            'noise_ratio': 0.1,     # relative to Vpp
            'clipping_threshold': 0.02,  # % of samples at min/max
            'slow_rise_factor': 3.0,    # times expected
            'thd_threshold': 5.0,   # percent
            'dc_offset_ratio': 0.2, # relative to Vpp
        }
    
    def extract_features(self, waveform: np.ndarray, sample_rate: float = 1e6) -> Dict[str, float]:
        """Extract features from waveform"""
        n = len(waveform)
        
        vmax = np.max(waveform)
        vmin = np.min(waveform)
        vpp = vmax - vmin
        
        if vpp == 0:
            vpp = 1e-10  # Avoid division by zero
        
        # Basic stats
        features = {
            'vpp': vpp,
            'vmax': vmax,
            'vmin': vmin,
            'mean': np.mean(waveform),
            'std': np.std(waveform),
        }
        
        # Overshoot (for step response)
        initial = np.mean(waveform[:max(1, n//20)])
        final = np.mean(waveform[-max(1, n//4):])
        step_size = abs(final - initial)
        
        if step_size > 0.1 * vpp:  # Looks like a step
            peak = np.max(waveform) if final > initial else np.min(waveform)
            overshoot = abs(peak - final) / step_size * 100
            features['overshoot'] = min(overshoot, 100)
        else:
            features['overshoot'] = 0
        
        # Ringing detection
        derivative = np.diff(waveform)
        zero_crossings = np.sum(np.diff(np.signbit(derivative)))
        features['zero_crossings'] = zero_crossings
        
        # Noise estimation (high-frequency content)
        diff2 = np.diff(waveform, n=2)
        features['noise_ratio'] = np.std(diff2) / vpp
        
        # Clipping detection
        at_max = np.sum(np.abs(waveform - vmax) < 0.01 * vpp)
        at_min = np.sum(np.abs(waveform - vmin) < 0.01 * vpp)
        features['clipping_ratio'] = (at_max + at_min) / n
        
        # Rise time (normalized)
        v10 = vmin + 0.1 * vpp
        v90 = vmin + 0.9 * vpp
        above_10 = np.where(waveform > v10)[0]
        above_90 = np.where(waveform > v90)[0]
        if len(above_10) > 0 and len(above_90) > 0:
            t10 = above_10[0]
            t90_candidates = above_90[above_90 > t10]
            if len(t90_candidates) > 0:
                features['rise_time_samples'] = t90_candidates[0] - t10
            else:
                features['rise_time_samples'] = 0
        else:
            features['rise_time_samples'] = 0
        
        # THD estimation
        spectrum = np.abs(np.fft.rfft(waveform))
        if len(spectrum) > 5:
            spectrum[0] = 0  # Remove DC
            fund_idx = np.argmax(spectrum)
            if fund_idx > 0:
                fundamental = spectrum[fund_idx]
                harmonic_power = sum(
                    spectrum[fund_idx * h]**2 
                    for h in range(2, 6) 
                    if fund_idx * h < len(spectrum)
                )
                features['thd'] = np.sqrt(harmonic_power) / fundamental * 100 if fundamental > 0 else 0
            else:
                features['thd'] = 0
        else:
            features['thd'] = 0
        
        # DC offset
        features['dc_offset_ratio'] = abs(features['mean'] - (vmax + vmin) / 2) / vpp
        
        return features
    
    def classify(self, waveform: np.ndarray) -> ClassificationResult:
        """Classify waveform using feature thresholds"""
        features = self.extract_features(waveform)
        
        # Decision logic (priority order)
        probs = {c.name: 0.0 for c in WaveformClass}
        
        # Clipping
        if features['clipping_ratio'] > self.thresholds['clipping_threshold']:
            probs['CLIPPING'] = min(features['clipping_ratio'] * 10, 1.0)
        
        # Noise
        if features['noise_ratio'] > self.thresholds['noise_ratio']:
            probs['NOISE'] = min(features['noise_ratio'] * 5, 1.0)
        
        # Overshoot
        if features['overshoot'] > self.thresholds['overshoot']:
            probs['OVERSHOOT'] = min(features['overshoot'] / 50, 1.0)
        
        # Ringing (many zero crossings in derivative)
        if features['zero_crossings'] > 10:
            probs['RINGING'] = min(features['zero_crossings'] / 50, 1.0)
        
        # Slow rise
        if features['rise_time_samples'] > len(waveform) * 0.3:
            probs['SLOW_RISE'] = 0.7
        
        # THD (distortion)
        if features['thd'] > self.thresholds['thd_threshold']:
            probs['DISTORTION'] = min(features['thd'] / 20, 1.0)
        
        # DC offset
        if features['dc_offset_ratio'] > self.thresholds['dc_offset_ratio']:
            probs['DC_OFFSET'] = min(features['dc_offset_ratio'] * 2, 1.0)
        
        # If nothing detected, it's normal
        max_prob = max(probs.values())
        if max_prob < 0.3:
            probs['NORMAL'] = 0.8
        
        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        else:
            probs['NORMAL'] = 1.0
        
        # Get prediction
        predicted_name = max(probs, key=probs.get)
        predicted_class = WaveformClass[predicted_name]
        confidence = probs[predicted_name]
        
        return ClassificationResult(
            predicted_class=predicted_class,
            confidence=confidence,
            all_probabilities=probs,
            features_used=np.array(list(features.values()))
        )


# =============================================================================
# Synthetic Data Generator
# =============================================================================

class SyntheticDataGenerator:
    """
    Generate synthetic waveforms for training the CNN.
    Creates labeled examples of each problem type.
    """
    
    def __init__(self, sample_rate: float = 1e6, duration: float = 100e-6):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        self.t = np.linspace(0, duration, self.n_samples)
    
    def generate_normal(self) -> np.ndarray:
        """Clean step response or sine wave"""
        choice = np.random.randint(3)
        
        if choice == 0:
            # Clean step response
            tau = np.random.uniform(5e-6, 20e-6)
            return 3.3 * (1 - np.exp(-self.t / tau))
        
        elif choice == 1:
            # Clean sine wave
            freq = np.random.uniform(10e3, 100e3)
            return 1.65 + 1.5 * np.sin(2 * np.pi * freq * self.t)
        
        else:
            # Clean square wave
            freq = np.random.uniform(10e3, 50e3)
            return 1.65 + 1.5 * np.sign(np.sin(2 * np.pi * freq * self.t))
    
    def generate_overshoot(self) -> np.ndarray:
        """Step response with overshoot (underdamped)"""
        omega_0 = np.random.uniform(1e6, 5e6)
        zeta = np.random.uniform(0.1, 0.4)  # Low damping = overshoot
        
        omega_d = omega_0 * np.sqrt(1 - zeta**2)
        
        waveform = 3.3 * (1 - np.exp(-zeta * omega_0 * self.t) * 
                   (np.cos(omega_d * self.t) + 
                    (zeta / np.sqrt(1 - zeta**2)) * np.sin(omega_d * self.t)))
        
        return waveform
    
    def generate_ringing(self) -> np.ndarray:
        """Signal with sustained oscillation"""
        base = self.generate_overshoot()
        
        # Add extra ringing
        ring_freq = np.random.uniform(500e3, 2e6)
        ring_amp = np.random.uniform(0.2, 0.5)
        decay = np.random.uniform(1e6, 3e6)
        
        ringing = ring_amp * np.sin(2 * np.pi * ring_freq * self.t) * np.exp(-self.t * decay)
        
        return base + ringing
    
    def generate_noise(self) -> np.ndarray:
        """Signal with excessive noise"""
        base = self.generate_normal()
        noise_level = np.random.uniform(0.1, 0.3) * (base.max() - base.min())
        
        return base + np.random.normal(0, noise_level, len(base))
    
    def generate_clipping(self) -> np.ndarray:
        """Clipped sine wave"""
        freq = np.random.uniform(20e3, 100e3)
        amplitude = np.random.uniform(2.0, 4.0)  # Will clip
        
        waveform = 1.65 + amplitude * np.sin(2 * np.pi * freq * self.t)
        
        # Clip to 0-3.3V
        return np.clip(waveform, 0, 3.3)
    
    def generate_slow_rise(self) -> np.ndarray:
        """Very slow rise time"""
        tau = np.random.uniform(30e-6, 80e-6)  # Much slower than normal
        return 3.3 * (1 - np.exp(-self.t / tau))
    
    def generate_distortion(self) -> np.ndarray:
        """Sine with harmonics (distorted)"""
        freq = np.random.uniform(20e3, 80e3)
        fundamental = np.sin(2 * np.pi * freq * self.t)
        
        # Add harmonics
        harmonics = (
            0.2 * np.sin(2 * np.pi * 2 * freq * self.t) +
            0.15 * np.sin(2 * np.pi * 3 * freq * self.t) +
            0.1 * np.sin(2 * np.pi * 4 * freq * self.t)
        )
        
        return 1.65 + 1.5 * (fundamental + harmonics) / 1.45
    
    def generate_dc_offset(self) -> np.ndarray:
        """Signal with DC offset"""
        base = self.generate_normal()
        offset = np.random.uniform(0.5, 1.5) * np.random.choice([-1, 1])
        
        return base + offset
    
    def generate_dataset(self, samples_per_class: int = 500) -> Tuple[List[np.ndarray], List[int]]:
        """
        Generate balanced training dataset.
        
        Returns:
            (waveforms, labels) tuple
        """
        generators = {
            WaveformClass.NORMAL: self.generate_normal,
            WaveformClass.OVERSHOOT: self.generate_overshoot,
            WaveformClass.RINGING: self.generate_ringing,
            WaveformClass.NOISE: self.generate_noise,
            WaveformClass.CLIPPING: self.generate_clipping,
            WaveformClass.SLOW_RISE: self.generate_slow_rise,
            WaveformClass.DISTORTION: self.generate_distortion,
            WaveformClass.DC_OFFSET: self.generate_dc_offset,
        }
        
        waveforms = []
        labels = []
        
        for waveform_class, generator in generators.items():
            print(f"Generating {samples_per_class} samples for {waveform_class.name}...")
            for _ in range(samples_per_class):
                waveforms.append(generator())
                labels.append(waveform_class.value)
        
        return waveforms, labels


# =============================================================================
# Training Utilities
# =============================================================================

if HAS_TORCH:
    
    def train_cnn(
        waveforms: List[np.ndarray],
        labels: List[int],
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
        save_path: Optional[str] = None
    ) -> WaveformCNN:
        """
        Train the CNN classifier.
        
        Args:
            waveforms: List of waveform arrays
            labels: List of class labels
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            validation_split: Fraction for validation
            save_path: Path to save trained model
        
        Returns:
            Trained WaveformCNN model
        """
        # Split data
        n = len(waveforms)
        indices = np.random.permutation(n)
        split = int(n * (1 - validation_split))
        
        train_idx = indices[:split]
        val_idx = indices[split:]
        
        train_data = WaveformDataset(
            [waveforms[i] for i in train_idx],
            [labels[i] for i in train_idx]
        )
        val_data = WaveformDataset(
            [waveforms[i] for i in val_idx],
            [labels[i] for i in val_idx]
        )
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)
        
        # Create model
        model = WaveformCNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        print(f"\nTraining CNN for {epochs} epochs...")
        print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
        print("-" * 50)
        
        best_val_acc = 0
        
        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0
            train_correct = 0
            
            for x, y in train_loader:
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_correct += (outputs.argmax(1) == y).sum().item()
            
            train_acc = train_correct / len(train_data)
            
            # Validate
            model.eval()
            val_correct = 0
            
            with torch.no_grad():
                for x, y in val_loader:
                    outputs = model(x)
                    val_correct += (outputs.argmax(1) == y).sum().item()
            
            val_acc = val_correct / len(val_data)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if save_path:
                    torch.save(model.state_dict(), save_path)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")
        
        print("-" * 50)
        print(f"Best validation accuracy: {best_val_acc:.3f}")
        
        if save_path:
            print(f"Model saved to: {save_path}")
        
        return model


# =============================================================================
# Unified Classifier Interface
# =============================================================================

class WaveformClassifier:
    """
    Unified interface for waveform classification.
    
    Uses CNN if PyTorch is available and model exists,
    otherwise falls back to numpy-based classifier.
    """
    
    # Default paths to search for trained model
    DEFAULT_MODEL_PATHS = [
        "/tmp/waveform_cnn.pt",
        os.path.expanduser("~/.waveformgpt/waveform_cnn.pt"),
        os.path.join(os.path.dirname(__file__), "models", "waveform_cnn.pt"),
        "waveform_cnn.pt",
    ]
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.use_cnn = False
        self.model = None
        
        # Auto-search for model if no path specified
        if model_path is None and HAS_TORCH:
            for path in self.DEFAULT_MODEL_PATHS:
                if os.path.exists(path):
                    model_path = path
                    break
        
        if HAS_TORCH and model_path and os.path.exists(model_path):
            # Load trained CNN
            try:
                self.model = WaveformCNN()
                self.model.load_state_dict(torch.load(model_path, weights_only=True))
                self.model.eval()
                self.use_cnn = True
                print(f"Loaded CNN model from {model_path}")
            except Exception as e:
                print(f"Failed to load CNN: {e}")
                self.model = NumpyWaveformClassifier()
                print("Falling back to feature-based classifier")
        else:
            # Use numpy classifier
            self.model = NumpyWaveformClassifier()
            print("Using feature-based classifier (no CNN model found)")
    
    def classify(self, waveform: np.ndarray) -> ClassificationResult:
        """Classify a waveform"""
        if self.use_cnn:
            return self.model.predict(waveform)
        else:
            return self.model.classify(waveform)
    
    def classify_batch(self, waveforms: List[np.ndarray]) -> List[ClassificationResult]:
        """Classify multiple waveforms"""
        return [self.classify(w) for w in waveforms]
    
    def get_diagnosis(self, waveform: np.ndarray) -> str:
        """Get human-readable diagnosis"""
        result = self.classify(waveform)
        
        diagnosis = {
            WaveformClass.NORMAL: "✅ Waveform looks healthy",
            WaveformClass.OVERSHOOT: "⚠️ Overshoot detected - add snubber capacitor or increase damping",
            WaveformClass.RINGING: "⚠️ Ringing detected - check for parasitic inductance, add RC snubber",
            WaveformClass.NOISE: "⚠️ Excessive noise - add bypass capacitors, check grounding",
            WaveformClass.CLIPPING: "⚠️ Clipping detected - reduce signal amplitude or increase supply voltage",
            WaveformClass.SLOW_RISE: "⚠️ Slow rise time - check for excess capacitance or weak driver",
            WaveformClass.DISTORTION: "⚠️ Distortion detected - reduce amplitude or add negative feedback",
            WaveformClass.DC_OFFSET: "⚠️ DC offset present - add coupling capacitor or check bias",
        }
        
        return f"{diagnosis[result.predicted_class]} (confidence: {result.confidence:.1%})"


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Waveform CNN Classifier - Demo")
    print("=" * 60)
    
    # Generate synthetic data
    print("\n📊 Generating synthetic training data...")
    generator = SyntheticDataGenerator()
    waveforms, labels = generator.generate_dataset(samples_per_class=100)
    
    print(f"\n   Total samples: {len(waveforms)}")
    print(f"   Classes: {len(set(labels))}")
    
    # Test numpy classifier
    print("\n🔍 Testing numpy-based classifier...")
    np_classifier = NumpyWaveformClassifier()
    
    # Test on each type
    test_generators = [
        ("Normal", generator.generate_normal),
        ("Overshoot", generator.generate_overshoot),
        ("Ringing", generator.generate_ringing),
        ("Noise", generator.generate_noise),
        ("Clipping", generator.generate_clipping),
    ]
    
    print("\n   Sample classifications:")
    for name, gen_func in test_generators:
        test_waveform = gen_func()
        result = np_classifier.classify(test_waveform)
        print(f"   - {name}: Predicted {result.predicted_class.name} "
              f"({result.confidence:.1%})")
    
    # Train CNN if PyTorch available
    if HAS_TORCH:
        print("\n🧠 Training CNN classifier...")
        print("   (This may take a few minutes on CPU)")
        
        model = train_cnn(
            waveforms, labels,
            epochs=30,
            batch_size=32,
            save_path="/tmp/waveform_cnn.pt"
        )
        
        # Test CNN
        print("\n📊 Testing CNN classifier...")
        for name, gen_func in test_generators:
            test_waveform = gen_func()
            result = model.predict(test_waveform)
            print(f"   - {name}: Predicted {result.predicted_class.name} "
                  f"({result.confidence:.1%})")
    
    else:
        print("\n⚠️ PyTorch not available. Install with: pip install torch")
        print("   Using numpy-only classifier instead.")
    
    # Show unified interface
    print("\n🎯 Unified Classifier Interface:")
    classifier = WaveformClassifier()
    
    test_waveform = generator.generate_overshoot()
    print(f"\n   {classifier.get_diagnosis(test_waveform)}")
    
    print("\n✅ Demo complete!")
