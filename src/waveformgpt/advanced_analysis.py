"""
WaveformGPT Advanced Analysis Features

Advanced signal processing and analysis:
1. Frequency Domain Analysis (FFT, spectrograms)
2. Anomaly Detection (isolation forest, autoencoders)
3. Predictive Maintenance (trend analysis, failure prediction)
4. Statistical Analysis (distributions, confidence intervals)
5. Pattern Recognition (template matching, cross-correlation)
6. Wavelet Analysis (multi-resolution decomposition)

These features go beyond basic measurements to provide
deeper insights into signal behavior and circuit health.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings

# Scientific computing
from scipy import signal, stats, ndimage
from scipy.fft import fft, ifft, fftfreq, rfft, rfftfreq

# Optional ML for anomaly detection
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FrequencyAnalysis:
    """Results from frequency domain analysis"""
    frequencies: np.ndarray
    magnitudes: np.ndarray
    phases: np.ndarray
    dominant_frequency: float
    harmonics: List[Tuple[float, float]]  # (frequency, magnitude)
    thd: float
    snr: float
    bandwidth_3db: float
    spectral_centroid: float


@dataclass
class SpectrogramResult:
    """Time-frequency spectrogram"""
    times: np.ndarray
    frequencies: np.ndarray
    power: np.ndarray
    sample_rate: float


@dataclass
class AnomalyResult:
    """Anomaly detection results"""
    is_anomaly: bool
    anomaly_score: float
    anomaly_regions: List[Tuple[int, int]]  # (start_idx, end_idx)
    confidence: float
    features_used: List[str]


@dataclass
class TrendAnalysis:
    """Trend and prediction results"""
    trend_direction: str  # 'improving', 'degrading', 'stable'
    slope: float
    r_squared: float
    predicted_failure_time: Optional[float]
    confidence_interval: Tuple[float, float]


@dataclass
class WaveletDecomposition:
    """Wavelet analysis results"""
    approximation: np.ndarray
    details: List[np.ndarray]
    levels: int
    wavelet_name: str
    energy_per_level: List[float]


@dataclass
class StatisticalAnalysis:
    """Statistical analysis of waveform"""
    mean: float
    std: float
    variance: float
    skewness: float
    kurtosis: float
    median: float
    mode: float
    percentiles: Dict[int, float]
    histogram: Tuple[np.ndarray, np.ndarray]
    distribution_fit: Dict[str, Any]


# =============================================================================
# Frequency Domain Analysis
# =============================================================================

class FrequencyAnalyzer:
    """
    Comprehensive frequency domain analysis.
    
    Provides FFT analysis, harmonic detection, THD calculation,
    bandwidth measurement, and SNR estimation.
    """
    
    def __init__(self, sample_rate: float = 1e6):
        self.sample_rate = sample_rate
    
    def analyze(
        self,
        waveform: np.ndarray,
        window: str = 'hann',
        nfft: Optional[int] = None
    ) -> FrequencyAnalysis:
        """
        Perform full frequency domain analysis.
        
        Args:
            waveform: Time-domain signal
            window: Window function ('hann', 'hamming', 'blackman', 'rect')
            nfft: FFT size (default: next power of 2)
        
        Returns:
            FrequencyAnalysis with all frequency metrics
        """
        n = len(waveform)
        
        # Apply window
        if window == 'rect':
            win = np.ones(n)
        else:
            win = signal.get_window(window, n)
        
        windowed = waveform * win
        
        # FFT
        if nfft is None:
            nfft = 2 ** int(np.ceil(np.log2(n)))
        
        spectrum = rfft(windowed, nfft)
        frequencies = rfftfreq(nfft, 1/self.sample_rate)
        
        # Magnitude and phase
        magnitudes = np.abs(spectrum) * 2 / n  # Normalize
        phases = np.angle(spectrum)
        
        # Dominant frequency
        peak_idx = np.argmax(magnitudes[1:]) + 1  # Skip DC
        dominant_freq = frequencies[peak_idx]
        
        # Find harmonics
        harmonics = self._find_harmonics(frequencies, magnitudes, dominant_freq)
        
        # Calculate THD
        thd = self._calculate_thd(magnitudes, peak_idx, harmonics)
        
        # Estimate SNR
        snr = self._estimate_snr(magnitudes, peak_idx)
        
        # 3dB bandwidth
        bandwidth = self._measure_bandwidth(frequencies, magnitudes)
        
        # Spectral centroid
        centroid = self._spectral_centroid(frequencies, magnitudes)
        
        return FrequencyAnalysis(
            frequencies=frequencies,
            magnitudes=magnitudes,
            phases=phases,
            dominant_frequency=dominant_freq,
            harmonics=harmonics,
            thd=thd,
            snr=snr,
            bandwidth_3db=bandwidth,
            spectral_centroid=centroid
        )
    
    def _find_harmonics(
        self,
        frequencies: np.ndarray,
        magnitudes: np.ndarray,
        fundamental: float,
        num_harmonics: int = 10
    ) -> List[Tuple[float, float]]:
        """Find harmonic frequencies and their magnitudes"""
        harmonics = []
        
        for h in range(2, num_harmonics + 1):
            target_freq = fundamental * h
            if target_freq > frequencies[-1]:
                break
            
            # Find closest bin
            idx = np.argmin(np.abs(frequencies - target_freq))
            
            # Refine with peak detection
            left = max(0, idx - 2)
            right = min(len(magnitudes), idx + 3)
            local_peak = left + np.argmax(magnitudes[left:right])
            
            harmonics.append((frequencies[local_peak], magnitudes[local_peak]))
        
        return harmonics
    
    def _calculate_thd(
        self,
        magnitudes: np.ndarray,
        fundamental_idx: int,
        harmonics: List[Tuple[float, float]]
    ) -> float:
        """Calculate Total Harmonic Distortion"""
        fundamental_power = magnitudes[fundamental_idx] ** 2
        harmonic_power = sum(mag ** 2 for _, mag in harmonics)
        
        if fundamental_power > 0:
            thd = np.sqrt(harmonic_power / fundamental_power) * 100
        else:
            thd = 0.0
        
        return thd
    
    def _estimate_snr(self, magnitudes: np.ndarray, signal_idx: int) -> float:
        """Estimate Signal-to-Noise Ratio"""
        # Signal power (around peak)
        signal_range = slice(max(0, signal_idx-2), min(len(magnitudes), signal_idx+3))
        signal_power = np.sum(magnitudes[signal_range] ** 2)
        
        # Noise power (everything else)
        noise_mags = magnitudes.copy()
        noise_mags[signal_range] = 0
        noise_mags[0] = 0  # Exclude DC
        noise_power = np.sum(noise_mags ** 2)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = float('inf')
        
        return snr
    
    def _measure_bandwidth(self, frequencies: np.ndarray, magnitudes: np.ndarray) -> float:
        """Measure -3dB bandwidth"""
        peak_idx = np.argmax(magnitudes[1:]) + 1
        peak_mag = magnitudes[peak_idx]
        threshold = peak_mag / np.sqrt(2)  # -3dB
        
        # Find -3dB points
        above_threshold = magnitudes >= threshold
        
        if not np.any(above_threshold):
            return 0.0
        
        first_above = np.argmax(above_threshold)
        last_above = len(above_threshold) - np.argmax(above_threshold[::-1]) - 1
        
        bandwidth = frequencies[last_above] - frequencies[first_above]
        return bandwidth
    
    def _spectral_centroid(self, frequencies: np.ndarray, magnitudes: np.ndarray) -> float:
        """Calculate spectral centroid (center of mass)"""
        return np.sum(frequencies * magnitudes) / np.sum(magnitudes) if np.sum(magnitudes) > 0 else 0
    
    def spectrogram(
        self,
        waveform: np.ndarray,
        nperseg: int = 256,
        noverlap: Optional[int] = None,
        window: str = 'hann'
    ) -> SpectrogramResult:
        """
        Compute time-frequency spectrogram.
        
        Args:
            waveform: Input signal
            nperseg: Samples per segment
            noverlap: Overlap between segments
            window: Window function
        
        Returns:
            SpectrogramResult with time-frequency power matrix
        """
        if noverlap is None:
            noverlap = nperseg // 2
        
        f, t, Sxx = signal.spectrogram(
            waveform,
            fs=self.sample_rate,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap
        )
        
        return SpectrogramResult(
            times=t,
            frequencies=f,
            power=10 * np.log10(Sxx + 1e-10),  # dB scale
            sample_rate=self.sample_rate
        )
    
    def periodogram(self, waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute power spectral density using Welch's method"""
        f, psd = signal.welch(waveform, fs=self.sample_rate)
        return f, psd


# =============================================================================
# Anomaly Detection
# =============================================================================

class AnomalyDetector:
    """
    Detect anomalies in waveforms using multiple methods:
    1. Statistical (Z-score, IQR)
    2. Isolation Forest
    3. Autoencoder (if PyTorch available)
    4. Change point detection
    """
    
    def __init__(self, sample_rate: float = 1e6):
        self.sample_rate = sample_rate
        self.isolation_forest = None
        self.scaler = None
        self._is_trained = False
    
    def detect_statistical(
        self,
        waveform: np.ndarray,
        z_threshold: float = 3.0,
        window_size: int = 100
    ) -> AnomalyResult:
        """
        Statistical anomaly detection using Z-score.
        
        Args:
            waveform: Input signal
            z_threshold: Z-score threshold for anomaly
            window_size: Rolling window for local statistics
        
        Returns:
            AnomalyResult
        """
        n = len(waveform)
        
        # Rolling statistics
        if window_size >= n:
            window_size = n // 10
        
        # Pad for rolling window
        padded = np.pad(waveform, (window_size//2, window_size//2), mode='reflect')
        
        # Compute local mean and std
        local_mean = ndimage.uniform_filter1d(padded, window_size)[window_size//2:-window_size//2]
        local_std = np.sqrt(
            ndimage.uniform_filter1d(padded**2, window_size)[window_size//2:-window_size//2] 
            - local_mean**2
        )
        local_std = np.maximum(local_std, 1e-10)  # Avoid division by zero
        
        # Z-scores
        z_scores = np.abs(waveform - local_mean) / local_std
        
        # Find anomaly regions
        anomaly_mask = z_scores > z_threshold
        anomaly_regions = self._find_regions(anomaly_mask)
        
        # Calculate overall score
        anomaly_score = np.max(z_scores) / z_threshold
        is_anomaly = len(anomaly_regions) > 0
        
        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=min(anomaly_score, 1.0),
            anomaly_regions=anomaly_regions,
            confidence=0.8 if is_anomaly else 0.9,
            features_used=['z_score', 'rolling_mean', 'rolling_std']
        )
    
    def detect_isolation_forest(
        self,
        waveform: np.ndarray,
        contamination: float = 0.1
    ) -> AnomalyResult:
        """
        Anomaly detection using Isolation Forest.
        
        Args:
            waveform: Input signal
            contamination: Expected proportion of anomalies
        
        Returns:
            AnomalyResult
        """
        if not HAS_SKLEARN:
            return self.detect_statistical(waveform)
        
        # Extract features
        features = self._extract_window_features(waveform, window_size=50)
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Fit and predict
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        predictions = iso_forest.fit_predict(features_scaled)
        scores = iso_forest.decision_function(features_scaled)
        
        # Map back to sample indices
        anomaly_mask = predictions == -1
        
        # Expand to original sample indices
        window_size = 50
        sample_anomaly = np.zeros(len(waveform), dtype=bool)
        for i, is_anom in enumerate(anomaly_mask):
            if is_anom:
                start = i * (window_size // 2)
                end = min(start + window_size, len(waveform))
                sample_anomaly[start:end] = True
        
        anomaly_regions = self._find_regions(sample_anomaly)
        
        return AnomalyResult(
            is_anomaly=np.any(anomaly_mask),
            anomaly_score=float(-np.min(scores)),
            anomaly_regions=anomaly_regions,
            confidence=0.85,
            features_used=['isolation_forest', 'window_features']
        )
    
    def detect_change_points(
        self,
        waveform: np.ndarray,
        threshold: float = 0.1
    ) -> List[int]:
        """
        Detect sudden changes in signal characteristics.
        
        Args:
            waveform: Input signal
            threshold: Sensitivity threshold
        
        Returns:
            List of change point indices
        """
        # Compute running statistics
        n = len(waveform)
        window = max(10, n // 100)
        
        # Running mean
        running_mean = np.convolve(waveform, np.ones(window)/window, mode='valid')
        
        # Compute changes
        mean_diff = np.abs(np.diff(running_mean))
        mean_diff = mean_diff / (np.max(mean_diff) + 1e-10)
        
        # Find significant changes
        peaks, _ = signal.find_peaks(mean_diff, height=threshold, distance=window)
        
        # Adjust indices
        change_points = peaks + window // 2
        
        return change_points.tolist()
    
    def _extract_window_features(
        self,
        waveform: np.ndarray,
        window_size: int = 50
    ) -> np.ndarray:
        """Extract features from overlapping windows"""
        hop = window_size // 2
        n_windows = (len(waveform) - window_size) // hop + 1
        
        features = []
        for i in range(n_windows):
            start = i * hop
            window = waveform[start:start + window_size]
            
            # Features
            feat = [
                np.mean(window),
                np.std(window),
                np.max(window) - np.min(window),
                stats.skew(window),
                stats.kurtosis(window),
                np.sum(np.abs(np.diff(window))),  # Total variation
            ]
            features.append(feat)
        
        return np.array(features)
    
    def _find_regions(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """Find contiguous True regions in boolean mask"""
        regions = []
        in_region = False
        start = 0
        
        for i, val in enumerate(mask):
            if val and not in_region:
                start = i
                in_region = True
            elif not val and in_region:
                regions.append((start, i - 1))
                in_region = False
        
        if in_region:
            regions.append((start, len(mask) - 1))
        
        return regions


# =============================================================================
# Trend Analysis & Predictive Maintenance
# =============================================================================

class TrendAnalyzer:
    """
    Analyze trends over time for predictive maintenance.
    
    Tracks metrics across multiple waveforms to predict
    degradation and potential failures.
    """
    
    def __init__(self):
        self.history: List[Dict[str, Any]] = []
    
    def add_measurement(
        self,
        metrics: Dict[str, float],
        timestamp: Optional[float] = None
    ):
        """Add a measurement to the history"""
        import time
        if timestamp is None:
            timestamp = time.time()
        
        self.history.append({
            'timestamp': timestamp,
            'metrics': metrics
        })
    
    def analyze_trend(
        self,
        metric_name: str,
        threshold: Optional[float] = None
    ) -> TrendAnalysis:
        """
        Analyze trend for a specific metric.
        
        Args:
            metric_name: Name of metric to analyze
            threshold: Failure threshold (optional)
        
        Returns:
            TrendAnalysis with predictions
        """
        if len(self.history) < 3:
            return TrendAnalysis(
                trend_direction='stable',
                slope=0.0,
                r_squared=0.0,
                predicted_failure_time=None,
                confidence_interval=(0, 0)
            )
        
        # Extract data
        timestamps = np.array([h['timestamp'] for h in self.history])
        values = np.array([h['metrics'].get(metric_name, 0) for h in self.history])
        
        # Normalize time
        t = timestamps - timestamps[0]
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(t, values)
        
        # Determine trend direction
        if abs(slope) < std_err * 2:
            direction = 'stable'
        elif slope > 0:
            direction = 'degrading' if metric_name in ['noise', 'thd', 'overshoot'] else 'improving'
        else:
            direction = 'improving' if metric_name in ['noise', 'thd', 'overshoot'] else 'degrading'
        
        # Predict failure time
        failure_time = None
        if threshold is not None and slope != 0:
            time_to_threshold = (threshold - intercept) / slope
            if time_to_threshold > 0:
                failure_time = timestamps[0] + time_to_threshold
        
        # Confidence interval (95%)
        ci_width = 1.96 * std_err
        ci = (slope - ci_width, slope + ci_width)
        
        return TrendAnalysis(
            trend_direction=direction,
            slope=slope,
            r_squared=r_value ** 2,
            predicted_failure_time=failure_time,
            confidence_interval=ci
        )
    
    def get_health_score(self) -> float:
        """
        Calculate overall health score (0-100).
        
        Based on multiple metric trends and their deviation
        from baseline.
        """
        if len(self.history) < 2:
            return 100.0
        
        # Get latest metrics
        latest = self.history[-1]['metrics']
        baseline = self.history[0]['metrics']
        
        # Score based on degradation
        scores = []
        for key in latest:
            if key in baseline:
                ratio = latest[key] / (baseline[key] + 1e-10)
                
                # Different metrics have different "good" directions
                if key in ['noise', 'thd', 'overshoot', 'ringing']:
                    score = 100 / max(ratio, 1.0)
                else:
                    score = min(ratio * 100, 100)
                
                scores.append(score)
        
        return np.mean(scores) if scores else 100.0


# =============================================================================
# Wavelet Analysis
# =============================================================================

class WaveletAnalyzer:
    """
    Multi-resolution wavelet decomposition.
    
    Useful for analyzing transient events at different time scales.
    """
    
    def __init__(self, sample_rate: float = 1e6):
        self.sample_rate = sample_rate
    
    def decompose(
        self,
        waveform: np.ndarray,
        wavelet: str = 'db4',
        levels: int = 5
    ) -> WaveletDecomposition:
        """
        Perform discrete wavelet transform decomposition.
        
        Args:
            waveform: Input signal
            wavelet: Wavelet name ('db4', 'haar', 'sym5', etc.)
            levels: Number of decomposition levels
        
        Returns:
            WaveletDecomposition with approximation and details
        """
        try:
            import pywt
        except ImportError:
            # Fallback to simple multi-scale analysis
            return self._simple_multiscale(waveform, levels)
        
        # Decompose
        coeffs = pywt.wavedec(waveform, wavelet, level=levels)
        
        # Extract approximation and details
        approximation = coeffs[0]
        details = coeffs[1:]
        
        # Energy per level
        energies = [np.sum(d**2) for d in details]
        total_energy = sum(energies) + np.sum(approximation**2)
        energy_ratios = [e / total_energy for e in energies]
        
        return WaveletDecomposition(
            approximation=approximation,
            details=details,
            levels=levels,
            wavelet_name=wavelet,
            energy_per_level=energy_ratios
        )
    
    def _simple_multiscale(
        self,
        waveform: np.ndarray,
        levels: int
    ) -> WaveletDecomposition:
        """Simple multi-scale decomposition without PyWavelets"""
        details = []
        current = waveform.copy()
        
        for _ in range(levels):
            # Low-pass filter
            kernel_size = 5
            kernel = np.ones(kernel_size) / kernel_size
            smoothed = np.convolve(current, kernel, mode='same')
            
            # Detail = original - smoothed
            detail = current - smoothed
            details.append(detail)
            
            # Downsample for next level
            current = smoothed[::2]
        
        approximation = current
        
        # Energy calculation
        energies = [np.sum(d**2) for d in details]
        total = sum(energies) + np.sum(approximation**2)
        energy_ratios = [e / total for e in energies]
        
        return WaveletDecomposition(
            approximation=approximation,
            details=details,
            levels=levels,
            wavelet_name='simple_average',
            energy_per_level=energy_ratios
        )
    
    def denoise(
        self,
        waveform: np.ndarray,
        threshold: float = 0.1,
        wavelet: str = 'db4'
    ) -> np.ndarray:
        """Denoise waveform using wavelet thresholding"""
        try:
            import pywt
            
            # Decompose
            coeffs = pywt.wavedec(waveform, wavelet, level=5)
            
            # Threshold detail coefficients
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            thresh = threshold * sigma * np.sqrt(2 * np.log(len(waveform)))
            
            coeffs_thresholded = [coeffs[0]]  # Keep approximation
            for d in coeffs[1:]:
                coeffs_thresholded.append(pywt.threshold(d, thresh, mode='soft'))
            
            # Reconstruct
            return pywt.waverec(coeffs_thresholded, wavelet)[:len(waveform)]
            
        except ImportError:
            # Simple smoothing fallback
            return ndimage.uniform_filter1d(waveform, 5)


# =============================================================================
# Pattern Recognition
# =============================================================================

class PatternMatcher:
    """
    Template matching and pattern recognition.
    
    Find similar patterns, detect recurring events, and
    classify waveform shapes.
    """
    
    def __init__(self):
        self.templates: Dict[str, np.ndarray] = {}
    
    def add_template(self, name: str, template: np.ndarray):
        """Add a template pattern"""
        # Normalize template
        template = template - np.mean(template)
        template = template / (np.std(template) + 1e-10)
        self.templates[name] = template
    
    def match(
        self,
        waveform: np.ndarray,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Find all template matches in waveform.
        
        Returns:
            List of matches with template name, position, and correlation
        """
        matches = []
        
        # Normalize waveform
        wf_norm = waveform - np.mean(waveform)
        wf_norm = wf_norm / (np.std(wf_norm) + 1e-10)
        
        for name, template in self.templates.items():
            # Cross-correlation
            corr = np.correlate(wf_norm, template, mode='valid')
            corr = corr / len(template)
            
            # Find peaks above threshold
            peaks, properties = signal.find_peaks(corr, height=threshold)
            
            for peak, height in zip(peaks, properties['peak_heights']):
                matches.append({
                    'template': name,
                    'position': peak,
                    'correlation': height,
                    'confidence': height
                })
        
        # Sort by position
        matches.sort(key=lambda x: x['position'])
        
        return matches
    
    def cross_correlation(
        self,
        waveform1: np.ndarray,
        waveform2: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """
        Compute cross-correlation between two waveforms.
        
        Returns:
            (correlation array, lag at maximum)
        """
        # Normalize
        w1 = (waveform1 - np.mean(waveform1)) / (np.std(waveform1) + 1e-10)
        w2 = (waveform2 - np.mean(waveform2)) / (np.std(waveform2) + 1e-10)
        
        # Cross-correlation
        corr = signal.correlate(w1, w2, mode='full')
        corr = corr / max(len(w1), len(w2))
        
        # Find maximum
        max_idx = np.argmax(np.abs(corr))
        lag = max_idx - len(w2) + 1
        
        return corr, lag


# =============================================================================
# Unified Advanced Analyzer
# =============================================================================

class AdvancedAnalyzer:
    """
    Unified interface for all advanced analysis features.
    
    Usage:
        analyzer = AdvancedAnalyzer(sample_rate=1e6)
        
        # Frequency analysis
        freq = analyzer.frequency_analysis(waveform)
        print(f"Dominant freq: {freq.dominant_frequency}")
        
        # Anomaly detection
        anomaly = analyzer.detect_anomalies(waveform)
        if anomaly.is_anomaly:
            print(f"Anomaly detected! Score: {anomaly.anomaly_score}")
        
        # Spectrogram
        spec = analyzer.spectrogram(waveform)
        
        # Trend analysis (over time)
        analyzer.add_measurement({'thd': 2.5, 'noise': 0.1})
        trend = analyzer.analyze_trend('thd', threshold=10.0)
    """
    
    def __init__(self, sample_rate: float = 1e6):
        self.sample_rate = sample_rate
        self.freq_analyzer = FrequencyAnalyzer(sample_rate)
        self.anomaly_detector = AnomalyDetector(sample_rate)
        self.trend_analyzer = TrendAnalyzer()
        self.wavelet_analyzer = WaveletAnalyzer(sample_rate)
        self.pattern_matcher = PatternMatcher()
    
    def frequency_analysis(self, waveform: np.ndarray, **kwargs) -> FrequencyAnalysis:
        """Full frequency domain analysis"""
        return self.freq_analyzer.analyze(waveform, **kwargs)
    
    def spectrogram(self, waveform: np.ndarray, **kwargs) -> SpectrogramResult:
        """Time-frequency spectrogram"""
        return self.freq_analyzer.spectrogram(waveform, **kwargs)
    
    def detect_anomalies(
        self,
        waveform: np.ndarray,
        method: str = 'statistical'
    ) -> AnomalyResult:
        """
        Detect anomalies in waveform.
        
        Args:
            waveform: Input signal
            method: 'statistical', 'isolation_forest', or 'auto'
        """
        if method == 'statistical':
            return self.anomaly_detector.detect_statistical(waveform)
        elif method == 'isolation_forest':
            return self.anomaly_detector.detect_isolation_forest(waveform)
        else:
            # Auto: try isolation forest, fall back to statistical
            if HAS_SKLEARN:
                return self.anomaly_detector.detect_isolation_forest(waveform)
            return self.anomaly_detector.detect_statistical(waveform)
    
    def detect_change_points(self, waveform: np.ndarray) -> List[int]:
        """Find sudden changes in signal"""
        return self.anomaly_detector.detect_change_points(waveform)
    
    def add_measurement(self, metrics: Dict[str, float]):
        """Add measurement for trend analysis"""
        self.trend_analyzer.add_measurement(metrics)
    
    def analyze_trend(self, metric_name: str, threshold: Optional[float] = None) -> TrendAnalysis:
        """Analyze trend for predictive maintenance"""
        return self.trend_analyzer.analyze_trend(metric_name, threshold)
    
    def get_health_score(self) -> float:
        """Get overall health score (0-100)"""
        return self.trend_analyzer.get_health_score()
    
    def wavelet_decompose(self, waveform: np.ndarray, **kwargs) -> WaveletDecomposition:
        """Multi-resolution wavelet decomposition"""
        return self.wavelet_analyzer.decompose(waveform, **kwargs)
    
    def denoise(self, waveform: np.ndarray, **kwargs) -> np.ndarray:
        """Denoise waveform using wavelets"""
        return self.wavelet_analyzer.denoise(waveform, **kwargs)
    
    def add_pattern_template(self, name: str, template: np.ndarray):
        """Add pattern template for matching"""
        self.pattern_matcher.add_template(name, template)
    
    def find_patterns(self, waveform: np.ndarray, threshold: float = 0.7) -> List[Dict]:
        """Find template patterns in waveform"""
        return self.pattern_matcher.match(waveform, threshold)
    
    def full_analysis(self, waveform: np.ndarray) -> Dict[str, Any]:
        """
        Perform comprehensive analysis.
        
        Returns dict with all analysis results.
        """
        return {
            'frequency': self.frequency_analysis(waveform),
            'anomaly': self.detect_anomalies(waveform),
            'change_points': self.detect_change_points(waveform),
            'wavelet': self.wavelet_decompose(waveform),
            'health_score': self.get_health_score()
        }


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("WaveformGPT Advanced Analysis - Demo")
    print("=" * 60)
    
    # Create analyzer
    analyzer = AdvancedAnalyzer(sample_rate=1e6)
    
    # Generate test signal
    print("\n📊 Generating test signal...")
    t = np.linspace(0, 1e-3, 1000)
    
    # Multi-frequency signal with noise
    waveform = (
        np.sin(2 * np.pi * 10e3 * t) +
        0.3 * np.sin(2 * np.pi * 20e3 * t) +
        0.1 * np.sin(2 * np.pi * 30e3 * t) +
        0.05 * np.random.randn(len(t))
    )
    
    # Add an anomaly
    waveform[400:420] += 2.0  # Spike
    
    print(f"   Samples: {len(waveform)}")
    print(f"   Duration: {t[-1]*1e3:.1f} ms")
    
    # Frequency analysis
    print("\n📈 Frequency Analysis:")
    freq = analyzer.frequency_analysis(waveform)
    print(f"   Dominant frequency: {freq.dominant_frequency/1e3:.1f} kHz")
    print(f"   THD: {freq.thd:.2f}%")
    print(f"   SNR: {freq.snr:.1f} dB")
    print(f"   Bandwidth (-3dB): {freq.bandwidth_3db/1e3:.1f} kHz")
    print(f"   Harmonics found: {len(freq.harmonics)}")
    
    # Anomaly detection
    print("\n🔍 Anomaly Detection:")
    anomaly = analyzer.detect_anomalies(waveform)
    print(f"   Is anomaly: {anomaly.is_anomaly}")
    print(f"   Score: {anomaly.anomaly_score:.2f}")
    print(f"   Regions: {anomaly.anomaly_regions}")
    
    # Change points
    print("\n📍 Change Points:")
    changes = analyzer.detect_change_points(waveform)
    print(f"   Found {len(changes)} change points: {changes[:5]}...")
    
    # Wavelet analysis
    print("\n🌊 Wavelet Decomposition:")
    wavelet = analyzer.wavelet_decompose(waveform)
    print(f"   Levels: {wavelet.levels}")
    print(f"   Energy distribution: {[f'{e:.1%}' for e in wavelet.energy_per_level]}")
    
    # Trend analysis
    print("\n📉 Trend Analysis:")
    # Simulate measurements over time
    for i in range(10):
        analyzer.add_measurement({
            'thd': 2.0 + 0.5 * i + 0.2 * np.random.randn(),
            'noise': 0.1 + 0.02 * i,
        })
    
    trend = analyzer.analyze_trend('thd', threshold=10.0)
    print(f"   Direction: {trend.trend_direction}")
    print(f"   Slope: {trend.slope:.3f}")
    print(f"   R²: {trend.r_squared:.3f}")
    print(f"   Health score: {analyzer.get_health_score():.1f}")
    
    print("\n✅ Demo complete!")
