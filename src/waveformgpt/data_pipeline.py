"""
WaveformGPT Data Collection Pipeline

This is the REAL data infrastructure that makes this project valuable.

Features:
1. Collect waveforms from multiple sources (ESP32, oscilloscopes, uploads)
2. Label and annotate with problem types
3. Store in structured format (HDF5 for ML, SQLite for metadata)
4. Export for training (TFRecord, NumPy, CSV)
5. Dataset versioning and statistics
6. Active learning integration

This creates a REAL dataset - not synthetic garbage.
"""

import numpy as np
import json
import sqlite3
import hashlib
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import struct
import threading
from queue import Queue
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WaveformGPT.DataPipeline")

# Optional imports
try:
    import h5py
    HAS_HDF5 = True
except ImportError:
    HAS_HDF5 = False
    logger.warning("h5py not installed - HDF5 export disabled")

try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False


# =============================================================================
# Data Classes
# =============================================================================

class WaveformSource(Enum):
    """Where the waveform came from"""
    ESP32 = "esp32"
    OSCILLOSCOPE = "oscilloscope"
    UPLOAD = "upload"
    SIMULATION = "simulation"
    SYNTHETIC = "synthetic"
    IMAGE_EXTRACTION = "image_extraction"


class ProblemLabel(Enum):
    """Waveform problem classifications"""
    CLEAN = "clean"
    NOISY = "noisy"
    CLIPPED = "clipped"
    RINGING = "ringing"
    OVERSHOOT = "overshoot"
    UNDERSHOOT = "undershoot"
    GROUND_BOUNCE = "ground_bounce"
    SLOW_EDGES = "slow_edges"
    CROSSTALK = "crosstalk"
    EMI = "emi"
    POWER_SUPPLY_NOISE = "power_supply_noise"
    REFLECTION = "reflection"
    UNKNOWN = "unknown"


class ConfidenceLevel(Enum):
    """How confident we are in the label"""
    VERIFIED = "verified"       # Human verified
    HIGH = "high"               # Model confident
    MEDIUM = "medium"           # Model somewhat confident
    LOW = "low"                 # Model uncertain - needs review
    UNLABELED = "unlabeled"     # No label yet


@dataclass
class WaveformSample:
    """A single waveform sample with full metadata"""
    # Unique identifier
    id: str
    
    # The actual data
    samples: np.ndarray
    sample_rate: float
    
    # Source information
    source: WaveformSource
    source_device: str = ""
    capture_timestamp: float = 0.0
    
    # Labels
    primary_label: ProblemLabel = ProblemLabel.UNKNOWN
    secondary_labels: List[ProblemLabel] = field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.UNLABELED
    
    # Analysis results
    measurements: Dict[str, float] = field(default_factory=dict)
    features: Dict[str, float] = field(default_factory=dict)
    
    # Annotations
    notes: str = ""
    annotator: str = ""
    annotation_timestamp: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.capture_timestamp == 0.0:
            self.capture_timestamp = time.time()
        if not self.id:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique ID from content hash"""
        content = f"{self.samples.tobytes()}{self.sample_rate}{self.capture_timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict:
        """Convert to serializable dict"""
        return {
            'id': self.id,
            'samples': self.samples.tolist(),
            'sample_rate': self.sample_rate,
            'source': self.source.value,
            'source_device': self.source_device,
            'capture_timestamp': self.capture_timestamp,
            'primary_label': self.primary_label.value,
            'secondary_labels': [l.value for l in self.secondary_labels],
            'confidence': self.confidence.value,
            'measurements': self.measurements,
            'features': self.features,
            'notes': self.notes,
            'annotator': self.annotator,
            'annotation_timestamp': self.annotation_timestamp,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'WaveformSample':
        """Create from dict"""
        return cls(
            id=data['id'],
            samples=np.array(data['samples']),
            sample_rate=data['sample_rate'],
            source=WaveformSource(data['source']),
            source_device=data.get('source_device', ''),
            capture_timestamp=data.get('capture_timestamp', 0.0),
            primary_label=ProblemLabel(data.get('primary_label', 'unknown')),
            secondary_labels=[ProblemLabel(l) for l in data.get('secondary_labels', [])],
            confidence=ConfidenceLevel(data.get('confidence', 'unlabeled')),
            measurements=data.get('measurements', {}),
            features=data.get('features', {}),
            notes=data.get('notes', ''),
            annotator=data.get('annotator', ''),
            annotation_timestamp=data.get('annotation_timestamp', 0.0),
            metadata=data.get('metadata', {})
        )


@dataclass
class DatasetStats:
    """Statistics about the dataset"""
    total_samples: int = 0
    labeled_samples: int = 0
    verified_samples: int = 0
    samples_by_label: Dict[str, int] = field(default_factory=dict)
    samples_by_source: Dict[str, int] = field(default_factory=dict)
    avg_sample_length: float = 0.0
    min_sample_length: int = 0
    max_sample_length: int = 0
    total_duration_hours: float = 0.0
    created_at: float = 0.0
    updated_at: float = 0.0


# =============================================================================
# ESP32 Collector
# =============================================================================

class ESP32Collector:
    """
    Collect real waveform data from ESP32 hardware.
    
    This is where REAL data comes from.
    """
    
    def __init__(
        self,
        port: str = "/dev/cu.usbserial-0001",
        baud_rate: int = 115200,
        sample_rate: float = 44100.0
    ):
        self.port = port
        self.baud_rate = baud_rate
        self.sample_rate = sample_rate
        self.serial: Optional[serial.Serial] = None
        self._collecting = False
        self._buffer = []
        self._collection_thread: Optional[threading.Thread] = None
    
    def connect(self) -> bool:
        """Connect to ESP32"""
        if not HAS_SERIAL:
            logger.error("pyserial not installed")
            return False
        
        try:
            self.serial = serial.Serial(
                self.port,
                self.baud_rate,
                timeout=1
            )
            time.sleep(2)  # Wait for ESP32 reset
            
            # Clear buffer
            while self.serial.in_waiting:
                self.serial.read(self.serial.in_waiting)
            
            logger.info(f"Connected to ESP32 on {self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from ESP32"""
        if self.serial and self.serial.is_open:
            self.serial.close()
            logger.info("Disconnected from ESP32")
    
    def capture_single(self, duration_ms: int = 100) -> Optional[WaveformSample]:
        """
        Capture a single waveform sample.
        
        Args:
            duration_ms: Capture duration in milliseconds
        
        Returns:
            WaveformSample or None if failed
        """
        if not self.serial or not self.serial.is_open:
            logger.error("Not connected to ESP32")
            return None
        
        try:
            # Send capture command
            self.serial.write(b'c')  # 'c' = capture command
            
            # Read response
            samples = []
            start = time.time()
            timeout = duration_ms / 1000 + 1.0
            
            while time.time() - start < timeout:
                if self.serial.in_waiting:
                    line = self.serial.readline().decode('utf-8', errors='ignore').strip()
                    
                    # Parse sample values
                    if line.startswith("DATA:"):
                        values = line[5:].split(',')
                        samples.extend([float(v) for v in values if v])
                    elif line == "END":
                        break
            
            if len(samples) < 10:
                logger.warning("Too few samples captured")
                return None
            
            # Create sample
            return WaveformSample(
                id="",
                samples=np.array(samples),
                sample_rate=self.sample_rate,
                source=WaveformSource.ESP32,
                source_device=self.port,
                capture_timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Capture failed: {e}")
            return None
    
    def capture_continuous(
        self,
        callback: Callable[[WaveformSample], None],
        interval_ms: int = 500
    ):
        """
        Start continuous capture in background.
        
        Args:
            callback: Function to call with each captured sample
            interval_ms: Interval between captures
        """
        self._collecting = True
        
        def collection_loop():
            while self._collecting:
                sample = self.capture_single(100)
                if sample:
                    callback(sample)
                time.sleep(interval_ms / 1000)
        
        self._collection_thread = threading.Thread(target=collection_loop, daemon=True)
        self._collection_thread.start()
        logger.info("Started continuous collection")
    
    def stop_continuous(self):
        """Stop continuous capture"""
        self._collecting = False
        if self._collection_thread:
            self._collection_thread.join(timeout=2)
        logger.info("Stopped continuous collection")
    
    @staticmethod
    def list_ports() -> List[str]:
        """List available serial ports"""
        if not HAS_SERIAL:
            return []
        ports = serial.tools.list_ports.comports()
        return [p.device for p in ports]


# =============================================================================
# Oscilloscope Integration
# =============================================================================

class SCPIOscilloscope:
    """
    Connect to real oscilloscopes via SCPI protocol.
    
    Supports: Rigol, Tektronix, Keysight, Siglent
    
    This pulls REAL waveform data from professional equipment.
    """
    
    # Common SCPI commands
    COMMANDS = {
        'idn': '*IDN?',
        'waveform_data': ':WAV:DATA?',
        'waveform_source': ':WAV:SOUR CHAN{}',
        'waveform_mode': ':WAV:MODE RAW',
        'waveform_format': ':WAV:FORM BYTE',
        'waveform_points': ':WAV:POIN?',
        'sample_rate': ':ACQ:SRAT?',
        'time_scale': ':TIM:SCAL?',
        'voltage_scale': ':CHAN{}:SCAL?',
        'trigger': ':TRIG:STAT?',
        'run': ':RUN',
        'stop': ':STOP',
        'single': ':SING',
    }
    
    def __init__(self, address: str, port: int = 5555):
        """
        Initialize oscilloscope connection.
        
        Args:
            address: IP address or VISA resource string
            port: TCP port (default 5555 for LXI)
        """
        self.address = address
        self.port = port
        self.socket = None
        self.manufacturer = ""
        self.model = ""
    
    def connect(self) -> bool:
        """Connect to oscilloscope via TCP/IP"""
        import socket
        
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5)
            self.socket.connect((self.address, self.port))
            
            # Get identification
            idn = self._query(self.COMMANDS['idn'])
            if idn:
                parts = idn.split(',')
                self.manufacturer = parts[0] if len(parts) > 0 else "Unknown"
                self.model = parts[1] if len(parts) > 1 else "Unknown"
                logger.info(f"Connected to {self.manufacturer} {self.model}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to connect to oscilloscope: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from oscilloscope"""
        if self.socket:
            self.socket.close()
            self.socket = None
    
    def _write(self, command: str):
        """Send command"""
        if self.socket:
            self.socket.send((command + '\n').encode())
    
    def _read(self, size: int = 4096) -> str:
        """Read response"""
        if self.socket:
            return self.socket.recv(size).decode('utf-8', errors='ignore').strip()
        return ""
    
    def _query(self, command: str) -> str:
        """Send command and get response"""
        self._write(command)
        time.sleep(0.1)
        return self._read()
    
    def capture_channel(self, channel: int = 1) -> Optional[WaveformSample]:
        """
        Capture waveform from specified channel.
        
        Args:
            channel: Channel number (1-4)
        
        Returns:
            WaveformSample or None
        """
        try:
            # Configure acquisition
            self._write(self.COMMANDS['waveform_source'].format(channel))
            self._write(self.COMMANDS['waveform_mode'])
            self._write(self.COMMANDS['waveform_format'])
            
            # Get sample rate
            sample_rate = float(self._query(self.COMMANDS['sample_rate']))
            
            # Capture
            self._write(':SING')  # Single trigger
            time.sleep(0.5)
            
            # Get data
            self._write(self.COMMANDS['waveform_data'])
            time.sleep(0.2)
            
            # Read raw data (binary with header)
            raw = self.socket.recv(100000)
            
            # Parse IEEE 488.2 block data
            if raw[0:1] == b'#':
                header_len = int(raw[1:2])
                data_len = int(raw[2:2+header_len])
                data = raw[2+header_len:2+header_len+data_len]
                samples = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
                
                # Normalize to voltage (approximate)
                voltage_scale = float(self._query(self.COMMANDS['voltage_scale'].format(channel)))
                samples = (samples - 128) / 128 * voltage_scale * 4
                
                return WaveformSample(
                    id="",
                    samples=samples,
                    sample_rate=sample_rate,
                    source=WaveformSource.OSCILLOSCOPE,
                    source_device=f"{self.manufacturer} {self.model}",
                    metadata={'channel': channel, 'voltage_scale': voltage_scale}
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Capture failed: {e}")
            return None
    
    def auto_discover(self) -> List[str]:
        """
        Auto-discover oscilloscopes on network.
        
        Returns:
            List of discovered IP addresses
        """
        import socket
        
        discovered = []
        
        # Common oscilloscope ports
        ports = [5555, 5025, 111]
        
        # Scan local subnet
        try:
            # Get local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            
            subnet = '.'.join(local_ip.split('.')[:-1])
            
            for i in range(1, 255):
                ip = f"{subnet}.{i}"
                for port in ports:
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(0.1)
                        result = sock.connect_ex((ip, port))
                        if result == 0:
                            discovered.append(f"{ip}:{port}")
                        sock.close()
                    except:
                        pass
        except:
            pass
        
        return discovered


# =============================================================================
# Dataset Manager
# =============================================================================

class DatasetManager:
    """
    Manages the waveform dataset.
    
    - Stores samples in HDF5 (efficient for ML) + SQLite (for metadata)
    - Provides labeling interface
    - Exports in multiple formats
    - Tracks dataset versions
    """
    
    def __init__(self, data_dir: str = "~/.waveformgpt/data"):
        self.data_dir = Path(data_dir).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.data_dir / "metadata.db"
        self.hdf5_path = self.data_dir / "waveforms.h5"
        
        self._init_database()
        logger.info(f"Dataset initialized at {self.data_dir}")
    
    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS samples (
                id TEXT PRIMARY KEY,
                source TEXT,
                source_device TEXT,
                capture_timestamp REAL,
                sample_rate REAL,
                sample_length INTEGER,
                primary_label TEXT,
                confidence TEXT,
                notes TEXT,
                annotator TEXT,
                annotation_timestamp REAL,
                created_at REAL DEFAULT (strftime('%s', 'now')),
                hdf5_index INTEGER
            );
            
            CREATE TABLE IF NOT EXISTS secondary_labels (
                sample_id TEXT,
                label TEXT,
                FOREIGN KEY (sample_id) REFERENCES samples(id)
            );
            
            CREATE TABLE IF NOT EXISTS measurements (
                sample_id TEXT,
                name TEXT,
                value REAL,
                FOREIGN KEY (sample_id) REFERENCES samples(id)
            );
            
            CREATE TABLE IF NOT EXISTS features (
                sample_id TEXT,
                name TEXT,
                value REAL,
                FOREIGN KEY (sample_id) REFERENCES samples(id)
            );
            
            CREATE TABLE IF NOT EXISTS dataset_versions (
                version INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL,
                sample_count INTEGER,
                labeled_count INTEGER,
                notes TEXT
            );
            
            CREATE INDEX IF NOT EXISTS idx_samples_label ON samples(primary_label);
            CREATE INDEX IF NOT EXISTS idx_samples_source ON samples(source);
            CREATE INDEX IF NOT EXISTS idx_samples_confidence ON samples(confidence);
        """)
        
        conn.commit()
        conn.close()
    
    def add_sample(self, sample: WaveformSample) -> bool:
        """
        Add a waveform sample to the dataset.
        
        Args:
            sample: WaveformSample to add
        
        Returns:
            True if successful
        """
        try:
            # Store waveform data in HDF5
            hdf5_index = self._store_waveform(sample)
            
            # Store metadata in SQLite
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO samples 
                (id, source, source_device, capture_timestamp, sample_rate,
                 sample_length, primary_label, confidence, notes, annotator,
                 annotation_timestamp, hdf5_index)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sample.id,
                sample.source.value,
                sample.source_device,
                sample.capture_timestamp,
                sample.sample_rate,
                len(sample.samples),
                sample.primary_label.value,
                sample.confidence.value,
                sample.notes,
                sample.annotator,
                sample.annotation_timestamp,
                hdf5_index
            ))
            
            # Store secondary labels
            for label in sample.secondary_labels:
                cursor.execute(
                    "INSERT INTO secondary_labels (sample_id, label) VALUES (?, ?)",
                    (sample.id, label.value)
                )
            
            # Store measurements
            for name, value in sample.measurements.items():
                cursor.execute(
                    "INSERT INTO measurements (sample_id, name, value) VALUES (?, ?, ?)",
                    (sample.id, name, value)
                )
            
            # Store features
            for name, value in sample.features.items():
                cursor.execute(
                    "INSERT INTO features (sample_id, name, value) VALUES (?, ?, ?)",
                    (sample.id, name, value)
                )
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Added sample {sample.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add sample: {e}")
            return False
    
    def _store_waveform(self, sample: WaveformSample) -> int:
        """Store waveform data in HDF5"""
        if not HAS_HDF5:
            # Fallback: store as numpy file
            np_path = self.data_dir / "waveforms" / f"{sample.id}.npy"
            np_path.parent.mkdir(exist_ok=True)
            np.save(np_path, sample.samples)
            return 0
        
        with h5py.File(self.hdf5_path, 'a') as f:
            if 'waveforms' not in f:
                # Create resizable dataset
                f.create_dataset(
                    'waveforms',
                    shape=(0,),
                    maxshape=(None,),
                    dtype=h5py.vlen_dtype(np.float32)
                )
                f.create_dataset(
                    'sample_ids',
                    shape=(0,),
                    maxshape=(None,),
                    dtype=h5py.string_dtype()
                )
            
            idx = len(f['waveforms'])
            
            # Resize and add
            f['waveforms'].resize(idx + 1, axis=0)
            f['sample_ids'].resize(idx + 1, axis=0)
            
            f['waveforms'][idx] = sample.samples.astype(np.float32)
            f['sample_ids'][idx] = sample.id
            
            return idx
    
    def get_sample(self, sample_id: str) -> Optional[WaveformSample]:
        """Retrieve a sample by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM samples WHERE id = ?", (sample_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return None
        
        # Get waveform data
        hdf5_index = row[12]
        samples = self._load_waveform(sample_id, hdf5_index)
        
        # Get secondary labels
        cursor.execute(
            "SELECT label FROM secondary_labels WHERE sample_id = ?",
            (sample_id,)
        )
        secondary_labels = [ProblemLabel(r[0]) for r in cursor.fetchall()]
        
        # Get measurements
        cursor.execute(
            "SELECT name, value FROM measurements WHERE sample_id = ?",
            (sample_id,)
        )
        measurements = {r[0]: r[1] for r in cursor.fetchall()}
        
        # Get features
        cursor.execute(
            "SELECT name, value FROM features WHERE sample_id = ?",
            (sample_id,)
        )
        features = {r[0]: r[1] for r in cursor.fetchall()}
        
        conn.close()
        
        return WaveformSample(
            id=row[0],
            samples=samples,
            sample_rate=row[4],
            source=WaveformSource(row[1]),
            source_device=row[2],
            capture_timestamp=row[3],
            primary_label=ProblemLabel(row[6]),
            secondary_labels=secondary_labels,
            confidence=ConfidenceLevel(row[7]),
            measurements=measurements,
            features=features,
            notes=row[8] or "",
            annotator=row[9] or "",
            annotation_timestamp=row[10] or 0.0
        )
    
    def _load_waveform(self, sample_id: str, hdf5_index: int) -> np.ndarray:
        """Load waveform data from HDF5"""
        if not HAS_HDF5:
            np_path = self.data_dir / "waveforms" / f"{sample_id}.npy"
            if np_path.exists():
                return np.load(np_path)
            return np.array([])
        
        try:
            with h5py.File(self.hdf5_path, 'r') as f:
                return np.array(f['waveforms'][hdf5_index])
        except:
            return np.array([])
    
    def label_sample(
        self,
        sample_id: str,
        label: ProblemLabel,
        confidence: ConfidenceLevel = ConfidenceLevel.HIGH,
        annotator: str = "user",
        notes: str = ""
    ) -> bool:
        """
        Label or relabel a sample.
        
        Args:
            sample_id: Sample ID to label
            label: Primary problem label
            confidence: Confidence level
            annotator: Who labeled it
            notes: Additional notes
        
        Returns:
            True if successful
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE samples 
            SET primary_label = ?, confidence = ?, annotator = ?,
                annotation_timestamp = ?, notes = ?
            WHERE id = ?
        """, (label.value, confidence.value, annotator, time.time(), notes, sample_id))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        if success:
            logger.info(f"Labeled sample {sample_id} as {label.value}")
        
        return success
    
    def get_unlabeled(self, limit: int = 100) -> List[str]:
        """Get IDs of unlabeled samples"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id FROM samples 
            WHERE confidence = 'unlabeled' OR primary_label = 'unknown'
            LIMIT ?
        """, (limit,))
        
        ids = [r[0] for r in cursor.fetchall()]
        conn.close()
        return ids
    
    def get_samples_by_label(
        self,
        label: ProblemLabel,
        limit: int = 1000
    ) -> List[str]:
        """Get sample IDs by label"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id FROM samples WHERE primary_label = ? LIMIT ?
        """, (label.value, limit))
        
        ids = [r[0] for r in cursor.fetchall()]
        conn.close()
        return ids
    
    def get_stats(self) -> DatasetStats:
        """Get dataset statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total samples
        cursor.execute("SELECT COUNT(*) FROM samples")
        total = cursor.fetchone()[0]
        
        # Labeled samples
        cursor.execute("""
            SELECT COUNT(*) FROM samples 
            WHERE confidence != 'unlabeled' AND primary_label != 'unknown'
        """)
        labeled = cursor.fetchone()[0]
        
        # Verified samples
        cursor.execute("""
            SELECT COUNT(*) FROM samples WHERE confidence = 'verified'
        """)
        verified = cursor.fetchone()[0]
        
        # By label
        cursor.execute("""
            SELECT primary_label, COUNT(*) FROM samples GROUP BY primary_label
        """)
        by_label = {r[0]: r[1] for r in cursor.fetchall()}
        
        # By source
        cursor.execute("""
            SELECT source, COUNT(*) FROM samples GROUP BY source
        """)
        by_source = {r[0]: r[1] for r in cursor.fetchall()}
        
        # Sample lengths
        cursor.execute("""
            SELECT AVG(sample_length), MIN(sample_length), MAX(sample_length),
                   SUM(sample_length * 1.0 / sample_rate)
            FROM samples
        """)
        row = cursor.fetchone()
        
        conn.close()
        
        return DatasetStats(
            total_samples=total,
            labeled_samples=labeled,
            verified_samples=verified,
            samples_by_label=by_label,
            samples_by_source=by_source,
            avg_sample_length=row[0] or 0,
            min_sample_length=row[1] or 0,
            max_sample_length=row[2] or 0,
            total_duration_hours=(row[3] or 0) / 3600,
            updated_at=time.time()
        )
    
    def export_for_training(
        self,
        output_path: str,
        format: str = "numpy",
        labels: Optional[List[ProblemLabel]] = None,
        max_samples: int = 10000,
        normalize: bool = True,
        pad_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Export dataset for ML training.
        
        Args:
            output_path: Output file path
            format: 'numpy', 'hdf5', 'csv', or 'tfrecord'
            labels: Filter by labels (None = all)
            max_samples: Maximum samples per label
            normalize: Normalize waveforms
            pad_length: Pad/truncate to this length
        
        Returns:
            Export statistics
        """
        output_path = Path(output_path).expanduser()
        
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build query
        if labels:
            label_list = ','.join(f"'{l.value}'" for l in labels)
            query = f"""
                SELECT id, primary_label, hdf5_index 
                FROM samples 
                WHERE primary_label IN ({label_list})
                AND confidence != 'unlabeled'
            """
        else:
            query = """
                SELECT id, primary_label, hdf5_index 
                FROM samples
                WHERE confidence != 'unlabeled'
            """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        
        # Collect data
        X = []
        y = []
        label_map = {l.value: i for i, l in enumerate(ProblemLabel)}
        
        # Balance by label
        label_counts = {}
        
        for sample_id, label, hdf5_idx in rows:
            if label not in label_counts:
                label_counts[label] = 0
            
            if label_counts[label] >= max_samples:
                continue
            
            samples = self._load_waveform(sample_id, hdf5_idx)
            
            if len(samples) == 0:
                continue
            
            # Normalize
            if normalize:
                samples = (samples - np.mean(samples)) / (np.std(samples) + 1e-10)
            
            # Pad/truncate
            if pad_length:
                if len(samples) > pad_length:
                    samples = samples[:pad_length]
                elif len(samples) < pad_length:
                    samples = np.pad(samples, (0, pad_length - len(samples)))
            
            X.append(samples)
            y.append(label_map[label])
            label_counts[label] += 1
        
        X = np.array(X, dtype=object) if not pad_length else np.array(X)
        y = np.array(y)
        
        # Export
        if format == "numpy":
            np.savez(output_path, X=X, y=y, label_map=label_map)
        
        elif format == "hdf5" and HAS_HDF5:
            with h5py.File(output_path, 'w') as f:
                if pad_length:
                    f.create_dataset('X', data=X)
                else:
                    f.create_dataset('X', data=X, dtype=h5py.vlen_dtype(np.float32))
                f.create_dataset('y', data=y)
                f.attrs['label_map'] = json.dumps(label_map)
        
        elif format == "csv":
            import csv
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['label'] + [f's{i}' for i in range(pad_length or 1000)])
                for xi, yi in zip(X, y):
                    writer.writerow([yi] + list(xi)[:pad_length or 1000])
        
        stats = {
            'total_exported': len(y),
            'labels': label_counts,
            'format': format,
            'path': str(output_path)
        }
        
        logger.info(f"Exported {len(y)} samples to {output_path}")
        return stats
    
    def create_version(self, notes: str = "") -> int:
        """Create a dataset version snapshot"""
        stats = self.get_stats()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO dataset_versions (created_at, sample_count, labeled_count, notes)
            VALUES (?, ?, ?, ?)
        """, (time.time(), stats.total_samples, stats.labeled_samples, notes))
        
        version = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Created dataset version {version}")
        return version


# =============================================================================
# Active Learning
# =============================================================================

class ActiveLearner:
    """
    Active learning system for efficient labeling.
    
    Prioritizes samples that would most improve the model.
    """
    
    def __init__(self, dataset: DatasetManager):
        self.dataset = dataset
        self.model = None
        self.uncertainty_threshold = 0.7
    
    def set_model(self, model):
        """Set the classifier model for uncertainty estimation"""
        self.model = model
    
    def get_samples_for_labeling(self, count: int = 10) -> List[Tuple[str, float]]:
        """
        Get samples that would most benefit from labeling.
        
        Uses uncertainty sampling - returns samples where
        the model is least confident.
        
        Returns:
            List of (sample_id, uncertainty_score)
        """
        # Get unlabeled samples
        unlabeled = self.dataset.get_unlabeled(limit=1000)
        
        if not unlabeled:
            return []
        
        if self.model is None:
            # No model - return random samples
            import random
            random.shuffle(unlabeled)
            return [(sid, 0.5) for sid in unlabeled[:count]]
        
        # Score each sample by uncertainty
        scored = []
        for sample_id in unlabeled:
            sample = self.dataset.get_sample(sample_id)
            if sample is None:
                continue
            
            try:
                # Get prediction probabilities
                probs = self.model.predict_proba(sample.samples)
                
                # Uncertainty = 1 - max probability
                uncertainty = 1 - np.max(probs)
                scored.append((sample_id, uncertainty))
            except:
                scored.append((sample_id, 0.5))
        
        # Sort by uncertainty (highest first)
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored[:count]
    
    def process_feedback(
        self,
        sample_id: str,
        correct_label: ProblemLabel,
        was_prediction_correct: bool
    ):
        """
        Process user feedback on a prediction.
        
        Updates the dataset with verified label.
        """
        confidence = ConfidenceLevel.VERIFIED
        
        self.dataset.label_sample(
            sample_id,
            correct_label,
            confidence,
            annotator="user_feedback"
        )
        
        logger.info(f"Feedback processed for {sample_id}: {correct_label.value}")


# =============================================================================
# Main Pipeline
# =============================================================================

class DataPipeline:
    """
    Main data collection pipeline.
    
    Orchestrates:
    - ESP32 collection
    - Oscilloscope capture
    - Dataset management
    - Active learning
    """
    
    def __init__(self, data_dir: str = "~/.waveformgpt/data"):
        self.dataset = DatasetManager(data_dir)
        self.esp32: Optional[ESP32Collector] = None
        self.oscilloscope: Optional[SCPIOscilloscope] = None
        self.active_learner = ActiveLearner(self.dataset)
        
        # Feature extractor for auto-labeling
        self._feature_extractor = None
    
    def connect_esp32(self, port: str = "/dev/cu.usbserial-0001") -> bool:
        """Connect to ESP32 for data collection"""
        self.esp32 = ESP32Collector(port)
        return self.esp32.connect()
    
    def connect_oscilloscope(self, address: str, port: int = 5555) -> bool:
        """Connect to oscilloscope"""
        self.oscilloscope = SCPIOscilloscope(address, port)
        return self.oscilloscope.connect()
    
    def start_collection(
        self,
        source: str = "esp32",
        interval_ms: int = 1000,
        auto_label: bool = True
    ):
        """
        Start continuous data collection.
        
        Args:
            source: 'esp32' or 'oscilloscope'
            interval_ms: Collection interval
            auto_label: Try to auto-label using existing model
        """
        def on_sample(sample: WaveformSample):
            # Extract features
            sample.features = self._extract_features(sample.samples)
            
            # Auto-label if possible
            if auto_label and self._feature_extractor:
                try:
                    label, confidence = self._auto_label(sample)
                    sample.primary_label = label
                    sample.confidence = confidence
                except:
                    pass
            
            # Add to dataset
            self.dataset.add_sample(sample)
        
        if source == "esp32" and self.esp32:
            self.esp32.capture_continuous(on_sample, interval_ms)
        elif source == "oscilloscope" and self.oscilloscope:
            # Oscilloscope collection would go here
            pass
    
    def stop_collection(self):
        """Stop data collection"""
        if self.esp32:
            self.esp32.stop_continuous()
    
    def _extract_features(self, samples: np.ndarray) -> Dict[str, float]:
        """Extract features from waveform"""
        if len(samples) == 0:
            return {}
        
        return {
            'mean': float(np.mean(samples)),
            'std': float(np.std(samples)),
            'min': float(np.min(samples)),
            'max': float(np.max(samples)),
            'peak_to_peak': float(np.max(samples) - np.min(samples)),
            'rms': float(np.sqrt(np.mean(samples**2))),
            'crest_factor': float(np.max(np.abs(samples)) / (np.sqrt(np.mean(samples**2)) + 1e-10)),
            'zero_crossings': float(np.sum(np.diff(np.sign(samples)) != 0)),
        }
    
    def _auto_label(self, sample: WaveformSample) -> Tuple[ProblemLabel, ConfidenceLevel]:
        """Attempt to auto-label using heuristics"""
        features = sample.features
        
        # Simple rule-based auto-labeling
        p2p = features.get('peak_to_peak', 0)
        std = features.get('std', 0)
        crest = features.get('crest_factor', 0)
        
        # Clipped detection
        if crest > 5.0:
            return ProblemLabel.CLIPPED, ConfidenceLevel.MEDIUM
        
        # Noisy detection
        if std > 0.3:
            return ProblemLabel.NOISY, ConfidenceLevel.MEDIUM
        
        # Clean signal
        if std < 0.1 and crest < 2.0:
            return ProblemLabel.CLEAN, ConfidenceLevel.LOW
        
        return ProblemLabel.UNKNOWN, ConfidenceLevel.UNLABELED
    
    def import_from_file(self, file_path: str, label: Optional[ProblemLabel] = None) -> int:
        """
        Import waveforms from file.
        
        Supports: .npy, .csv, .wav
        
        Returns:
            Number of samples imported
        """
        file_path = Path(file_path)
        count = 0
        
        if file_path.suffix == '.npy':
            data = np.load(file_path)
            if data.ndim == 1:
                data = [data]
            
            for samples in data:
                sample = WaveformSample(
                    id="",
                    samples=samples,
                    sample_rate=44100.0,
                    source=WaveformSource.UPLOAD,
                    source_device=str(file_path),
                    primary_label=label or ProblemLabel.UNKNOWN
                )
                if self.dataset.add_sample(sample):
                    count += 1
        
        elif file_path.suffix == '.csv':
            import csv
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    samples = np.array([float(v) for v in row if v])
                    sample = WaveformSample(
                        id="",
                        samples=samples,
                        sample_rate=44100.0,
                        source=WaveformSource.UPLOAD,
                        source_device=str(file_path),
                        primary_label=label or ProblemLabel.UNKNOWN
                    )
                    if self.dataset.add_sample(sample):
                        count += 1
        
        elif file_path.suffix == '.wav':
            from scipy.io import wavfile
            sr, samples = wavfile.read(file_path)
            samples = samples.astype(np.float32) / 32768.0
            
            sample = WaveformSample(
                id="",
                samples=samples,
                sample_rate=float(sr),
                source=WaveformSource.UPLOAD,
                source_device=str(file_path),
                primary_label=label or ProblemLabel.UNKNOWN
            )
            if self.dataset.add_sample(sample):
                count = 1
        
        logger.info(f"Imported {count} samples from {file_path}")
        return count
    
    def get_stats(self) -> DatasetStats:
        """Get dataset statistics"""
        return self.dataset.get_stats()


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("WaveformGPT Data Pipeline - Demo")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = DataPipeline("~/.waveformgpt/demo_data")
    
    # Generate some synthetic samples for demo
    print("\n📊 Generating demo samples...")
    
    for i in range(50):
        t = np.linspace(0, 0.01, 1000)
        
        # Random waveform type
        waveform_type = np.random.choice(['clean', 'noisy', 'clipped', 'ringing'])
        
        if waveform_type == 'clean':
            samples = np.sin(2 * np.pi * 1000 * t)
            label = ProblemLabel.CLEAN
        elif waveform_type == 'noisy':
            samples = np.sin(2 * np.pi * 1000 * t) + 0.3 * np.random.randn(len(t))
            label = ProblemLabel.NOISY
        elif waveform_type == 'clipped':
            samples = np.clip(1.5 * np.sin(2 * np.pi * 1000 * t), -1, 1)
            label = ProblemLabel.CLIPPED
        else:  # ringing
            samples = np.sin(2 * np.pi * 1000 * t) * (1 + 0.2 * np.sin(2 * np.pi * 5000 * t))
            label = ProblemLabel.RINGING
        
        sample = WaveformSample(
            id="",
            samples=samples,
            sample_rate=100000.0,
            source=WaveformSource.SYNTHETIC,
            primary_label=label,
            confidence=ConfidenceLevel.VERIFIED
        )
        sample.features = pipeline._extract_features(samples)
        pipeline.dataset.add_sample(sample)
    
    # Show stats
    print("\n📈 Dataset Statistics:")
    stats = pipeline.get_stats()
    print(f"   Total samples: {stats.total_samples}")
    print(f"   Labeled: {stats.labeled_samples}")
    print(f"   Verified: {stats.verified_samples}")
    print(f"\n   By label:")
    for label, count in stats.samples_by_label.items():
        print(f"      {label}: {count}")
    
    # Export for training
    print("\n📦 Exporting for training...")
    export_stats = pipeline.dataset.export_for_training(
        "~/.waveformgpt/demo_data/training_data.npz",
        format="numpy",
        pad_length=1000
    )
    print(f"   Exported: {export_stats['total_exported']} samples")
    print(f"   Path: {export_stats['path']}")
    
    # Active learning demo
    print("\n🎯 Active Learning:")
    unlabeled = pipeline.dataset.get_unlabeled(5)
    print(f"   Unlabeled samples needing review: {len(unlabeled)}")
    
    # Version the dataset
    version = pipeline.dataset.create_version("Demo dataset v1")
    print(f"\n📌 Created dataset version: {version}")
    
    print("\n✅ Demo complete!")
    print(f"\nData stored at: ~/.waveformgpt/demo_data/")
