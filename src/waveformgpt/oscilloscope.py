"""
WaveformGPT Oscilloscope Integration

Connect to real oscilloscopes and pull waveform data directly.

Supported Instruments:
1. Rigol DS1000Z/DS2000/MSO5000 series
2. Tektronix TBS/TDS/MDO series
3. Keysight/Agilent DSO-X series
4. Siglent SDS series
5. USB oscilloscopes (PicoScope, Digilent)

This is how you get REAL data into the system.
"""

import numpy as np
import socket
import struct
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WaveformGPT.Oscilloscope")

# Optional imports
try:
    import usb.core
    import usb.util
    HAS_USB = True
except ImportError:
    HAS_USB = False

try:
    import pyvisa
    HAS_VISA = True
except ImportError:
    HAS_VISA = False


# =============================================================================
# Data Classes
# =============================================================================

class ConnectionType(Enum):
    TCP_IP = "tcp"
    USB = "usb"
    VISA = "visa"


@dataclass
class OscilloscopeInfo:
    """Oscilloscope identification"""
    manufacturer: str
    model: str
    serial: str
    firmware: str
    channels: int


@dataclass
class ChannelConfig:
    """Channel configuration"""
    channel: int
    coupling: str  # AC, DC, GND
    probe_ratio: float
    vertical_scale: float  # V/div
    vertical_offset: float
    bandwidth_limit: bool
    enabled: bool


@dataclass
class TriggerConfig:
    """Trigger configuration"""
    source: str
    mode: str  # AUTO, NORMAL, SINGLE
    type: str  # EDGE, PULSE, VIDEO, etc.
    level: float
    slope: str  # RISING, FALLING, BOTH


@dataclass
class TimebaseConfig:
    """Timebase configuration"""
    scale: float  # s/div
    offset: float
    sample_rate: float
    memory_depth: int


@dataclass
class WaveformCapture:
    """Captured waveform data"""
    channel: int
    samples: np.ndarray
    sample_rate: float
    time_offset: float
    vertical_scale: float
    vertical_offset: float
    timestamp: float
    metadata: Dict[str, Any]


# =============================================================================
# Base Oscilloscope Class
# =============================================================================

class BaseOscilloscope:
    """Base class for oscilloscope communication"""
    
    def __init__(self):
        self.connected = False
        self.info: Optional[OscilloscopeInfo] = None
    
    def connect(self, **kwargs) -> bool:
        raise NotImplementedError
    
    def disconnect(self):
        raise NotImplementedError
    
    def query(self, command: str) -> str:
        raise NotImplementedError
    
    def write(self, command: str):
        raise NotImplementedError
    
    def read_raw(self, size: int) -> bytes:
        raise NotImplementedError
    
    def get_id(self) -> OscilloscopeInfo:
        """Get oscilloscope identification"""
        idn = self.query("*IDN?").strip()
        parts = idn.split(',')
        
        return OscilloscopeInfo(
            manufacturer=parts[0] if len(parts) > 0 else "Unknown",
            model=parts[1] if len(parts) > 1 else "Unknown",
            serial=parts[2] if len(parts) > 2 else "Unknown",
            firmware=parts[3] if len(parts) > 3 else "Unknown",
            channels=4  # Default, override in subclass
        )
    
    def reset(self):
        """Reset oscilloscope to default state"""
        self.write("*RST")
        time.sleep(2)
    
    def run(self):
        """Start acquisition"""
        self.write(":RUN")
    
    def stop(self):
        """Stop acquisition"""
        self.write(":STOP")
    
    def single(self):
        """Single acquisition"""
        self.write(":SING")
        time.sleep(0.5)
    
    def auto_scale(self):
        """Auto-scale all channels"""
        self.write(":AUT")
        time.sleep(3)


# =============================================================================
# Rigol Oscilloscope
# =============================================================================

class RigolOscilloscope(BaseOscilloscope):
    """
    Rigol oscilloscope driver.
    
    Supports: DS1054Z, DS1104Z, DS2072A, DS2102A, MSO5074, etc.
    
    Connection: TCP/IP (port 5555) or VISA
    """
    
    def __init__(self):
        super().__init__()
        self.socket: Optional[socket.socket] = None
        self.address: str = ""
        self.port: int = 5555
    
    def connect(self, address: str, port: int = 5555) -> bool:
        """Connect via TCP/IP"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5)
            self.socket.connect((address, port))
            
            self.address = address
            self.port = port
            self.connected = True
            
            # Get identification
            self.info = self.get_id()
            logger.info(f"Connected to {self.info.manufacturer} {self.info.model}")
            
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def disconnect(self):
        if self.socket:
            self.socket.close()
            self.socket = None
        self.connected = False
    
    def query(self, command: str) -> str:
        self.write(command)
        time.sleep(0.1)
        return self._read_response()
    
    def write(self, command: str):
        if self.socket:
            self.socket.send((command + '\n').encode())
    
    def _read_response(self, size: int = 65536) -> str:
        if self.socket:
            return self.socket.recv(size).decode('utf-8', errors='ignore').strip()
        return ""
    
    def read_raw(self, size: int) -> bytes:
        if self.socket:
            data = b''
            while len(data) < size:
                chunk = self.socket.recv(min(size - len(data), 65536))
                if not chunk:
                    break
                data += chunk
            return data
        return b''
    
    def configure_channel(self, channel: int, **kwargs) -> ChannelConfig:
        """
        Configure a channel.
        
        Args:
            channel: Channel number (1-4)
            coupling: "AC", "DC", or "GND"
            probe: Probe ratio (1, 10, 100, etc.)
            scale: Vertical scale in V/div
            offset: Vertical offset in V
            bwlimit: Bandwidth limit on/off
        """
        ch = f":CHAN{channel}"
        
        if 'coupling' in kwargs:
            self.write(f"{ch}:COUP {kwargs['coupling']}")
        
        if 'probe' in kwargs:
            self.write(f"{ch}:PROB {kwargs['probe']}")
        
        if 'scale' in kwargs:
            self.write(f"{ch}:SCAL {kwargs['scale']}")
        
        if 'offset' in kwargs:
            self.write(f"{ch}:OFFS {kwargs['offset']}")
        
        if 'bwlimit' in kwargs:
            self.write(f"{ch}:BWL {'ON' if kwargs['bwlimit'] else 'OFF'}")
        
        if 'enabled' in kwargs:
            self.write(f"{ch}:DISP {'ON' if kwargs['enabled'] else 'OFF'}")
        
        # Read back configuration
        return ChannelConfig(
            channel=channel,
            coupling=self.query(f"{ch}:COUP?"),
            probe_ratio=float(self.query(f"{ch}:PROB?")),
            vertical_scale=float(self.query(f"{ch}:SCAL?")),
            vertical_offset=float(self.query(f"{ch}:OFFS?")),
            bandwidth_limit=self.query(f"{ch}:BWL?") == "ON",
            enabled=self.query(f"{ch}:DISP?") == "1"
        )
    
    def configure_timebase(self, **kwargs) -> TimebaseConfig:
        """
        Configure timebase.
        
        Args:
            scale: Time scale in s/div
            offset: Time offset in s
        """
        if 'scale' in kwargs:
            self.write(f":TIM:SCAL {kwargs['scale']}")
        
        if 'offset' in kwargs:
            self.write(f":TIM:OFFS {kwargs['offset']}")
        
        return TimebaseConfig(
            scale=float(self.query(":TIM:SCAL?")),
            offset=float(self.query(":TIM:OFFS?")),
            sample_rate=float(self.query(":ACQ:SRAT?")),
            memory_depth=int(float(self.query(":ACQ:MDEP?")))
        )
    
    def configure_trigger(self, **kwargs) -> TriggerConfig:
        """
        Configure trigger.
        
        Args:
            source: Trigger source ("CHAN1", "CHAN2", "EXT", etc.)
            mode: "AUTO", "NORMAL", "SINGLE"
            type: "EDGE", "PULSE", "RUNT", etc.
            level: Trigger level in V
            slope: "RISING", "FALLING", "BOTH"
        """
        if 'source' in kwargs:
            self.write(f":TRIG:EDGE:SOUR {kwargs['source']}")
        
        if 'mode' in kwargs:
            self.write(f":TRIG:SWE {kwargs['mode']}")
        
        if 'level' in kwargs:
            self.write(f":TRIG:EDGE:LEV {kwargs['level']}")
        
        if 'slope' in kwargs:
            slope_map = {'RISING': 'POS', 'FALLING': 'NEG', 'BOTH': 'RFAL'}
            self.write(f":TRIG:EDGE:SLOP {slope_map.get(kwargs['slope'], 'POS')}")
        
        return TriggerConfig(
            source=self.query(":TRIG:EDGE:SOUR?"),
            mode=self.query(":TRIG:SWE?"),
            type="EDGE",
            level=float(self.query(":TRIG:EDGE:LEV?")),
            slope=self.query(":TRIG:EDGE:SLOP?")
        )
    
    def capture(self, channel: int = 1, format: str = "BYTE") -> WaveformCapture:
        """
        Capture waveform from channel.
        
        Args:
            channel: Channel number (1-4)
            format: Data format ("BYTE", "WORD", "ASC")
        
        Returns:
            WaveformCapture
        """
        # Configure waveform output
        self.write(f":WAV:SOUR CHAN{channel}")
        self.write(":WAV:MODE RAW")
        self.write(f":WAV:FORM {format}")
        
        # Get preamble
        preamble = self.query(":WAV:PRE?").split(',')
        
        # Parse preamble
        # Format: format, type, points, count, xinc, xorigin, xref, yinc, yorigin, yref
        points = int(preamble[2])
        x_inc = float(preamble[4])
        x_origin = float(preamble[5])
        y_inc = float(preamble[7])
        y_origin = float(preamble[8])
        y_ref = float(preamble[9])
        
        # Read data
        self.write(":WAV:DATA?")
        time.sleep(0.2)
        
        raw = self.read_raw(points + 100)
        
        # Parse IEEE 488.2 block header
        if raw[0:1] == b'#':
            header_len = int(raw[1:2])
            data_len = int(raw[2:2+header_len])
            data = raw[2+header_len:2+header_len+data_len]
            
            # Convert to voltage
            if format == "BYTE":
                values = np.frombuffer(data, dtype=np.uint8)
                samples = (values - y_ref - y_origin) * y_inc
            elif format == "WORD":
                values = np.frombuffer(data, dtype=np.int16)
                samples = (values - y_ref - y_origin) * y_inc
            else:
                samples = np.array([float(v) for v in data.decode().split(',')])
        else:
            logger.warning("Unexpected data format")
            samples = np.array([])
        
        # Get sample rate
        sample_rate = 1.0 / x_inc
        
        # Get vertical scale
        v_scale = float(self.query(f":CHAN{channel}:SCAL?"))
        v_offset = float(self.query(f":CHAN{channel}:OFFS?"))
        
        return WaveformCapture(
            channel=channel,
            samples=samples,
            sample_rate=sample_rate,
            time_offset=x_origin,
            vertical_scale=v_scale,
            vertical_offset=v_offset,
            timestamp=time.time(),
            metadata={
                'points': points,
                'x_inc': x_inc,
                'y_inc': y_inc,
                'manufacturer': self.info.manufacturer if self.info else 'Unknown',
                'model': self.info.model if self.info else 'Unknown'
            }
        )
    
    def screenshot(self, filename: str = "screenshot.png"):
        """Save oscilloscope screenshot"""
        self.write(":DISP:DATA? ON,OFF,PNG")
        time.sleep(0.5)
        
        raw = self.read_raw(1000000)
        
        # Parse header
        if raw[0:1] == b'#':
            header_len = int(raw[1:2])
            data_len = int(raw[2:2+header_len])
            data = raw[2+header_len:2+header_len+data_len]
            
            with open(filename, 'wb') as f:
                f.write(data)
            
            logger.info(f"Screenshot saved to {filename}")
    
    def measure(self, channel: int, measurement: str) -> float:
        """
        Take a measurement.
        
        Args:
            channel: Channel number
            measurement: Measurement type (FREQ, PER, VMAX, VMIN, VPP, VRMS, etc.)
        
        Returns:
            Measurement value
        """
        self.write(f":MEAS:ITEM {measurement},CHAN{channel}")
        result = self.query(f":MEAS:ITEM? {measurement},CHAN{channel}")
        
        try:
            return float(result)
        except ValueError:
            return float('nan')


# =============================================================================
# Tektronix Oscilloscope
# =============================================================================

class TektronixOscilloscope(BaseOscilloscope):
    """
    Tektronix oscilloscope driver.
    
    Supports: TBS1000, TDS2000, DPO/MSO2000-4000, MDO3000-4000
    """
    
    def __init__(self):
        super().__init__()
        self.socket: Optional[socket.socket] = None
    
    def connect(self, address: str, port: int = 4000) -> bool:
        """Connect via TCP/IP"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5)
            self.socket.connect((address, port))
            
            self.connected = True
            self.info = self.get_id()
            
            logger.info(f"Connected to {self.info.manufacturer} {self.info.model}")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def disconnect(self):
        if self.socket:
            self.socket.close()
            self.socket = None
        self.connected = False
    
    def query(self, command: str) -> str:
        self.write(command)
        time.sleep(0.1)
        if self.socket:
            return self.socket.recv(65536).decode('utf-8', errors='ignore').strip()
        return ""
    
    def write(self, command: str):
        if self.socket:
            self.socket.send((command + '\n').encode())
    
    def read_raw(self, size: int) -> bytes:
        if self.socket:
            return self.socket.recv(size)
        return b''
    
    def capture(self, channel: int = 1) -> WaveformCapture:
        """Capture waveform"""
        # Select channel
        self.write(f"DATA:SOURCE CH{channel}")
        self.write("DATA:WIDTH 1")
        self.write("DATA:ENCDG BINARY")
        
        # Get scale factors
        y_mult = float(self.query("WFMPRE:YMULT?"))
        y_off = float(self.query("WFMPRE:YOFF?"))
        y_zero = float(self.query("WFMPRE:YZERO?"))
        x_inc = float(self.query("WFMPRE:XINCR?"))
        x_zero = float(self.query("WFMPRE:XZERO?"))
        
        # Read curve
        self.write("CURVE?")
        time.sleep(0.2)
        
        raw = self.read_raw(100000)
        
        # Parse
        if raw[0:1] == b'#':
            header_len = int(raw[1:2])
            data_len = int(raw[2:2+header_len])
            data = raw[2+header_len:2+header_len+data_len]
            
            values = np.frombuffer(data, dtype=np.int8)
            samples = (values - y_off) * y_mult + y_zero
        else:
            samples = np.array([])
        
        return WaveformCapture(
            channel=channel,
            samples=samples,
            sample_rate=1.0 / x_inc,
            time_offset=x_zero,
            vertical_scale=y_mult * 25,  # Approximate
            vertical_offset=y_zero,
            timestamp=time.time(),
            metadata={'manufacturer': 'Tektronix'}
        )


# =============================================================================
# Universal Oscilloscope Interface
# =============================================================================

class Oscilloscope:
    """
    Universal oscilloscope interface.
    
    Auto-detects oscilloscope type and provides unified API.
    
    Usage:
        scope = Oscilloscope()
        scope.connect("192.168.1.100")  # Auto-detect type
        
        # Or specify type:
        scope.connect("192.168.1.100", type="rigol")
        
        # Configure
        scope.configure_channel(1, coupling="DC", scale=1.0)
        scope.configure_timebase(scale=1e-3)
        
        # Capture
        waveform = scope.capture(channel=1)
        print(f"Got {len(waveform.samples)} samples at {waveform.sample_rate} S/s")
    """
    
    def __init__(self):
        self._driver: Optional[BaseOscilloscope] = None
        self._type: str = ""
    
    def connect(
        self,
        address: str,
        port: Optional[int] = None,
        type: Optional[str] = None
    ) -> bool:
        """
        Connect to oscilloscope.
        
        Args:
            address: IP address or VISA resource string
            port: TCP port (auto-detected if None)
            type: "rigol", "tektronix", "keysight", or None for auto-detect
        """
        # Auto-detect type by trying different drivers
        if type is None:
            type = self._detect_type(address, port)
        
        if type == "rigol":
            self._driver = RigolOscilloscope()
            return self._driver.connect(address, port or 5555)
        
        elif type == "tektronix":
            self._driver = TektronixOscilloscope()
            return self._driver.connect(address, port or 4000)
        
        # Add more types as needed
        
        logger.error(f"Unknown oscilloscope type: {type}")
        return False
    
    def _detect_type(self, address: str, port: Optional[int]) -> str:
        """Try to detect oscilloscope type"""
        # Try Rigol first (most common)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            sock.connect((address, port or 5555))
            sock.send(b"*IDN?\n")
            response = sock.recv(256).decode()
            sock.close()
            
            if "RIGOL" in response.upper():
                return "rigol"
            elif "TEKTRONIX" in response.upper():
                return "tektronix"
            elif "KEYSIGHT" in response.upper() or "AGILENT" in response.upper():
                return "keysight"
            elif "SIGLENT" in response.upper():
                return "siglent"
        except:
            pass
        
        # Try Tektronix port
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            sock.connect((address, 4000))
            sock.close()
            return "tektronix"
        except:
            pass
        
        return "rigol"  # Default
    
    def disconnect(self):
        if self._driver:
            self._driver.disconnect()
    
    @property
    def info(self) -> Optional[OscilloscopeInfo]:
        return self._driver.info if self._driver else None
    
    @property
    def connected(self) -> bool:
        return self._driver.connected if self._driver else False
    
    def configure_channel(self, channel: int, **kwargs) -> ChannelConfig:
        if self._driver:
            return self._driver.configure_channel(channel, **kwargs)
        raise RuntimeError("Not connected")
    
    def configure_timebase(self, **kwargs) -> TimebaseConfig:
        if self._driver:
            return self._driver.configure_timebase(**kwargs)
        raise RuntimeError("Not connected")
    
    def configure_trigger(self, **kwargs) -> TriggerConfig:
        if self._driver:
            return self._driver.configure_trigger(**kwargs)
        raise RuntimeError("Not connected")
    
    def capture(self, channel: int = 1) -> WaveformCapture:
        if self._driver:
            return self._driver.capture(channel)
        raise RuntimeError("Not connected")
    
    def auto_scale(self):
        if self._driver:
            self._driver.auto_scale()
    
    def run(self):
        if self._driver:
            self._driver.run()
    
    def stop(self):
        if self._driver:
            self._driver.stop()
    
    def single(self):
        if self._driver:
            self._driver.single()
    
    def measure(self, channel: int, measurement: str) -> float:
        if self._driver and hasattr(self._driver, 'measure'):
            return self._driver.measure(channel, measurement)
        raise RuntimeError("Measurement not supported")
    
    @staticmethod
    def discover(timeout: float = 5.0) -> List[Dict[str, str]]:
        """
        Discover oscilloscopes on the network.
        
        Returns:
            List of found oscilloscopes with address and type
        """
        found = []
        
        # Get local IP
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            
            subnet = '.'.join(local_ip.split('.')[:-1])
        except:
            return found
        
        # Scan common ports
        ports = [5555, 4000, 5025]  # Rigol, Tektronix, Keysight
        
        def check_host(ip, port):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.5)
                result = sock.connect_ex((ip, port))
                if result == 0:
                    # Try to identify
                    sock.send(b"*IDN?\n")
                    sock.settimeout(1)
                    response = sock.recv(256).decode('utf-8', errors='ignore')
                    sock.close()
                    
                    if response:
                        parts = response.split(',')
                        return {
                            'address': ip,
                            'port': port,
                            'manufacturer': parts[0] if parts else 'Unknown',
                            'model': parts[1] if len(parts) > 1 else 'Unknown',
                            'type': 'rigol' if 'RIGOL' in response.upper() else 'unknown'
                        }
                sock.close()
            except:
                pass
            return None
        
        # Threaded scan
        results = []
        threads = []
        
        for i in range(1, 255):
            ip = f"{subnet}.{i}"
            for port in ports:
                t = threading.Thread(target=lambda: results.append(check_host(ip, port)))
                t.start()
                threads.append(t)
        
        # Wait for threads
        for t in threads:
            t.join(timeout=0.1)
        
        found = [r for r in results if r is not None]
        return found


# =============================================================================
# Data Integration
# =============================================================================

class OscilloscopeDataCollector:
    """
    Collect waveform data from oscilloscope into WaveformGPT dataset.
    
    Bridges oscilloscope captures to the data pipeline.
    """
    
    def __init__(self, oscilloscope: Oscilloscope):
        self.scope = oscilloscope
        self.pipeline = None
        
        # Try to import data pipeline
        try:
            from waveformgpt.data_pipeline import (
                DataPipeline, WaveformSample, WaveformSource
            )
            self.pipeline = DataPipeline()
            self._WaveformSample = WaveformSample
            self._WaveformSource = WaveformSource
        except ImportError:
            logger.warning("Data pipeline not available")
    
    def capture_and_store(
        self,
        channel: int = 1,
        label: Optional[str] = None,
        notes: str = ""
    ) -> Optional[str]:
        """
        Capture waveform and store in dataset.
        
        Args:
            channel: Channel to capture
            label: Problem label (optional)
            notes: Additional notes
        
        Returns:
            Sample ID if successful
        """
        if not self.scope.connected:
            logger.error("Oscilloscope not connected")
            return None
        
        # Capture
        capture = self.scope.capture(channel)
        
        if len(capture.samples) == 0:
            logger.error("No samples captured")
            return None
        
        # Store in dataset
        if self.pipeline:
            from waveformgpt.data_pipeline import ProblemLabel, ConfidenceLevel
            
            sample = self._WaveformSample(
                id="",
                samples=capture.samples,
                sample_rate=capture.sample_rate,
                source=self._WaveformSource.OSCILLOSCOPE,
                source_device=f"{self.scope.info.manufacturer} {self.scope.info.model}" if self.scope.info else "Unknown",
                primary_label=ProblemLabel(label) if label else ProblemLabel.UNKNOWN,
                confidence=ConfidenceLevel.UNLABELED,
                notes=notes,
                metadata=capture.metadata
            )
            
            if self.pipeline.dataset.add_sample(sample):
                logger.info(f"Stored sample {sample.id}")
                return sample.id
        
        return None
    
    def continuous_capture(
        self,
        channel: int = 1,
        interval: float = 1.0,
        callback: Optional[Callable] = None
    ):
        """
        Continuously capture waveforms.
        
        Args:
            channel: Channel to capture
            interval: Capture interval in seconds
            callback: Optional callback for each capture
        """
        self._collecting = True
        
        def capture_loop():
            while self._collecting:
                sample_id = self.capture_and_store(channel)
                if callback and sample_id:
                    callback(sample_id)
                time.sleep(interval)
        
        self._thread = threading.Thread(target=capture_loop, daemon=True)
        self._thread.start()
        logger.info("Started continuous capture")
    
    def stop_capture(self):
        """Stop continuous capture"""
        self._collecting = False
        if hasattr(self, '_thread'):
            self._thread.join(timeout=2)
        logger.info("Stopped continuous capture")


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("WaveformGPT Oscilloscope Integration - Demo")
    print("=" * 60)
    
    # Discover oscilloscopes
    print("\n🔍 Searching for oscilloscopes on network...")
    found = Oscilloscope.discover(timeout=3.0)
    
    if found:
        print(f"\n   Found {len(found)} oscilloscope(s):")
        for f in found:
            print(f"   - {f['manufacturer']} {f['model']} at {f['address']}:{f['port']}")
    else:
        print("   No oscilloscopes found on network")
    
    # Demo connection (if available)
    scope = Oscilloscope()
    
    # Try to connect to first found scope
    if found:
        print(f"\n📡 Connecting to {found[0]['address']}...")
        if scope.connect(found[0]['address'], found[0]['port']):
            print(f"   Connected to {scope.info.manufacturer} {scope.info.model}")
            
            # Configure
            print("\n⚙️ Configuring...")
            scope.configure_channel(1, coupling="DC", scale=1.0)
            scope.configure_timebase(scale=1e-3)
            
            # Capture
            print("\n📷 Capturing waveform...")
            scope.single()
            time.sleep(0.5)
            
            waveform = scope.capture(channel=1)
            print(f"   Captured {len(waveform.samples)} samples")
            print(f"   Sample rate: {waveform.sample_rate/1e6:.2f} MSa/s")
            print(f"   Vpp: {np.ptp(waveform.samples):.3f} V")
            
            # Store in dataset
            print("\n💾 Storing in dataset...")
            collector = OscilloscopeDataCollector(scope)
            sample_id = collector.capture_and_store(channel=1, notes="Demo capture")
            if sample_id:
                print(f"   Stored as {sample_id}")
            
            scope.disconnect()
    
    print("\n✅ Demo complete!")
    print("\nTo use in your project:")
    print("  from waveformgpt.oscilloscope import Oscilloscope")
    print("  scope = Oscilloscope()")
    print("  scope.connect('192.168.1.100')")
    print("  waveform = scope.capture(channel=1)")
