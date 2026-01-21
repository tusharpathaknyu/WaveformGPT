"""
Live Waveform Streaming for WaveformGPT.

Enables real-time waveform analysis from various sources:
- File watching (VCD files being written)
- Named pipes/FIFOs
- WebSocket streaming
- Logic analyzer integration (Saleae, sigrok)
- Simulation tool integration (cocotb, Verilator)

Architecture:
    ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
    │   Source     │────▶│   Stream    │────▶│  WaveformGPT │
    │ (VCD/Sim/LA) │     │   Buffer    │     │   Analysis   │
    └──────────────┘     └─────────────┘     └──────────────┘
                                │
                                ▼
                         ┌─────────────┐
                         │  Real-time  │
                         │   Alerts    │
                         └─────────────┘
"""

import os
import time
import threading
import queue
from typing import Optional, List, Dict, Any, Callable, Generator
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod
import re


@dataclass
class LiveSignalValue:
    """A single signal value update."""
    signal: str
    time: int
    value: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LiveWaveformBuffer:
    """Circular buffer for live waveform data."""
    max_samples: int = 100000
    
    def __post_init__(self):
        self._data: Dict[str, List[tuple]] = {}
        self._lock = threading.Lock()
    
    def add(self, signal: str, time: int, value: str):
        """Add a new sample."""
        with self._lock:
            if signal not in self._data:
                self._data[signal] = []
            
            self._data[signal].append((time, value))
            
            # Trim if exceeds max
            if len(self._data[signal]) > self.max_samples:
                self._data[signal] = self._data[signal][-self.max_samples:]
    
    def get_signal(self, signal: str) -> List[tuple]:
        """Get all samples for a signal."""
        with self._lock:
            return list(self._data.get(signal, []))
    
    def get_signals(self) -> List[str]:
        """Get all signal names."""
        with self._lock:
            return list(self._data.keys())
    
    def get_latest(self, signal: str) -> Optional[tuple]:
        """Get latest value for a signal."""
        with self._lock:
            data = self._data.get(signal, [])
            return data[-1] if data else None
    
    def get_time_range(self) -> tuple:
        """Get (min_time, max_time) across all signals."""
        with self._lock:
            min_t = float('inf')
            max_t = 0
            for sig_data in self._data.values():
                if sig_data:
                    min_t = min(min_t, sig_data[0][0])
                    max_t = max(max_t, sig_data[-1][0])
            return (min_t if min_t != float('inf') else 0, max_t)
    
    def clear(self):
        """Clear all data."""
        with self._lock:
            self._data.clear()


class WaveformSource(ABC):
    """Abstract base class for waveform sources."""
    
    @abstractmethod
    def start(self, callback: Callable[[str, int, str], None]):
        """Start streaming data. Call callback(signal, time, value) for each update."""
        pass
    
    @abstractmethod
    def stop(self):
        """Stop streaming."""
        pass
    
    @abstractmethod
    def is_running(self) -> bool:
        """Check if source is running."""
        pass


class VCDFileWatcher(WaveformSource):
    """
    Watch a VCD file for new data (tail -f style).
    
    Useful when simulation is writing VCD in real-time.
    """
    
    def __init__(self, filepath: str, poll_interval: float = 0.1):
        self.filepath = filepath
        self.poll_interval = poll_interval
        self._running = False
        self._thread = None
        self._last_position = 0
        self._current_time = 0
        self._signals = {}  # id -> name mapping
    
    def start(self, callback: Callable[[str, int, str], None]):
        """Start watching the file."""
        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, args=(callback,))
        self._thread.daemon = True
        self._thread.start()
    
    def stop(self):
        """Stop watching."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
    
    def is_running(self) -> bool:
        return self._running
    
    def _watch_loop(self, callback: Callable):
        """Main watch loop."""
        # First, parse existing content
        if os.path.exists(self.filepath):
            self._parse_initial(callback)
        
        # Then watch for new content
        while self._running:
            try:
                if os.path.exists(self.filepath):
                    current_size = os.path.getsize(self.filepath)
                    
                    if current_size > self._last_position:
                        self._read_new_content(callback)
                
                time.sleep(self.poll_interval)
                
            except Exception as e:
                print(f"VCD watcher error: {e}")
                time.sleep(1)
    
    def _parse_initial(self, callback: Callable):
        """Parse initial file content."""
        with open(self.filepath, 'r') as f:
            in_header = True
            
            for line in f:
                line = line.strip()
                
                if not line:
                    continue
                
                # Parse header for signal definitions
                if line.startswith('$var'):
                    parts = line.split()
                    if len(parts) >= 5:
                        var_id = parts[3]
                        var_name = parts[4]
                        self._signals[var_id] = var_name
                
                elif line.startswith('$enddefinitions'):
                    in_header = False
                
                elif not in_header:
                    # Parse value changes
                    if line.startswith('#'):
                        self._current_time = int(line[1:])
                    elif line[0] in '01xXzZ':
                        value = line[0]
                        sig_id = line[1:]
                        if sig_id in self._signals:
                            callback(self._signals[sig_id], self._current_time, value)
                    elif line[0] == 'b':
                        parts = line.split()
                        if len(parts) == 2:
                            value = parts[0][1:]  # Remove 'b'
                            sig_id = parts[1]
                            if sig_id in self._signals:
                                callback(self._signals[sig_id], self._current_time, value)
            
            self._last_position = f.tell()
    
    def _read_new_content(self, callback: Callable):
        """Read new content since last position."""
        with open(self.filepath, 'r') as f:
            f.seek(self._last_position)
            
            for line in f:
                line = line.strip()
                
                if not line:
                    continue
                
                if line.startswith('#'):
                    self._current_time = int(line[1:])
                elif line[0] in '01xXzZ':
                    value = line[0]
                    sig_id = line[1:]
                    if sig_id in self._signals:
                        callback(self._signals[sig_id], self._current_time, value)
                elif line[0] == 'b':
                    parts = line.split()
                    if len(parts) == 2:
                        value = parts[0][1:]
                        sig_id = parts[1]
                        if sig_id in self._signals:
                            callback(self._signals[sig_id], self._current_time, value)
            
            self._last_position = f.tell()


class FIFOSource(WaveformSource):
    """
    Read waveform data from a named pipe (FIFO).
    
    Protocol: Each line is "signal_name time value"
    
    Usage:
        # Terminal 1: Create FIFO and start listening
        mkfifo /tmp/waveform_pipe
        waveformgpt --live fifo:/tmp/waveform_pipe
        
        # Terminal 2: Send data
        echo "clk 100 1" > /tmp/waveform_pipe
        echo "data 100 0xFF" > /tmp/waveform_pipe
    """
    
    def __init__(self, fifo_path: str):
        self.fifo_path = fifo_path
        self._running = False
        self._thread = None
    
    def start(self, callback: Callable[[str, int, str], None]):
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, args=(callback,))
        self._thread.daemon = True
        self._thread.start()
    
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
    
    def is_running(self) -> bool:
        return self._running
    
    def _read_loop(self, callback: Callable):
        while self._running:
            try:
                with open(self.fifo_path, 'r') as fifo:
                    for line in fifo:
                        if not self._running:
                            break
                        
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            signal = parts[0]
                            time_val = int(parts[1])
                            value = parts[2]
                            callback(signal, time_val, value)
                            
            except Exception as e:
                if self._running:
                    print(f"FIFO read error: {e}")
                    time.sleep(0.5)


class WebSocketSource(WaveformSource):
    """
    Receive waveform data via WebSocket.
    
    Protocol: JSON messages {"signal": "clk", "time": 100, "value": "1"}
    
    Useful for browser-based waveform viewers or remote debugging.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self._running = False
        self._thread = None
    
    def start(self, callback: Callable[[str, int, str], None]):
        self._running = True
        self._thread = threading.Thread(target=self._server_loop, args=(callback,))
        self._thread.daemon = True
        self._thread.start()
    
    def stop(self):
        self._running = False
    
    def is_running(self) -> bool:
        return self._running
    
    def _server_loop(self, callback: Callable):
        try:
            import asyncio
            import websockets
            import json
            
            async def handler(websocket):
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        callback(
                            data["signal"],
                            int(data["time"]),
                            str(data["value"])
                        )
                    except Exception as e:
                        print(f"WebSocket message error: {e}")
            
            async def main():
                async with websockets.serve(handler, self.host, self.port):
                    print(f"📡 WebSocket server running on ws://{self.host}:{self.port}")
                    while self._running:
                        await asyncio.sleep(0.1)
            
            asyncio.run(main())
            
        except ImportError:
            raise ImportError("Install websockets: pip install websockets")


class SigrokSource(WaveformSource):
    """
    Connect to sigrok-compatible logic analyzers.
    
    Supports many USB logic analyzers like Saleae clones,
    DSLogic, FX2-based devices, etc.
    
    Requires: sigrok-cli installed
    """
    
    def __init__(self, driver: str = "fx2lafw", 
                 channels: List[str] = None,
                 sample_rate: str = "1M"):
        self.driver = driver
        self.channels = channels or ["D0", "D1", "D2", "D3"]
        self.sample_rate = sample_rate
        self._running = False
        self._process = None
    
    def start(self, callback: Callable[[str, int, str], None]):
        import subprocess
        
        self._running = True
        
        # Build sigrok command
        cmd = [
            "sigrok-cli",
            "-d", self.driver,
            "--config", f"samplerate={self.sample_rate}",
            "-C", ",".join(self.channels),
            "-O", "ascii",
            "--continuous"
        ]
        
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Start reader thread
        self._thread = threading.Thread(
            target=self._read_output, 
            args=(callback,)
        )
        self._thread.daemon = True
        self._thread.start()
    
    def stop(self):
        self._running = False
        if self._process:
            self._process.terminate()
    
    def is_running(self) -> bool:
        return self._running
    
    def _read_output(self, callback: Callable):
        sample_num = 0
        while self._running and self._process.poll() is None:
            line = self._process.stdout.readline()
            if line:
                # Parse ASCII output (binary digits)
                for i, bit in enumerate(line.strip()):
                    if i < len(self.channels):
                        callback(self.channels[i], sample_num, bit)
                sample_num += 1


@dataclass 
class LiveAlert:
    """An alert condition for live monitoring."""
    name: str
    condition: str  # e.g., "clk == 1 and data == 0xFF"
    callback: Optional[Callable] = None
    triggered_count: int = 0


class LiveWaveformAnalyzer:
    """
    Real-time waveform analyzer with alerts.
    
    Usage:
        analyzer = LiveWaveformAnalyzer()
        
        # Watch a VCD file being written
        analyzer.connect("file:/path/to/sim.vcd")
        
        # Or connect to WebSocket
        analyzer.connect("ws://localhost:8765")
        
        # Set up alerts
        analyzer.add_alert("error_detected", "error == 1")
        analyzer.add_alert("timeout", "req == 1 and ack == 0", 
                          duration=1000)  # 1000 time units
        
        # Start monitoring
        analyzer.start()
        
        # Query live data
        print(analyzer.ask("What's happening with the clock?"))
    """
    
    def __init__(self):
        self.buffer = LiveWaveformBuffer()
        self.source: Optional[WaveformSource] = None
        self.alerts: List[LiveAlert] = []
        self._running = False
        self._update_callbacks: List[Callable] = []
        self._llm_chat = None
    
    def connect(self, uri: str):
        """
        Connect to a waveform source.
        
        URIs:
            file:/path/to/file.vcd   - Watch VCD file
            fifo:/path/to/pipe       - Read from FIFO
            ws://host:port           - WebSocket server
            sigrok://driver          - Logic analyzer
        """
        if uri.startswith("file:"):
            path = uri[5:]
            self.source = VCDFileWatcher(path)
        elif uri.startswith("fifo:"):
            path = uri[5:]
            self.source = FIFOSource(path)
        elif uri.startswith("ws://"):
            parts = uri[5:].split(":")
            host = parts[0]
            port = int(parts[1]) if len(parts) > 1 else 8765
            self.source = WebSocketSource(host, port)
        elif uri.startswith("sigrok://"):
            driver = uri[9:]
            self.source = SigrokSource(driver)
        else:
            raise ValueError(f"Unknown URI scheme: {uri}")
    
    def add_alert(self, name: str, condition: str, 
                  callback: Optional[Callable] = None):
        """Add an alert condition."""
        self.alerts.append(LiveAlert(name, condition, callback))
    
    def on_update(self, callback: Callable[[str, int, str], None]):
        """Register callback for all updates."""
        self._update_callbacks.append(callback)
    
    def _handle_update(self, signal: str, time: int, value: str):
        """Handle incoming data."""
        # Store in buffer
        self.buffer.add(signal, time, value)
        
        # Check alerts
        for alert in self.alerts:
            if self._check_alert_condition(alert.condition, signal, time, value):
                alert.triggered_count += 1
                print(f"🚨 Alert '{alert.name}' triggered at t={time}: {signal}={value}")
                if alert.callback:
                    alert.callback(signal, time, value)
        
        # Notify callbacks
        for cb in self._update_callbacks:
            try:
                cb(signal, time, value)
            except:
                pass
    
    def _check_alert_condition(self, condition: str, 
                                signal: str, time: int, value: str) -> bool:
        """Evaluate alert condition."""
        # Build context with current values
        ctx = {
            signal: value,
            "time": time,
        }
        
        # Add latest values for all signals
        for sig in self.buffer.get_signals():
            latest = self.buffer.get_latest(sig)
            if latest:
                ctx[sig] = latest[1]
        
        try:
            return eval(condition, {"__builtins__": {}}, ctx)
        except:
            return False
    
    def start(self):
        """Start live monitoring."""
        if not self.source:
            raise RuntimeError("No source connected. Call connect() first.")
        
        self._running = True
        self.source.start(self._handle_update)
        print("▶️  Live monitoring started")
    
    def stop(self):
        """Stop monitoring."""
        self._running = False
        if self.source:
            self.source.stop()
        print("⏹️  Live monitoring stopped")
    
    def ask(self, question: str) -> str:
        """
        Ask a question about the live waveform data.
        Uses LLM with current buffer data as context.
        """
        if self._llm_chat is None:
            from waveformgpt.llm_engine import WaveformLLM, OpenAIBackend
            
            backend = OpenAIBackend()
            self._llm_chat = WaveformLLM(backend)
        
        # Build context from buffer
        signals = self.buffer.get_signals()
        time_range = self.buffer.get_time_range()
        
        sample_data = {}
        for sig in signals[:10]:
            sample_data[sig] = self.buffer.get_signal(sig)[:20]
        
        self._llm_chat.set_waveform_context(
            signals=signals,
            time_range=time_range,
            time_unit="ns",
            sample_data=sample_data
        )
        
        response = self._llm_chat.query(question)
        return response.answer
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        return {
            "running": self._running,
            "signals": self.buffer.get_signals(),
            "time_range": self.buffer.get_time_range(),
            "alerts_triggered": {a.name: a.triggered_count for a in self.alerts},
            "source": type(self.source).__name__ if self.source else None
        }


def create_live_session(uri: str, alerts: Optional[List[Dict]] = None):
    """
    Convenience function to create a live monitoring session.
    
    Args:
        uri: Source URI (file:/path, ws://host:port, etc.)
        alerts: List of alert dicts [{"name": "x", "condition": "y"}, ...]
    
    Returns:
        LiveWaveformAnalyzer instance
    """
    analyzer = LiveWaveformAnalyzer()
    analyzer.connect(uri)
    
    if alerts:
        for alert in alerts:
            analyzer.add_alert(alert["name"], alert["condition"])
    
    return analyzer
