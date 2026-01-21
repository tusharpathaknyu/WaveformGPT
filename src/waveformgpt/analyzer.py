"""
Waveform Statistics and Analysis.

Provides advanced statistical analysis for timing, signal activity,
and protocol verification.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import math
from collections import defaultdict

from waveformgpt.vcd_parser import VCDParser


@dataclass
class TimingStats:
    """Timing statistics for a signal or event pair."""
    count: int = 0
    min_time: float = float('inf')
    max_time: float = 0.0
    mean: float = 0.0
    std_dev: float = 0.0
    median: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    histogram: Dict[str, int] = field(default_factory=dict)
    
    def summary(self) -> str:
        """Generate text summary."""
        if self.count == 0:
            return "No measurements"
        return (
            f"Count: {self.count}\n"
            f"Min: {self.min_time:.2f}, Max: {self.max_time:.2f}\n"
            f"Mean: {self.mean:.2f}, Std: {self.std_dev:.2f}\n"
            f"Median: {self.median:.2f}\n"
            f"95th %ile: {self.p95:.2f}, 99th %ile: {self.p99:.2f}"
        )


@dataclass
class SignalActivity:
    """Activity metrics for a signal."""
    signal_name: str
    total_transitions: int = 0
    rising_edges: int = 0
    falling_edges: int = 0
    time_high: int = 0
    time_low: int = 0
    duty_cycle: float = 0.0
    toggle_rate: float = 0.0  # transitions per time unit
    avg_pulse_width_high: float = 0.0
    avg_pulse_width_low: float = 0.0


@dataclass
class GlitchInfo:
    """Information about a detected glitch."""
    signal: str
    time: int
    duration: int
    value_sequence: List[str]


@dataclass  
class ProtocolViolation:
    """Protocol violation detection result."""
    violation_type: str
    time: int
    signals: Dict[str, str]
    description: str
    severity: str = "error"  # "error", "warning", "info"


class WaveformAnalyzer:
    """
    Advanced waveform analysis engine.
    
    Provides statistical analysis, timing checks, and protocol verification.
    """
    
    def __init__(self, parser: VCDParser):
        self.parser = parser
        self._cache: Dict[str, Any] = {}
    
    def measure_timing(self, 
                        start_signal: str, 
                        start_edge: str,
                        end_signal: str,
                        end_edge: str,
                        start_time: int = None,
                        end_time: int = None) -> TimingStats:
        """
        Measure timing between two events.
        
        Args:
            start_signal: Signal name for start event
            start_edge: Edge type ("rise", "fall", "any")
            end_signal: Signal name for end event  
            end_edge: Edge type ("rise", "fall", "any")
            start_time: Optional start of measurement window
            end_time: Optional end of measurement window
        
        Returns:
            TimingStats with measurements
        """
        measurements = []
        
        # Get signal values
        start_values = self.parser.get_signal_values(start_signal)
        end_values = self.parser.get_signal_values(end_signal)
        
        if not start_values or not end_values:
            return TimingStats()
        
        # Find start events
        start_times = []
        prev_val = None
        for time, value in start_values:
            if start_time and time < start_time:
                prev_val = value
                continue
            if end_time and time > end_time:
                break
            
            if self._is_edge(prev_val, value, start_edge):
                start_times.append(time)
            prev_val = value
        
        # Find end events and measure
        prev_val = None
        end_idx = 0
        for start_t in start_times:
            # Find next end event after this start
            for i in range(end_idx, len(end_values)):
                time, value = end_values[i]
                if time <= start_t:
                    prev_val = value
                    continue
                    
                if self._is_edge(prev_val, value, end_edge):
                    measurements.append(time - start_t)
                    end_idx = i
                    break
                prev_val = value
        
        return self._calculate_stats(measurements)
    
    def analyze_signal_activity(self, 
                                 signal_name: str,
                                 start_time: int = None,
                                 end_time: int = None) -> SignalActivity:
        """
        Analyze signal activity over time.
        
        Args:
            signal_name: Signal to analyze
            start_time: Optional window start
            end_time: Optional window end
        
        Returns:
            SignalActivity with metrics
        """
        values = self.parser.get_signal_values(signal_name)
        if not values:
            return SignalActivity(signal_name=signal_name)
        
        # Apply time window
        if start_time:
            values = [(t, v) for t, v in values if t >= start_time]
        if end_time:
            values = [(t, v) for t, v in values if t <= end_time]
        
        if not values:
            return SignalActivity(signal_name=signal_name)
        
        activity = SignalActivity(signal_name=signal_name)
        
        prev_time = values[0][0]
        prev_val = values[0][1]
        high_times = []
        low_times = []
        current_high_start = None
        current_low_start = None
        
        if self._is_high(prev_val):
            current_high_start = prev_time
        else:
            current_low_start = prev_time
        
        for time, value in values[1:]:
            is_high = self._is_high(value)
            was_high = self._is_high(prev_val)
            
            if is_high and not was_high:
                # Rising edge
                activity.rising_edges += 1
                activity.total_transitions += 1
                if current_low_start is not None:
                    low_times.append(time - current_low_start)
                current_high_start = time
                current_low_start = None
                
            elif not is_high and was_high:
                # Falling edge
                activity.falling_edges += 1
                activity.total_transitions += 1
                if current_high_start is not None:
                    high_times.append(time - current_high_start)
                current_low_start = time
                current_high_start = None
            
            prev_val = value
            prev_time = time
        
        # Calculate metrics
        total_time = values[-1][0] - values[0][0]
        if total_time > 0:
            activity.time_high = sum(high_times)
            activity.time_low = sum(low_times)
            activity.duty_cycle = activity.time_high / total_time
            activity.toggle_rate = activity.total_transitions / total_time
        
        if high_times:
            activity.avg_pulse_width_high = sum(high_times) / len(high_times)
        if low_times:
            activity.avg_pulse_width_low = sum(low_times) / len(low_times)
        
        return activity
    
    def detect_glitches(self,
                        signal_name: str,
                        min_pulse_width: int,
                        start_time: int = None,
                        end_time: int = None) -> List[GlitchInfo]:
        """
        Detect glitches (pulses shorter than minimum width).
        
        Args:
            signal_name: Signal to check
            min_pulse_width: Minimum valid pulse width
            start_time: Optional window start
            end_time: Optional window end
        
        Returns:
            List of detected glitches
        """
        values = self.parser.get_signal_values(signal_name)
        if not values:
            return []
        
        glitches = []
        
        for i in range(1, len(values) - 1):
            time, value = values[i]
            
            if start_time and time < start_time:
                continue
            if end_time and time > end_time:
                break
            
            prev_time, prev_val = values[i - 1]
            next_time, next_val = values[i + 1]
            
            # Check for narrow pulse
            pulse_width = next_time - time
            
            if pulse_width < min_pulse_width and value != prev_val:
                glitches.append(GlitchInfo(
                    signal=signal_name,
                    time=time,
                    duration=pulse_width,
                    value_sequence=[prev_val, value, next_val]
                ))
        
        return glitches
    
    def check_setup_hold(self,
                          data_signal: str,
                          clock_signal: str,
                          clock_edge: str = "rise",
                          setup_time: int = 0,
                          hold_time: int = 0) -> List[ProtocolViolation]:
        """
        Check setup/hold timing violations.
        
        Args:
            data_signal: Data signal name
            clock_signal: Clock signal name
            clock_edge: Clock edge to check ("rise" or "fall")
            setup_time: Required setup time before clock edge
            hold_time: Required hold time after clock edge
        
        Returns:
            List of timing violations
        """
        violations = []
        
        data_values = self.parser.get_signal_values(data_signal)
        clock_values = self.parser.get_signal_values(clock_signal)
        
        if not data_values or not clock_values:
            return violations
        
        # Find clock edges
        clock_edges = []
        prev_val = None
        for time, value in clock_values:
            if self._is_edge(prev_val, value, clock_edge):
                clock_edges.append(time)
            prev_val = value
        
        # Check each clock edge
        for clk_time in clock_edges:
            # Find data changes near clock edge
            for i, (time, value) in enumerate(data_values):
                if time < clk_time - setup_time:
                    continue
                if time > clk_time + hold_time:
                    break
                
                # Check if data changed during setup window
                if clk_time - setup_time <= time < clk_time:
                    violations.append(ProtocolViolation(
                        violation_type="setup_violation",
                        time=clk_time,
                        signals={data_signal: value, clock_signal: "edge"},
                        description=f"Data changed {clk_time - time} units before clock edge (setup={setup_time})",
                        severity="error"
                    ))
                
                # Check if data changed during hold window
                elif clk_time < time <= clk_time + hold_time:
                    violations.append(ProtocolViolation(
                        violation_type="hold_violation",
                        time=clk_time,
                        signals={data_signal: value, clock_signal: "edge"},
                        description=f"Data changed {time - clk_time} units after clock edge (hold={hold_time})",
                        severity="error"
                    ))
        
        return violations
    
    def check_handshake_protocol(self,
                                   req_signal: str,
                                   ack_signal: str,
                                   max_latency: int = None,
                                   req_edge: str = "rise",
                                   ack_edge: str = "rise") -> List[ProtocolViolation]:
        """
        Check request/acknowledge handshake protocol.
        
        Args:
            req_signal: Request signal name
            ack_signal: Acknowledge signal name
            max_latency: Maximum allowed latency (optional)
            req_edge: Request edge type
            ack_edge: Acknowledge edge type
        
        Returns:
            List of protocol violations
        """
        violations = []
        
        req_values = self.parser.get_signal_values(req_signal)
        ack_values = self.parser.get_signal_values(ack_signal)
        
        if not req_values or not ack_values:
            return violations
        
        # Find request events
        req_times = []
        prev_val = None
        for time, value in req_values:
            if self._is_edge(prev_val, value, req_edge):
                req_times.append(time)
            prev_val = value
        
        # Check each request for corresponding acknowledge
        ack_idx = 0
        prev_val = None
        for req_time in req_times:
            found_ack = False
            
            for i in range(ack_idx, len(ack_values)):
                time, value = ack_values[i]
                if time <= req_time:
                    prev_val = value
                    continue
                
                if self._is_edge(prev_val, value, ack_edge):
                    latency = time - req_time
                    
                    if max_latency and latency > max_latency:
                        violations.append(ProtocolViolation(
                            violation_type="latency_violation",
                            time=req_time,
                            signals={req_signal: "edge", ack_signal: "edge"},
                            description=f"ACK latency {latency} exceeds max {max_latency}",
                            severity="warning"
                        ))
                    
                    found_ack = True
                    ack_idx = i
                    break
                
                prev_val = value
            
            if not found_ack:
                violations.append(ProtocolViolation(
                    violation_type="missing_ack",
                    time=req_time,
                    signals={req_signal: "edge", ack_signal: "none"},
                    description=f"No ACK received for request at {req_time}",
                    severity="error"
                ))
        
        return violations
    
    def get_signal_histogram(self,
                              signal_name: str,
                              num_bins: int = 10,
                              start_time: int = None,
                              end_time: int = None) -> Dict[str, int]:
        """
        Generate histogram of signal value durations.
        
        Args:
            signal_name: Signal to analyze
            num_bins: Number of histogram bins
            start_time: Optional window start
            end_time: Optional window end
        
        Returns:
            Dictionary mapping bin labels to counts
        """
        values = self.parser.get_signal_values(signal_name)
        if not values or len(values) < 2:
            return {}
        
        # Calculate durations
        durations = []
        for i in range(len(values) - 1):
            time, _ = values[i]
            next_time, _ = values[i + 1]
            
            if start_time and time < start_time:
                continue
            if end_time and time > end_time:
                break
            
            durations.append(next_time - time)
        
        if not durations:
            return {}
        
        # Create histogram
        min_dur = min(durations)
        max_dur = max(durations)
        
        if min_dur == max_dur:
            return {f"{min_dur}": len(durations)}
        
        bin_width = (max_dur - min_dur) / num_bins
        histogram = defaultdict(int)
        
        for dur in durations:
            bin_idx = min(int((dur - min_dur) / bin_width), num_bins - 1)
            bin_start = min_dur + bin_idx * bin_width
            bin_end = bin_start + bin_width
            histogram[f"{bin_start:.1f}-{bin_end:.1f}"] += 1
        
        return dict(histogram)
    
    def _is_edge(self, prev_val: str, curr_val: str, edge_type: str) -> bool:
        """Check if value change matches edge type."""
        if prev_val is None:
            return False
        
        prev_high = self._is_high(prev_val)
        curr_high = self._is_high(curr_val)
        
        if edge_type == "rise":
            return not prev_high and curr_high
        elif edge_type == "fall":
            return prev_high and not curr_high
        elif edge_type == "any":
            return prev_high != curr_high
        
        return False
    
    def _is_high(self, value: str) -> bool:
        """Check if value represents logic high."""
        if not value:
            return False
        return value in ('1', 'H', 'h') or (value[0] == 'b' and '1' in value)
    
    def _calculate_stats(self, measurements: List[float]) -> TimingStats:
        """Calculate statistics from measurements."""
        stats = TimingStats()
        
        if not measurements:
            return stats
        
        measurements.sort()
        n = len(measurements)
        
        stats.count = n
        stats.min_time = measurements[0]
        stats.max_time = measurements[-1]
        stats.mean = sum(measurements) / n
        stats.median = measurements[n // 2]
        stats.p95 = measurements[int(n * 0.95)] if n >= 20 else measurements[-1]
        stats.p99 = measurements[int(n * 0.99)] if n >= 100 else measurements[-1]
        
        # Standard deviation
        if n > 1:
            variance = sum((x - stats.mean) ** 2 for x in measurements) / (n - 1)
            stats.std_dev = math.sqrt(variance)
        
        # Histogram (10 bins)
        if stats.max_time > stats.min_time:
            bin_width = (stats.max_time - stats.min_time) / 10
            for m in measurements:
                bin_idx = min(int((m - stats.min_time) / bin_width), 9)
                bin_label = f"{stats.min_time + bin_idx * bin_width:.1f}"
                stats.histogram[bin_label] = stats.histogram.get(bin_label, 0) + 1
        
        return stats
