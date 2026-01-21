"""
VCD (Value Change Dump) Parser for WaveformGPT.

Supports streaming parsing of large VCD files without loading entirely into memory.
"""

from dataclasses import dataclass, field
from typing import Iterator, Optional, Dict, List, Tuple
from enum import Enum
from pathlib import Path
import re


class SignalType(Enum):
    WIRE = "wire"
    REG = "reg"
    INTEGER = "integer"
    PARAMETER = "parameter"
    REAL = "real"


@dataclass
class Signal:
    """Represents a signal in the VCD file."""
    id_code: str
    name: str
    width: int
    signal_type: SignalType
    scope: str
    
    @property
    def full_name(self) -> str:
        return f"{self.scope}.{self.name}" if self.scope else self.name


@dataclass
class ValueChange:
    """Represents a value change event."""
    time: int
    signal_id: str
    value: str  # Can be binary string, 'x', 'z', or real number
    
    def as_int(self) -> Optional[int]:
        """Convert binary value to integer, None if contains x/z."""
        if 'x' in self.value.lower() or 'z' in self.value.lower():
            return None
        try:
            return int(self.value, 2)
        except ValueError:
            return None


@dataclass
class VCDHeader:
    """VCD file header information."""
    date: Optional[str] = None
    version: Optional[str] = None
    timescale: str = "1ns"
    timescale_value: int = 1
    timescale_unit: str = "ns"
    signals: Dict[str, Signal] = field(default_factory=dict)
    
    def get_signal_by_name(self, name: str) -> Optional[Signal]:
        """Find signal by name (partial match supported)."""
        name_lower = name.lower()
        for sig in self.signals.values():
            if sig.name.lower() == name_lower or sig.full_name.lower() == name_lower:
                return sig
            if name_lower in sig.full_name.lower():
                return sig
        return None
    
    def search_signals(self, pattern: str) -> List[Signal]:
        """Search signals by regex pattern."""
        regex = re.compile(pattern, re.IGNORECASE)
        return [s for s in self.signals.values() if regex.search(s.full_name)]


class VCDParser:
    """
    Streaming VCD parser that efficiently handles large files.
    
    Usage:
        parser = VCDParser("simulation.vcd")
        header = parser.parse_header()
        
        for change in parser.stream_changes():
            print(f"t={change.time}: {change.signal_id} = {change.value}")
    """
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self._header: Optional[VCDHeader] = None
        self._data_start_pos: int = 0
        
        if not self.filepath.exists():
            raise FileNotFoundError(f"VCD file not found: {filepath}")
    
    @property
    def header(self) -> VCDHeader:
        """Get parsed header (parses on first access)."""
        if self._header is None:
            self._header = self.parse_header()
        return self._header
    
    def search_signals(self, pattern: str) -> List[Signal]:
        """Search signals by regex pattern. Delegates to header.search_signals()."""
        return self.header.search_signals(pattern)
    
    @property
    def timescale(self) -> str:
        """Get timescale from header."""
        return self.header.timescale
    
    @property
    def version(self) -> str:
        """Get version from header."""
        return self.header.version
    
    @property
    def date(self) -> str:
        """Get date from header."""
        return self.header.date
    
    def parse_header(self) -> VCDHeader:
        """Parse VCD header to extract signal definitions."""
        header = VCDHeader()
        scope_stack: List[str] = []
        
        with open(self.filepath, 'r') as f:
            in_definitions = True
            
            while in_definitions:
                line = f.readline()
                if not line:
                    break
                
                line = line.strip()
                
                if line.startswith('$date'):
                    header.date = self._read_until_end(f, line)
                elif line.startswith('$version'):
                    header.version = self._read_until_end(f, line)
                elif line.startswith('$timescale'):
                    ts = self._read_until_end(f, line)
                    header.timescale = ts
                    self._parse_timescale(ts, header)
                elif line.startswith('$scope'):
                    parts = line.split()
                    if len(parts) >= 3:
                        scope_stack.append(parts[2])
                elif line.startswith('$upscope'):
                    if scope_stack:
                        scope_stack.pop()
                elif line.startswith('$var'):
                    sig = self._parse_var(line, '.'.join(scope_stack))
                    if sig:
                        header.signals[sig.id_code] = sig
                elif line.startswith('$enddefinitions'):
                    in_definitions = False
                    self._data_start_pos = f.tell()
        
        self._header = header
        return header
    
    def _read_until_end(self, f, first_line: str) -> str:
        """Read content until $end marker."""
        content = first_line
        while '$end' not in content:
            content += ' ' + f.readline().strip()
        # Extract content between $keyword and $end
        match = re.search(r'\$\w+\s+(.+?)\s*\$end', content)
        return match.group(1).strip() if match else ""
    
    def _parse_timescale(self, ts: str, header: VCDHeader):
        """Parse timescale like '1ns' or '100ps'."""
        match = re.match(r'(\d+)\s*(\w+)', ts)
        if match:
            header.timescale_value = int(match.group(1))
            header.timescale_unit = match.group(2)
    
    def _parse_var(self, line: str, scope: str) -> Optional[Signal]:
        """Parse $var line to Signal."""
        # $var wire 8 # data [7:0] $end
        parts = line.split()
        if len(parts) < 5:
            return None
        
        try:
            signal_type = SignalType(parts[1])
        except ValueError:
            signal_type = SignalType.WIRE
        
        width = int(parts[2])
        id_code = parts[3]
        name = parts[4]
        
        # Handle array notation [7:0]
        if len(parts) > 5 and parts[5] != '$end':
            name += ' ' + parts[5]
        
        return Signal(
            id_code=id_code,
            name=name,
            width=width,
            signal_type=signal_type,
            scope=scope
        )
    
    def stream_changes(self, 
                       start_time: int = 0, 
                       end_time: Optional[int] = None,
                       signal_filter: Optional[List[str]] = None) -> Iterator[ValueChange]:
        """
        Stream value changes from VCD file.
        
        Args:
            start_time: Only return changes at or after this time
            end_time: Only return changes at or before this time
            signal_filter: Only return changes for these signal IDs
        
        Yields:
            ValueChange objects
        """
        # Ensure header is parsed
        _ = self.header
        
        filter_set = set(signal_filter) if signal_filter else None
        current_time = 0
        
        with open(self.filepath, 'r') as f:
            # Skip to data section
            f.seek(self._data_start_pos)
            
            for line in f:
                line = line.strip()
                if not line or line.startswith('$'):
                    continue
                
                # Time marker
                if line.startswith('#'):
                    current_time = int(line[1:])
                    if end_time is not None and current_time > end_time:
                        break
                    continue
                
                # Skip if before start time
                if current_time < start_time:
                    continue
                
                # Value change
                change = self._parse_value_change(line, current_time)
                if change:
                    if filter_set is None or change.signal_id in filter_set:
                        yield change
    
    def _parse_value_change(self, line: str, time: int) -> Optional[ValueChange]:
        """Parse a value change line."""
        if not line:
            return None
        
        # Single bit: 0!, 1#, x$, z%
        if line[0] in '01xXzZ':
            return ValueChange(
                time=time,
                signal_id=line[1:].strip(),
                value=line[0]
            )
        
        # Multi-bit binary: b10101010 #
        if line[0] in 'bB':
            parts = line[1:].split()
            if len(parts) >= 2:
                return ValueChange(
                    time=time,
                    signal_id=parts[1],
                    value=parts[0]
                )
        
        # Real number: r1.234 #
        if line[0] in 'rR':
            parts = line[1:].split()
            if len(parts) >= 2:
                return ValueChange(
                    time=time,
                    signal_id=parts[1],
                    value=parts[0]
                )
        
        return None
    
    def get_signal_values(self, signal_name: str, 
                          start_time: int = 0,
                          end_time: Optional[int] = None) -> List[Tuple[int, str]]:
        """Get all value changes for a specific signal."""
        sig = self.header.get_signal_by_name(signal_name)
        if not sig:
            raise ValueError(f"Signal not found: {signal_name}")
        
        values = []
        for change in self.stream_changes(start_time, end_time, [sig.id_code]):
            values.append((change.time, change.value))
        return values
    
    def get_time_range(self) -> Tuple[int, int]:
        """Get the time range of the VCD file."""
        min_time = None
        max_time = 0
        
        with open(self.filepath, 'r') as f:
            f.seek(self._data_start_pos)
            for line in f:
                if line.startswith('#'):
                    t = int(line[1:].strip())
                    if min_time is None:
                        min_time = t
                    max_time = t
        
        return (min_time or 0, max_time)
    
    def build_signal_index(self, signals: Optional[List[str]] = None) -> Dict[str, List[Tuple[int, str]]]:
        """
        Build an in-memory index of signal transitions.
        Useful for repeated queries on smaller VCDs.
        """
        index: Dict[str, List[Tuple[int, str]]] = {}
        
        if signals:
            sig_ids = []
            for name in signals:
                sig = self.header.get_signal_by_name(name)
                if sig:
                    sig_ids.append(sig.id_code)
                    index[sig.id_code] = []
        else:
            sig_ids = list(self.header.signals.keys())
            for sid in sig_ids:
                index[sid] = []
        
        for change in self.stream_changes(signal_filter=sig_ids if signals else None):
            if change.signal_id in index:
                index[change.signal_id].append((change.time, change.value))
        
        return index
