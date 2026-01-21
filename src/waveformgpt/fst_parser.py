"""
FST Parser for WaveformGPT.

Provides FST (Fast Signal Trace) format support using libfst.
FST is a compressed waveform format used by Verilator, GHDL, and other tools.
"""

from dataclasses import dataclass
from typing import Iterator, List, Dict, Optional, Tuple
from pathlib import Path
import struct
import zlib

from waveformgpt.vcd_parser import Signal, ValueChange


@dataclass
class FSTSignal:
    """FST signal metadata."""
    handle: int
    name: str
    path: str
    width: int
    var_type: str
    direction: str = ""


class FSTParser:
    """
    FST waveform file parser.
    
    FST is a compressed binary format that's much faster to read than VCD
    for large waveforms. This parser provides the same interface as VCDParser.
    
    Note: For full FST support, install `pylibfst` or use GTKWave's fst2vcd.
    This implementation provides basic FST reading capability.
    """
    
    # FST block types
    FST_BL_HDR = 0
    FST_BL_VCDATA = 1
    FST_BL_BLACKOUT = 2
    FST_BL_GEOM = 3
    FST_BL_HIER = 4
    FST_BL_VCDATA_DYN_ALIAS = 5
    FST_BL_HIER_LZ4 = 6
    FST_BL_HIER_LZ4DUO = 7
    FST_BL_VCDATA_DYN_ALIAS2 = 8
    
    def __init__(self, filename: str):
        self.filename = Path(filename)
        self.signals: Dict[int, FSTSignal] = {}
        self.signal_by_name: Dict[str, FSTSignal] = {}
        self.timescale: str = "1ns"
        self.timescale_factor: int = 1
        self.start_time: int = 0
        self.end_time: int = 0
        self.version: str = ""
        self.date: str = ""
        
        self._file = None
        self._header_parsed = False
        
        if not self.filename.exists():
            raise FileNotFoundError(f"FST file not found: {filename}")
        
        self._parse_header()
    
    def _parse_header(self):
        """Parse FST file header."""
        with open(self.filename, 'rb') as f:
            # Check magic number
            magic = f.read(4)
            if magic != b'\x1b\x46\x53\x54':  # FST magic: \x1b FST
                raise ValueError(f"Invalid FST file: {self.filename}")
            
            # Read header block
            self._file = f
            self._parse_blocks()
        
        self._header_parsed = True
    
    def _parse_blocks(self):
        """Parse FST blocks to extract metadata."""
        f = self._file
        f.seek(4)  # Skip magic
        
        while True:
            block_type_byte = f.read(1)
            if not block_type_byte:
                break
            
            block_type = struct.unpack('B', block_type_byte)[0]
            
            if block_type == self.FST_BL_HDR:
                self._parse_header_block(f)
            elif block_type == self.FST_BL_HIER:
                self._parse_hierarchy_block(f)
            elif block_type in (self.FST_BL_VCDATA, self.FST_BL_VCDATA_DYN_ALIAS, 
                               self.FST_BL_VCDATA_DYN_ALIAS2):
                # Skip value change data during header parse
                block_len = self._read_varint(f)
                f.seek(block_len, 1)
            else:
                # Unknown block, try to skip
                try:
                    block_len = self._read_varint(f)
                    f.seek(block_len, 1)
                except:
                    break
    
    def _parse_header_block(self, f):
        """Parse FST header block."""
        block_len = self._read_varint(f)
        start_pos = f.tell()
        
        self.start_time = self._read_varint(f)
        self.end_time = self._read_varint(f)
        
        # Read version string
        version_len = struct.unpack('B', f.read(1))[0]
        self.version = f.read(version_len).decode('utf-8', errors='replace')
        
        # Read date string  
        date_len = struct.unpack('B', f.read(1))[0]
        self.date = f.read(date_len).decode('utf-8', errors='replace')
        
        # Skip rest of header
        f.seek(start_pos + block_len)
    
    def _parse_hierarchy_block(self, f):
        """Parse FST hierarchy block to extract signal definitions."""
        block_len = self._read_varint(f)
        start_pos = f.tell()
        
        # Read compressed hierarchy data
        comp_len = self._read_varint(f)
        uncomp_len = self._read_varint(f)
        
        compressed_data = f.read(comp_len)
        
        try:
            hierarchy_data = zlib.decompress(compressed_data)
            self._parse_hierarchy_data(hierarchy_data)
        except zlib.error:
            # Skip if decompression fails
            pass
        
        # Skip to end of block
        f.seek(start_pos + block_len)
    
    def _parse_hierarchy_data(self, data: bytes):
        """Parse decompressed hierarchy data."""
        pos = 0
        scope_stack = []
        handle = 0
        
        while pos < len(data):
            item_type = data[pos]
            pos += 1
            
            if item_type == 0:  # End of hierarchy
                break
            elif item_type == 1:  # Scope push
                # Read scope name
                end = data.find(b'\x00', pos)
                scope_name = data[pos:end].decode('utf-8', errors='replace')
                scope_stack.append(scope_name)
                pos = end + 1
            elif item_type == 2:  # Scope pop
                if scope_stack:
                    scope_stack.pop()
            elif item_type == 3:  # Variable
                # Read variable info
                var_type = data[pos]
                direction = data[pos + 1]
                pos += 2
                
                # Read name
                end = data.find(b'\x00', pos)
                var_name = data[pos:end].decode('utf-8', errors='replace')
                pos = end + 1
                
                # Read width
                width = self._read_varint_bytes(data, pos)
                pos += self._varint_size(width)
                
                handle += 1
                full_path = '.'.join(scope_stack + [var_name])
                
                signal = FSTSignal(
                    handle=handle,
                    name=var_name,
                    path=full_path,
                    width=width,
                    var_type=self._var_type_name(var_type),
                    direction=self._direction_name(direction)
                )
                
                self.signals[handle] = signal
                self.signal_by_name[var_name] = signal
                self.signal_by_name[full_path] = signal
    
    def _read_varint(self, f) -> int:
        """Read a variable-length integer."""
        result = 0
        shift = 0
        while True:
            byte = f.read(1)
            if not byte:
                break
            b = struct.unpack('B', byte)[0]
            result |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        return result
    
    def _read_varint_bytes(self, data: bytes, pos: int) -> int:
        """Read varint from byte array."""
        result = 0
        shift = 0
        while pos < len(data):
            b = data[pos]
            result |= (b & 0x7F) << shift
            pos += 1
            if not (b & 0x80):
                break
            shift += 7
        return result
    
    def _varint_size(self, value: int) -> int:
        """Get size of varint encoding."""
        if value == 0:
            return 1
        size = 0
        while value:
            size += 1
            value >>= 7
        return size
    
    def _var_type_name(self, type_code: int) -> str:
        """Convert variable type code to name."""
        types = {
            0: "event", 1: "integer", 2: "parameter", 3: "real",
            4: "reg", 5: "supply0", 6: "supply1", 7: "time",
            8: "tri", 9: "triand", 10: "trior", 11: "trireg",
            12: "tri0", 13: "tri1", 14: "wand", 15: "wire",
            16: "wor", 17: "port", 18: "bit", 19: "logic",
            20: "int", 21: "shortint", 22: "longint", 23: "byte",
            24: "enum", 25: "shortreal"
        }
        return types.get(type_code, "wire")
    
    def _direction_name(self, dir_code: int) -> str:
        """Convert direction code to name."""
        dirs = {0: "", 1: "input", 2: "output", 3: "inout",
                4: "buffer", 5: "linkage"}
        return dirs.get(dir_code, "")
    
    def get_signals(self) -> List[Signal]:
        """Get list of all signals as VCD-compatible Signal objects."""
        signals = []
        for fst_sig in self.signals.values():
            signals.append(Signal(
                id=str(fst_sig.handle),
                name=fst_sig.name,
                path=fst_sig.path,
                width=fst_sig.width,
                var_type=fst_sig.var_type
            ))
        return signals
    
    def search_signals(self, pattern: str) -> List[Signal]:
        """Search signals by name pattern."""
        import re
        regex = re.compile(pattern.replace('*', '.*'), re.IGNORECASE)
        return [s for s in self.get_signals() if regex.search(s.name) or regex.search(s.path)]
    
    def stream_changes(self, 
                       signals: List[str] = None,
                       start_time: int = None,
                       end_time: int = None) -> Iterator[ValueChange]:
        """
        Stream value changes from FST file.
        
        Note: Full streaming requires pylibfst. This is a basic implementation.
        """
        # For now, return empty - full implementation would use libfst
        # or convert to VCD on-the-fly
        yield from []
    
    def get_time_range(self) -> Tuple[int, int]:
        """Get simulation time range."""
        return (self.start_time, self.end_time)


def convert_fst_to_vcd(fst_path: str, vcd_path: str = None) -> str:
    """
    Convert FST file to VCD using GTKWave's fst2vcd.
    
    Args:
        fst_path: Path to input FST file
        vcd_path: Path to output VCD file (auto-generated if None)
    
    Returns:
        Path to converted VCD file
    """
    import subprocess
    import shutil
    
    if vcd_path is None:
        vcd_path = str(Path(fst_path).with_suffix('.vcd'))
    
    # Check if fst2vcd is available
    fst2vcd = shutil.which('fst2vcd')
    if not fst2vcd:
        raise RuntimeError(
            "fst2vcd not found. Install GTKWave or use: brew install gtkwave"
        )
    
    result = subprocess.run(
        [fst2vcd, fst_path, '-o', vcd_path],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"FST conversion failed: {result.stderr}")
    
    return vcd_path
