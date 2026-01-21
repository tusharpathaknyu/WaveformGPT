"""
Tests for WaveformGPT VCD Parser.
"""

import pytest
import tempfile
from pathlib import Path

from waveformgpt.vcd_parser import VCDParser, Signal, SignalType, ValueChange


# Sample VCD content for testing
SAMPLE_VCD = """$date
   Mon Jan 1 00:00:00 2024
$end
$version
   Test VCD Generator
$end
$timescale 1ns $end
$scope module tb $end
$var wire 1 ! clk $end
$var wire 1 " reset $end
$var wire 8 # data [7:0] $end
$var reg 1 $ valid $end
$upscope $end
$enddefinitions $end
$dumpvars
0!
1"
b00000000 #
0$
$end
#10
1!
#20
0!
#30
1!
0"
#40
0!
1$
b10101010 #
#50
1!
#60
0!
0$
#70
1!
#80
0!
#90
1!
#100
0!
"""


@pytest.fixture
def vcd_file():
    """Create a temporary VCD file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.vcd', delete=False) as f:
        f.write(SAMPLE_VCD)
        return f.name


@pytest.fixture
def parser(vcd_file):
    """Create a VCD parser instance."""
    return VCDParser(vcd_file)


class TestVCDParser:
    
    def test_parse_header(self, parser):
        """Test header parsing."""
        header = parser.header
        
        assert header.date is not None
        assert "2024" in header.date
        assert header.timescale == "1ns"
        assert header.timescale_value == 1
        assert header.timescale_unit == "ns"
    
    def test_parse_signals(self, parser):
        """Test signal parsing."""
        header = parser.header
        
        assert len(header.signals) == 4
        
        # Check clock signal
        clk = header.get_signal_by_name("clk")
        assert clk is not None
        assert clk.width == 1
        assert clk.signal_type == SignalType.WIRE
        assert clk.scope == "tb"
        assert clk.full_name == "tb.clk"
    
    def test_signal_search(self, parser):
        """Test signal search functionality."""
        header = parser.header
        
        # Exact match
        assert header.get_signal_by_name("clk") is not None
        
        # Partial match
        assert header.get_signal_by_name("dat") is not None
        
        # Regex search
        results = header.search_signals(r".*data.*")
        assert len(results) == 1
    
    def test_stream_changes(self, parser):
        """Test streaming value changes."""
        changes = list(parser.stream_changes())
        
        assert len(changes) > 0
        
        # Changes start from time 0 (dumpvars) or first timestamp
        first_change = changes[0]
        assert first_change.time >= 0
    
    def test_stream_changes_with_filter(self, parser):
        """Test streaming with signal filter."""
        clk = parser.header.get_signal_by_name("clk")
        changes = list(parser.stream_changes(signal_filter=[clk.id_code]))
        
        # All changes should be for clock
        for change in changes:
            assert change.signal_id == clk.id_code
    
    def test_stream_changes_time_range(self, parser):
        """Test streaming with time range."""
        changes = list(parser.stream_changes(start_time=30, end_time=60))
        
        for change in changes:
            assert 30 <= change.time <= 60
    
    def test_get_signal_values(self, parser):
        """Test getting values for a specific signal."""
        values = parser.get_signal_values("clk")
        
        assert len(values) > 0
        
        # Check clock toggles
        for i, (time, value) in enumerate(values):
            assert value in ("0", "1")
    
    def test_get_time_range(self, parser):
        """Test getting time range."""
        min_time, max_time = parser.get_time_range()
        
        assert min_time == 10
        assert max_time == 100
    
    def test_build_signal_index(self, parser):
        """Test building signal index."""
        index = parser.build_signal_index(["clk", "reset"])
        
        assert len(index) == 2
    
    def test_multibit_signal(self, parser):
        """Test multi-bit signal parsing."""
        values = parser.get_signal_values("data")
        
        # Should have initial value and one change
        assert len(values) >= 1
        
        # Check for multi-bit value
        has_multibit = any(len(v) > 1 for _, v in values)
        assert has_multibit


class TestValueChange:
    
    def test_as_int_binary(self):
        """Test converting binary value to int."""
        change = ValueChange(time=0, signal_id="!", value="10101010")
        assert change.as_int() == 170
    
    def test_as_int_single_bit(self):
        """Test single bit conversion."""
        change = ValueChange(time=0, signal_id="!", value="1")
        assert change.as_int() == 1
    
    def test_as_int_with_x(self):
        """Test value with X returns None."""
        change = ValueChange(time=0, signal_id="!", value="10x0")
        assert change.as_int() is None
    
    def test_as_int_with_z(self):
        """Test value with Z returns None."""
        change = ValueChange(time=0, signal_id="!", value="z")
        assert change.as_int() is None


class TestSignal:
    
    def test_full_name(self):
        """Test full name generation."""
        sig = Signal(
            id_code="!",
            name="clk",
            width=1,
            signal_type=SignalType.WIRE,
            scope="tb.dut"
        )
        assert sig.full_name == "tb.dut.clk"
    
    def test_full_name_no_scope(self):
        """Test full name with no scope."""
        sig = Signal(
            id_code="!",
            name="clk",
            width=1,
            signal_type=SignalType.WIRE,
            scope=""
        )
        assert sig.full_name == "clk"
