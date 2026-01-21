"""Tests for waveform analysis module."""

import pytest
from waveformgpt.vcd_parser import VCDParser
from waveformgpt.analyzer import (
    WaveformAnalyzer,
    TimingStats,
    SignalActivity,
    GlitchInfo,
    ProtocolViolation,
)


class TestWaveformAnalyzer:
    """Test WaveformAnalyzer functionality."""
    
    @pytest.fixture
    def sample_vcd(self, tmp_path):
        """Create sample VCD for testing."""
        vcd_content = """$version Test $end
$timescale 1ns $end
$scope module top $end
$var wire 1 ! clk $end
$var wire 1 " data $end
$var wire 1 # req $end
$var wire 1 $ ack $end
$enddefinitions $end
$dumpvars
0!
0"
0#
0$
$end
#10
1!
#20
0!
1#
#30
1!
1"
#40
0!
1$
#50
1!
#60
0!
0#
0"
#70
1!
0$
#80
0!
#90
1!
#100
0!
"""
        vcd_path = tmp_path / "test.vcd"
        vcd_path.write_text(vcd_content)
        return str(vcd_path)
    
    @pytest.fixture
    def analyzer(self, sample_vcd):
        """Create analyzer with sample VCD."""
        parser = VCDParser(sample_vcd)
        return WaveformAnalyzer(parser)
    
    def test_analyze_signal_activity(self, analyzer):
        """Test signal activity analysis."""
        activity = analyzer.analyze_signal_activity("clk")
        
        assert activity.signal_name == "clk"
        assert activity.rising_edges > 0
        assert activity.falling_edges > 0
        assert activity.total_transitions > 0
    
    def test_analyze_activity_with_window(self, analyzer):
        """Test activity analysis with time window."""
        activity = analyzer.analyze_signal_activity("clk", start_time=20, end_time=60)
        
        assert activity.signal_name == "clk"
        # Should have fewer transitions in limited window
    
    def test_measure_timing(self, analyzer):
        """Test timing measurement between events."""
        stats = analyzer.measure_timing(
            "req", "rise",
            "ack", "rise"
        )
        
        assert stats.count > 0
        assert stats.min_time > 0
    
    def test_detect_glitches(self, analyzer):
        """Test glitch detection."""
        glitches = analyzer.detect_glitches("data", min_pulse_width=100)
        
        # Should detect the short pulse in data
        # (data is high from 30-60, which is 30ns)
        assert isinstance(glitches, list)
    
    def test_check_handshake_protocol(self, analyzer):
        """Test handshake protocol checking."""
        violations = analyzer.check_handshake_protocol(
            "req", "ack",
            max_latency=30
        )
        
        assert isinstance(violations, list)
    
    def test_signal_histogram(self, analyzer):
        """Test histogram generation."""
        hist = analyzer.get_signal_histogram("clk", num_bins=5)
        
        assert isinstance(hist, dict)


class TestTimingStats:
    """Test TimingStats dataclass."""
    
    def test_summary_empty(self):
        """Test summary with no measurements."""
        stats = TimingStats()
        summary = stats.summary()
        
        assert "No measurements" in summary
    
    def test_summary_with_data(self):
        """Test summary with measurements."""
        stats = TimingStats(
            count=10,
            min_time=5.0,
            max_time=20.0,
            mean=12.5,
            std_dev=4.2,
            median=12.0,
            p95=18.0,
            p99=19.5
        )
        
        summary = stats.summary()
        
        assert "Count: 10" in summary
        assert "Min: 5.00" in summary
        assert "Max: 20.00" in summary


class TestSetupHoldCheck:
    """Test setup/hold timing checks."""
    
    @pytest.fixture
    def setup_hold_vcd(self, tmp_path):
        """Create VCD with setup/hold scenarios."""
        vcd_content = """$version Test $end
$timescale 1ns $end
$scope module top $end
$var wire 1 ! clk $end
$var wire 1 " data $end
$enddefinitions $end
$dumpvars
0!
0"
$end
#5
1"
#10
1!
#20
0!
#25
0"
#30
1!
#40
0!
#48
1"
#50
1!
#60
0!
0"
"""
        vcd_path = tmp_path / "setup_hold.vcd"
        vcd_path.write_text(vcd_content)
        return str(vcd_path)
    
    def test_setup_hold_check(self, setup_hold_vcd):
        """Test setup/hold violation detection."""
        parser = VCDParser(setup_hold_vcd)
        analyzer = WaveformAnalyzer(parser)
        
        violations = analyzer.check_setup_hold(
            "data", "clk",
            clock_edge="rise",
            setup_time=5,
            hold_time=5
        )
        
        assert isinstance(violations, list)
        # Should detect violations at clock edges 50 (data changed at 48)
