"""Tests for waveform comparison."""

import pytest
from waveformgpt.compare import (
    WaveformComparator,
    ComparisonResult,
    ComparisonOptions,
    DifferenceType,
    SignalDifference,
    create_comparison_report,
)


class TestWaveformComparator:
    """Test WaveformComparator functionality."""
    
    @pytest.fixture
    def golden_vcd(self, tmp_path):
        """Create golden/reference VCD."""
        vcd_content = """$version Golden $end
$timescale 1ns $end
$scope module top $end
$var wire 1 ! clk $end
$var wire 1 " data $end
$var wire 8 # bus [7:0] $end
$enddefinitions $end
$dumpvars
0!
0"
b00000000 #
$end
#10
1!
#20
0!
1"
b00001111 #
#30
1!
#40
0!
0"
b11110000 #
#50
1!
"""
        vcd_path = tmp_path / "golden.vcd"
        vcd_path.write_text(vcd_content)
        return str(vcd_path)
    
    @pytest.fixture
    def matching_vcd(self, tmp_path):
        """Create matching VCD."""
        vcd_content = """$version Test $end
$timescale 1ns $end
$scope module top $end
$var wire 1 ! clk $end
$var wire 1 " data $end
$var wire 8 # bus [7:0] $end
$enddefinitions $end
$dumpvars
0!
0"
b00000000 #
$end
#10
1!
#20
0!
1"
b00001111 #
#30
1!
#40
0!
0"
b11110000 #
#50
1!
"""
        vcd_path = tmp_path / "matching.vcd"
        vcd_path.write_text(vcd_content)
        return str(vcd_path)
    
    @pytest.fixture
    def different_vcd(self, tmp_path):
        """Create VCD with differences."""
        vcd_content = """$version Test $end
$timescale 1ns $end
$scope module top $end
$var wire 1 ! clk $end
$var wire 1 " data $end
$var wire 8 # bus [7:0] $end
$enddefinitions $end
$dumpvars
0!
0"
b00000000 #
$end
#10
1!
#20
0!
1"
b00001111 #
#30
1!
#35
0"
b11111111 #
#40
0!
#50
1!
"""
        vcd_path = tmp_path / "different.vcd"
        vcd_path.write_text(vcd_content)
        return str(vcd_path)
    
    def test_compare_matching_files(self, golden_vcd, matching_vcd):
        """Test comparing identical files."""
        comparator = WaveformComparator()
        result = comparator.compare(golden_vcd, matching_vcd)
        
        assert result.match is True
        assert len(result.differences) == 0
        assert result.match_percentage == 100.0
    
    def test_compare_different_files(self, golden_vcd, different_vcd):
        """Test comparing different files."""
        comparator = WaveformComparator()
        result = comparator.compare(golden_vcd, different_vcd)
        
        assert result.match is False
        assert len(result.differences) > 0
    
    def test_compare_with_timing_tolerance(self, golden_vcd, different_vcd):
        """Test comparison with timing tolerance."""
        options = ComparisonOptions(timing_tolerance=10)
        comparator = WaveformComparator(options)
        result = comparator.compare(golden_vcd, different_vcd)
        
        # Should still find some differences
        assert isinstance(result, ComparisonResult)
    
    def test_compare_ignore_signals(self, golden_vcd, different_vcd):
        """Test ignoring specific signals."""
        options = ComparisonOptions(ignore_signals=["data", "bus"])
        comparator = WaveformComparator(options)
        result = comparator.compare(golden_vcd, different_vcd)
        
        # Only comparing clk, which should match
        assert result.match is True
    
    def test_compare_include_signals(self, golden_vcd, different_vcd):
        """Test including only specific signals."""
        options = ComparisonOptions(include_signals=["clk"])
        comparator = WaveformComparator(options)
        result = comparator.compare(golden_vcd, different_vcd)
        
        # Only comparing clk
        assert result.total_signals == 1
    
    def test_comparison_summary(self, golden_vcd, different_vcd):
        """Test result summary generation."""
        comparator = WaveformComparator()
        result = comparator.compare(golden_vcd, different_vcd)
        
        summary = result.summary()
        
        assert "Comparison:" in summary
        assert "Status:" in summary
        assert "Signals:" in summary


class TestComparisonResult:
    """Test ComparisonResult dataclass."""
    
    def test_match_percentage(self):
        """Test match percentage calculation."""
        result = ComparisonResult(
            file1="a.vcd",
            file2="b.vcd",
            matched_signals=8,
            total_signals=10
        )
        
        assert result.match_percentage == 80.0
    
    def test_match_percentage_zero_signals(self):
        """Test match percentage with zero signals."""
        result = ComparisonResult(
            file1="a.vcd",
            file2="b.vcd",
            matched_signals=0,
            total_signals=0
        )
        
        assert result.match_percentage == 0.0


class TestComparisonReport:
    """Test comparison report generation."""
    
    def test_create_report(self):
        """Test report creation."""
        results = {
            "golden vs test1": ComparisonResult(
                file1="golden.vcd",
                file2="test1.vcd",
                match=True,
                matched_signals=5,
                total_signals=5
            ),
            "golden vs test2": ComparisonResult(
                file1="golden.vcd",
                file2="test2.vcd",
                match=False,
                matched_signals=3,
                total_signals=5,
                differences=[
                    SignalDifference(
                        signal_name="data",
                        diff_type=DifferenceType.VALUE_MISMATCH,
                        time=100,
                        expected="1",
                        actual="0",
                        description="Mismatch at 100"
                    )
                ]
            ),
        }
        
        report = create_comparison_report(results)
        
        assert "# Waveform Comparison Report" in report
        assert "golden vs test1" in report
        assert "golden vs test2" in report
        assert "Pass" in report
        assert "Fail" in report
