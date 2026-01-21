"""
Multi-File Waveform Comparison.

Compare waveforms across simulation runs, regression tests, 
and design iterations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
from pathlib import Path

from waveformgpt.vcd_parser import VCDParser


class DifferenceType(Enum):
    """Types of waveform differences."""
    MISSING_SIGNAL = "missing_signal"
    EXTRA_SIGNAL = "extra_signal"
    VALUE_MISMATCH = "value_mismatch"
    TIMING_SHIFT = "timing_shift"
    GLITCH = "glitch"
    EDGE_COUNT = "edge_count"
    WIDTH_MISMATCH = "width_mismatch"


@dataclass
class SignalDifference:
    """A difference found in signal comparison."""
    signal_name: str
    diff_type: DifferenceType
    time: Optional[int] = None
    expected: Optional[str] = None
    actual: Optional[str] = None
    description: str = ""


@dataclass
class ComparisonResult:
    """Result of waveform comparison."""
    file1: str
    file2: str
    match: bool = True
    differences: List[SignalDifference] = field(default_factory=list)
    matched_signals: int = 0
    total_signals: int = 0
    
    @property
    def match_percentage(self) -> float:
        if self.total_signals == 0:
            return 0.0
        return (self.matched_signals / self.total_signals) * 100
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        status = "✓ MATCH" if self.match else "✗ MISMATCH"
        lines = [
            f"Comparison: {self.file1} vs {self.file2}",
            f"Status: {status}",
            f"Signals: {self.matched_signals}/{self.total_signals} matched ({self.match_percentage:.1f}%)",
        ]
        
        if self.differences:
            lines.append(f"\nDifferences ({len(self.differences)}):")
            for diff in self.differences[:10]:
                lines.append(f"  - {diff.signal_name}: {diff.description}")
            if len(self.differences) > 10:
                lines.append(f"  ... and {len(self.differences) - 10} more")
        
        return "\n".join(lines)


@dataclass
class ComparisonOptions:
    """Options for waveform comparison."""
    ignore_signals: List[str] = field(default_factory=list)
    include_signals: Optional[List[str]] = None
    timing_tolerance: int = 0  # Allow timing shifts up to this amount
    ignore_glitches: bool = False
    glitch_threshold: int = 1  # Minimum pulse width to not be a glitch
    compare_x_as_match: bool = False  # Treat X values as matching anything
    start_time: Optional[int] = None
    end_time: Optional[int] = None


class WaveformComparator:
    """
    Compare waveforms from different VCD files.
    
    Useful for:
    - Regression testing (compare golden vs current)
    - Design iteration comparison
    - Multi-run analysis
    - Debug comparing expected vs actual
    """
    
    def __init__(self, options: ComparisonOptions = None):
        self.options = options or ComparisonOptions()
    
    def compare(self, file1: str, file2: str) -> ComparisonResult:
        """
        Compare two VCD files.
        
        Args:
            file1: Path to first (reference/golden) VCD file
            file2: Path to second (test/actual) VCD file
        
        Returns:
            ComparisonResult with all differences
        """
        parser1 = VCDParser(file1)
        parser2 = VCDParser(file2)
        
        result = ComparisonResult(
            file1=file1,
            file2=file2
        )
        
        # Get signal lists
        signals1 = {s.name: s for s in parser1.search_signals(".*")}
        signals2 = {s.name: s for s in parser2.search_signals(".*")}
        
        # Determine signals to compare
        if self.options.include_signals:
            compare_signals = set(self.options.include_signals)
        else:
            compare_signals = set(signals1.keys()) | set(signals2.keys())
        
        # Remove ignored signals
        compare_signals -= set(self.options.ignore_signals)
        
        result.total_signals = len(compare_signals)
        
        for sig_name in compare_signals:
            sig_result = self._compare_signal(
                sig_name, parser1, parser2, signals1, signals2
            )
            
            if sig_result:
                result.differences.extend(sig_result)
            else:
                result.matched_signals += 1
        
        result.match = len(result.differences) == 0
        
        return result
    
    def _compare_signal(self,
                         signal_name: str,
                         parser1: VCDParser,
                         parser2: VCDParser,
                         signals1: Dict,
                         signals2: Dict) -> List[SignalDifference]:
        """Compare a single signal between two parsers."""
        differences = []
        
        # Check signal existence
        if signal_name not in signals1:
            differences.append(SignalDifference(
                signal_name=signal_name,
                diff_type=DifferenceType.EXTRA_SIGNAL,
                description=f"Signal only exists in second file"
            ))
            return differences
        
        if signal_name not in signals2:
            differences.append(SignalDifference(
                signal_name=signal_name,
                diff_type=DifferenceType.MISSING_SIGNAL,
                description=f"Signal missing from second file"
            ))
            return differences
        
        # Check width match
        if signals1[signal_name].width != signals2[signal_name].width:
            differences.append(SignalDifference(
                signal_name=signal_name,
                diff_type=DifferenceType.WIDTH_MISMATCH,
                expected=str(signals1[signal_name].width),
                actual=str(signals2[signal_name].width),
                description=f"Width mismatch: {signals1[signal_name].width} vs {signals2[signal_name].width}"
            ))
            return differences
        
        # Get values
        values1 = parser1.get_signal_values(signal_name)
        values2 = parser2.get_signal_values(signal_name)
        
        # Apply time window
        if self.options.start_time is not None:
            values1 = [(t, v) for t, v in values1 if t >= self.options.start_time]
            values2 = [(t, v) for t, v in values2 if t >= self.options.start_time]
        
        if self.options.end_time is not None:
            values1 = [(t, v) for t, v in values1 if t <= self.options.end_time]
            values2 = [(t, v) for t, v in values2 if t <= self.options.end_time]
        
        # Compare value sequences
        differences.extend(self._compare_values(
            signal_name, values1, values2
        ))
        
        return differences
    
    def _compare_values(self,
                         signal_name: str,
                         values1: List[Tuple[int, str]],
                         values2: List[Tuple[int, str]]) -> List[SignalDifference]:
        """Compare value change sequences."""
        differences = []
        
        # Build combined timeline
        all_times = sorted(set(t for t, _ in values1) | set(t for t, _ in values2))
        
        # Create lookup
        val1_at = {}
        val2_at = {}
        
        current1 = None
        for time, value in values1:
            current1 = value
            val1_at[time] = value
        
        current2 = None
        for time, value in values2:
            current2 = value
            val2_at[time] = value
        
        # Get value at each time
        def get_value_at(values, at_time):
            result = None
            for time, value in values:
                if time <= at_time:
                    result = value
                else:
                    break
            return result
        
        # Compare at each time point
        for time in all_times:
            v1 = get_value_at(values1, time)
            v2 = get_value_at(values2, time)
            
            if not self._values_match(v1, v2):
                # Check if this is just a timing shift
                if self.options.timing_tolerance > 0:
                    # Look for matching value within tolerance
                    found_match = False
                    for offset in range(-self.options.timing_tolerance, 
                                        self.options.timing_tolerance + 1):
                        check_time = time + offset
                        v2_shifted = get_value_at(values2, check_time)
                        if self._values_match(v1, v2_shifted):
                            if offset != 0:
                                differences.append(SignalDifference(
                                    signal_name=signal_name,
                                    diff_type=DifferenceType.TIMING_SHIFT,
                                    time=time,
                                    expected=v1,
                                    actual=v2,
                                    description=f"Value shift of {offset} at time {time}"
                                ))
                            found_match = True
                            break
                    
                    if not found_match:
                        differences.append(SignalDifference(
                            signal_name=signal_name,
                            diff_type=DifferenceType.VALUE_MISMATCH,
                            time=time,
                            expected=v1,
                            actual=v2,
                            description=f"Mismatch at {time}: expected {v1}, got {v2}"
                        ))
                else:
                    differences.append(SignalDifference(
                        signal_name=signal_name,
                        diff_type=DifferenceType.VALUE_MISMATCH,
                        time=time,
                        expected=v1,
                        actual=v2,
                        description=f"Mismatch at {time}: expected {v1}, got {v2}"
                    ))
        
        return differences
    
    def _values_match(self, v1: Optional[str], v2: Optional[str]) -> bool:
        """Check if two values match."""
        if v1 == v2:
            return True
        
        if v1 is None or v2 is None:
            return False
        
        # Handle X values
        if self.options.compare_x_as_match:
            if 'x' in v1.lower() or 'x' in v2.lower():
                return True
        
        return False
    
    def compare_multiple(self, 
                          files: List[str],
                          reference: str = None) -> Dict[str, ComparisonResult]:
        """
        Compare multiple VCD files.
        
        Args:
            files: List of VCD file paths
            reference: Optional reference file (defaults to first file)
        
        Returns:
            Dictionary of file pairs to ComparisonResult
        """
        if not files:
            return {}
        
        reference = reference or files[0]
        results = {}
        
        for file in files:
            if file != reference:
                key = f"{reference} vs {file}"
                results[key] = self.compare(reference, file)
        
        return results


def create_comparison_report(results: Dict[str, ComparisonResult],
                              output_path: str = None) -> str:
    """
    Create a detailed comparison report.
    
    Args:
        results: Dictionary of comparison results
        output_path: Optional path to save report
    
    Returns:
        Markdown-formatted report
    """
    lines = [
        "# Waveform Comparison Report",
        "",
        f"**Total Comparisons:** {len(results)}",
        "",
    ]
    
    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Comparison | Status | Match % | Differences |")
    lines.append("|------------|--------|---------|-------------|")
    
    for name, result in results.items():
        status = "✓ Pass" if result.match else "✗ Fail"
        lines.append(
            f"| {name} | {status} | {result.match_percentage:.1f}% | {len(result.differences)} |"
        )
    
    lines.append("")
    
    # Detailed differences
    lines.append("## Detailed Differences")
    lines.append("")
    
    for name, result in results.items():
        if result.differences:
            lines.append(f"### {name}")
            lines.append("")
            
            # Group by signal
            by_signal = {}
            for diff in result.differences:
                if diff.signal_name not in by_signal:
                    by_signal[diff.signal_name] = []
                by_signal[diff.signal_name].append(diff)
            
            for sig_name, diffs in by_signal.items():
                lines.append(f"**{sig_name}**")
                for diff in diffs[:5]:
                    lines.append(f"  - {diff.diff_type.value}: {diff.description}")
                if len(diffs) > 5:
                    lines.append(f"  - ... and {len(diffs) - 5} more")
                lines.append("")
    
    report = "\n".join(lines)
    
    if output_path:
        Path(output_path).write_text(report)
    
    return report
