"""
Regression Testing for Waveforms.

Compare waveforms across simulation runs to detect:
- Behavioral changes
- Performance regressions  
- Bug introductions
- Coverage changes
"""

import os
import json
import hashlib
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime


@dataclass
class SignalDiff:
    """Difference in a signal between two waveforms."""
    signal: str
    diff_type: str  # "value", "timing", "missing", "new"
    times: List[int] = field(default_factory=list)
    old_values: List[Any] = field(default_factory=list)
    new_values: List[Any] = field(default_factory=list)
    description: str = ""


@dataclass
class RegressionResult:
    """Result of regression comparison."""
    passed: bool
    identical: bool
    baseline_file: str
    test_file: str
    signal_diffs: List[SignalDiff] = field(default_factory=list)
    missing_signals: List[str] = field(default_factory=list)
    new_signals: List[str] = field(default_factory=list)
    timing_delta: float = 0  # ns difference in simulation time
    summary: str = ""


class WaveformRegression:
    """
    Compare waveforms for regression testing.
    
    Usage:
        reg = WaveformRegression()
        
        # Compare two VCD files
        result = reg.compare("baseline.vcd", "new.vcd")
        
        if not result.passed:
            for diff in result.signal_diffs:
                print(f"Signal {diff.signal}: {diff.description}")
        
        # Save as new baseline if passed
        if result.passed:
            reg.save_baseline("new.vcd", "baseline.vcd")
    """
    
    def __init__(self, tolerance_ns: int = 0, ignore_signals: List[str] = None):
        """
        Args:
            tolerance_ns: Timing tolerance in nanoseconds
            ignore_signals: Signals to ignore in comparison
        """
        self.tolerance_ns = tolerance_ns
        self.ignore_signals = set(ignore_signals or [])
    
    def compare(self, baseline_file: str, test_file: str,
                signals: List[str] = None) -> RegressionResult:
        """
        Compare two VCD files.
        
        Args:
            baseline_file: Known good waveform
            test_file: New waveform to test
            signals: Specific signals to compare (None = all)
        """
        from waveformgpt import VCDParser
        
        baseline = VCDParser(baseline_file)
        test = VCDParser(test_file)
        
        result = RegressionResult(
            passed=True,
            identical=True,
            baseline_file=baseline_file,
            test_file=test_file,
        )
        
        # Get signal sets
        baseline_signals = set(s.full_name for s in baseline.header.signals.values())
        test_signals = set(s.full_name for s in test.header.signals.values())
        
        # Remove ignored signals
        baseline_signals -= self.ignore_signals
        test_signals -= self.ignore_signals
        
        # Filter to requested signals
        if signals:
            baseline_signals &= set(signals)
            test_signals &= set(signals)
        
        # Check for missing/new signals
        result.missing_signals = list(baseline_signals - test_signals)
        result.new_signals = list(test_signals - baseline_signals)
        
        if result.missing_signals:
            result.passed = False
            result.identical = False
        
        # Compare common signals
        common_signals = baseline_signals & test_signals
        
        for sig in common_signals:
            baseline_data = baseline.get_signal_values(sig)
            test_data = test.get_signal_values(sig)
            
            diff = self._compare_signal(sig, baseline_data, test_data)
            
            if diff:
                result.signal_diffs.append(diff)
                result.identical = False
                
                # Value differences are failures
                if diff.diff_type == "value":
                    result.passed = False
        
        # Compare total simulation time
        baseline_end = baseline.get_time_range()[1]
        test_end = test.get_time_range()[1]
        result.timing_delta = test_end - baseline_end
        
        # Generate summary
        result.summary = self._generate_summary(result)
        
        return result
    
    def _compare_signal(self, signal: str, 
                        baseline: List[Tuple[int, Any]], 
                        test: List[Tuple[int, Any]]) -> Optional[SignalDiff]:
        """Compare a single signal's waveform."""
        
        # Build time-value maps
        baseline_map = {t: v for t, v in baseline}
        test_map = {t: v for t, v in test}
        
        all_times = sorted(set(baseline_map.keys()) | set(test_map.keys()))
        
        diff_times = []
        old_vals = []
        new_vals = []
        
        baseline_val = None
        test_val = None
        
        for t in all_times:
            if t in baseline_map:
                baseline_val = baseline_map[t]
            if t in test_map:
                test_val = test_map[t]
            
            # Check if values differ at this time
            if baseline_val != test_val:
                diff_times.append(t)
                old_vals.append(baseline_val)
                new_vals.append(test_val)
        
        if diff_times:
            return SignalDiff(
                signal=signal,
                diff_type="value",
                times=diff_times[:20],  # Limit to first 20
                old_values=old_vals[:20],
                new_values=new_vals[:20],
                description=f"Value mismatch at {len(diff_times)} time points"
            )
        
        return None
    
    def _generate_summary(self, result: RegressionResult) -> str:
        """Generate human-readable summary."""
        lines = []
        
        if result.identical:
            lines.append("✅ Waveforms are identical")
        elif result.passed:
            lines.append("✅ Waveforms differ but within tolerance")
        else:
            lines.append("❌ Waveforms have regressions")
        
        if result.missing_signals:
            lines.append(f"⚠️  Missing signals: {', '.join(result.missing_signals)}")
        
        if result.new_signals:
            lines.append(f"ℹ️  New signals: {', '.join(result.new_signals)}")
        
        if result.signal_diffs:
            lines.append(f"📊 {len(result.signal_diffs)} signals with differences")
            for diff in result.signal_diffs[:5]:
                lines.append(f"   - {diff.signal}: {diff.description}")
        
        if result.timing_delta != 0:
            lines.append(f"⏱️  Simulation time delta: {result.timing_delta}ns")
        
        return "\n".join(lines)
    
    @staticmethod
    def save_baseline(vcd_file: str, baseline_path: str):
        """Save a VCD file as a baseline."""
        import shutil
        shutil.copy(vcd_file, baseline_path)
    
    @staticmethod
    def hash_waveform(vcd_file: str, signals: List[str] = None) -> str:
        """Generate a hash of waveform values for quick comparison."""
        from waveformgpt import VCDParser
        
        parser = VCDParser(vcd_file)
        
        hasher = hashlib.sha256()
        
        all_signals = signals or [s.full_name for s in parser.header.signals.values()]
        for sig in sorted(all_signals):
            values = parser.get_signal_values(sig)
            for t, v in values:
                hasher.update(f"{sig}:{t}:{v}".encode())
        
        return hasher.hexdigest()


@dataclass
class CoveragePoint:
    """A coverage point definition."""
    name: str
    signal: str
    condition: str  # e.g., "value == 1", "value > 100"
    hit: bool = False
    hit_count: int = 0
    first_hit_time: Optional[int] = None


@dataclass
class CoverageResult:
    """Coverage analysis result."""
    total_points: int
    hit_points: int
    coverage_percent: float
    points: List[CoveragePoint]
    uncovered: List[str]  # Names of unhit points


class CoverageAnalyzer:
    """
    Analyze signal coverage in waveforms.
    
    Usage:
        cov = CoverageAnalyzer("simulation.vcd")
        
        # Add coverage points
        cov.add_point("fifo_full", "fifo_full", "value == 1")
        cov.add_point("error_flag", "error", "value == 1")
        cov.add_point("high_count", "count", "value > 200")
        
        # Analyze
        result = cov.analyze()
        print(f"Coverage: {result.coverage_percent:.1f}%")
    """
    
    def __init__(self, vcd_file: str):
        from waveformgpt import VCDParser
        self.parser = VCDParser(vcd_file)
        self.points: List[CoveragePoint] = []
    
    def add_point(self, name: str, signal: str, condition: str):
        """Add a coverage point."""
        self.points.append(CoveragePoint(
            name=name,
            signal=signal,
            condition=condition
        ))
    
    def add_toggle_coverage(self, signal: str):
        """Add toggle coverage (0->1 and 1->0 transitions)."""
        self.add_point(f"{signal}_rise", signal, "transition == 'rise'")
        self.add_point(f"{signal}_fall", signal, "transition == 'fall'")
    
    def add_value_coverage(self, signal: str, values: List[Any]):
        """Add coverage for specific values."""
        for val in values:
            self.add_point(f"{signal}_{val}", signal, f"value == {val}")
    
    def _get_signal_list(self) -> List[str]:
        """Get list of signal names from parser."""
        return [s.full_name for s in self.parser.header.signals.values()]
    
    def analyze(self) -> CoverageResult:
        """Analyze coverage."""
        
        for point in self.points:
            self._check_point(point)
        
        hit = sum(1 for p in self.points if p.hit)
        total = len(self.points)
        
        return CoverageResult(
            total_points=total,
            hit_points=hit,
            coverage_percent=(hit / total * 100) if total > 0 else 0,
            points=self.points,
            uncovered=[p.name for p in self.points if not p.hit]
        )
    
    def _check_point(self, point: CoveragePoint):
        """Check if a coverage point is hit."""
        values = self.parser.get_signal_values(point.signal)
        
        prev_val = None
        
        for time, value in values:
            # Determine transition type
            transition = None
            if prev_val is not None:
                if prev_val == 0 and value == 1:
                    transition = 'rise'
                elif prev_val == 1 and value == 0:
                    transition = 'fall'
            
            # Evaluate condition
            try:
                # Create evaluation context
                ctx = {'value': value, 'transition': transition}
                
                if eval(point.condition, {"__builtins__": {}}, ctx):
                    point.hit = True
                    point.hit_count += 1
                    if point.first_hit_time is None:
                        point.first_hit_time = time
            except:
                pass
            
            prev_val = value
    
    def generate_report(self) -> str:
        """Generate coverage report."""
        result = self.analyze()
        
        lines = [
            "=" * 50,
            "COVERAGE REPORT",
            "=" * 50,
            f"Coverage: {result.hit_points}/{result.total_points} ({result.coverage_percent:.1f}%)",
            "",
            "Points:"
        ]
        
        for p in result.points:
            status = "✅ HIT" if p.hit else "❌ MISS"
            lines.append(f"  {status} {p.name}")
            if p.hit:
                lines.append(f"       Count: {p.hit_count}, First: {p.first_hit_time}")
        
        if result.uncovered:
            lines.extend([
                "",
                "Uncovered Points:",
                *[f"  - {name}" for name in result.uncovered]
            ])
        
        lines.append("=" * 50)
        
        return "\n".join(lines)


class GoldenModel:
    """
    Golden model comparison for verification.
    
    Compare simulation results against expected values from a golden model.
    
    Usage:
        golden = GoldenModel()
        
        # Define expected behavior
        golden.expect("output", {
            100: 0,
            200: 1,
            300: 0,
        })
        
        # Compare with simulation
        result = golden.verify("simulation.vcd")
    """
    
    def __init__(self, tolerance_ns: int = 1):
        self.expected: Dict[str, Dict[int, Any]] = {}
        self.tolerance = tolerance_ns
    
    def expect(self, signal: str, values: Dict[int, Any]):
        """
        Define expected values for a signal.
        
        Args:
            signal: Signal name
            values: Dict of {time: expected_value}
        """
        self.expected[signal] = values
    
    def expect_sequence(self, signal: str, sequence: List[Tuple[int, Any]]):
        """Define expected sequence."""
        self.expected[signal] = {t: v for t, v in sequence}
    
    def load_golden(self, golden_file: str):
        """Load expected values from a golden VCD file."""
        from waveformgpt import VCDParser
        
        golden = VCDParser(golden_file)
        
        for sig in [s.full_name for s in golden.header.signals.values()]:
            values = golden.get_signal_values(sig)
            self.expected[sig] = {t: v for t, v in values}
    
    def verify(self, vcd_file: str) -> Dict[str, Any]:
        """
        Verify simulation against golden model.
        
        Returns:
            Dict with verification results
        """
        from waveformgpt import VCDParser
        
        parser = VCDParser(vcd_file)
        
        results = {
            "passed": True,
            "signals_checked": len(self.expected),
            "mismatches": [],
            "summary": ""
        }
        
        for signal, expected_values in self.expected.items():
            actual = parser.get_signal_values(signal)
            actual_map = {t: v for t, v in actual}
            
            for exp_time, exp_val in expected_values.items():
                # Find actual value at or near expected time
                actual_val = None
                for t in range(exp_time - self.tolerance, exp_time + self.tolerance + 1):
                    if t in actual_map:
                        actual_val = actual_map[t]
                        break
                
                if actual_val != exp_val:
                    results["passed"] = False
                    results["mismatches"].append({
                        "signal": signal,
                        "time": exp_time,
                        "expected": exp_val,
                        "actual": actual_val
                    })
        
        # Generate summary
        if results["passed"]:
            results["summary"] = f"✅ All {results['signals_checked']} signals match golden model"
        else:
            results["summary"] = f"❌ {len(results['mismatches'])} mismatches found"
        
        return results


class TestSuite:
    """
    Organize and run multiple waveform tests.
    
    Usage:
        suite = TestSuite("verification_suite")
        
        suite.add_test("fifo_basic", "sim/fifo_basic.vcd", [
            ("FIFO fills correctly?", lambda r: "yes" in r.lower()),
            ("Any overflow?", lambda r: "no overflow" in r.lower()),
        ])
        
        suite.add_test("uart_timing", "sim/uart.vcd", [
            ("Baud rate correct?", lambda r: "9600" in r),
        ])
        
        results = suite.run()
        suite.print_summary(results)
    """
    
    def __init__(self, name: str):
        self.name = name
        self.tests: List[Dict] = []
    
    def add_test(self, name: str, vcd_file: str,
                 checks: List[Tuple[str, callable]]):
        """
        Add a test.
        
        Args:
            name: Test name
            vcd_file: VCD file to analyze
            checks: List of (question, pass_condition) tuples
        """
        self.tests.append({
            "name": name,
            "vcd_file": vcd_file,
            "checks": checks
        })
    
    def add_assertion_test(self, name: str, vcd_file: str,
                           assertions: List[str]):
        """Add a test with temporal assertions."""
        self.tests.append({
            "name": name,
            "vcd_file": vcd_file,
            "assertions": assertions
        })
    
    def run(self, use_llm: bool = True) -> Dict[str, Any]:
        """Run all tests."""
        from waveformgpt import WaveformChat, AssertionChecker
        
        results = {
            "suite_name": self.name,
            "total_tests": len(self.tests),
            "passed": 0,
            "failed": 0,
            "test_results": []
        }
        
        for test in self.tests:
            test_result = {
                "name": test["name"],
                "vcd_file": test["vcd_file"],
                "passed": True,
                "checks": []
            }
            
            try:
                chat = WaveformChat(test["vcd_file"], use_llm=use_llm)
                
                # Run question-based checks
                for question, condition in test.get("checks", []):
                    response = chat.ask(question)
                    content = response.content if hasattr(response, 'content') else str(response)
                    
                    check_passed = condition(content)
                    
                    test_result["checks"].append({
                        "question": question,
                        "answer": content,
                        "passed": check_passed
                    })
                    
                    if not check_passed:
                        test_result["passed"] = False
                
                # Run assertion checks
                if "assertions" in test:
                    checker = AssertionChecker(chat.parser)
                    for assertion in test["assertions"]:
                        assertion_result = checker.check(assertion)
                        
                        test_result["checks"].append({
                            "assertion": assertion,
                            "passed": assertion_result.passed,
                            "failures": len(assertion_result.failures)
                        })
                        
                        if not assertion_result.passed:
                            test_result["passed"] = False
                
            except Exception as e:
                test_result["passed"] = False
                test_result["error"] = str(e)
            
            if test_result["passed"]:
                results["passed"] += 1
            else:
                results["failed"] += 1
            
            results["test_results"].append(test_result)
        
        return results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print test summary."""
        print("\n" + "=" * 60)
        print(f"📋 Test Suite: {results['suite_name']}")
        print("=" * 60)
        
        for test in results["test_results"]:
            status = "✅" if test["passed"] else "❌"
            print(f"\n{status} {test['name']}")
            
            for check in test.get("checks", []):
                check_status = "✓" if check.get("passed") else "✗"
                if "question" in check:
                    print(f"   {check_status} {check['question'][:50]}...")
                elif "assertion" in check:
                    print(f"   {check_status} {check['assertion']}")
            
            if "error" in test:
                print(f"   ⚠️  Error: {test['error']}")
        
        print("\n" + "-" * 60)
        print(f"Results: {results['passed']}/{results['total_tests']} passed")
        
        if results["failed"] > 0:
            print(f"⚠️  {results['failed']} test(s) failed!")
        else:
            print("✅ All tests passed!")
