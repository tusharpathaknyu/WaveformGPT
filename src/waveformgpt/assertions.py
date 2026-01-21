"""
Temporal Assertion Language for Waveforms.

Express complex temporal properties using a simple DSL.
Based on SVA/PSL concepts but simplified for waveform analysis.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any, Union
from enum import Enum
import re

from waveformgpt.vcd_parser import VCDParser


class AssertionResult(Enum):
    """Result of assertion check."""
    PASS = "pass"
    FAIL = "fail"
    VACUOUS = "vacuous"  # Antecedent never occurred
    INCOMPLETE = "incomplete"  # Couldn't complete check


@dataclass
class AssertionMatch:
    """A single assertion match or failure."""
    result: AssertionResult
    start_time: int
    end_time: Optional[int] = None
    signals: Dict[str, str] = field(default_factory=dict)
    message: str = ""


@dataclass
class AssertionCheckResult:
    """Complete result of assertion checking."""
    assertion_name: str
    expression: str
    result: AssertionResult
    matches: List[AssertionMatch] = field(default_factory=list)
    pass_count: int = 0
    fail_count: int = 0
    vacuous_count: int = 0
    
    @property
    def total_checks(self) -> int:
        return self.pass_count + self.fail_count + self.vacuous_count
    
    def summary(self) -> str:
        """Generate summary string."""
        status = "✓ PASS" if self.result == AssertionResult.PASS else "✗ FAIL"
        return (
            f"{self.assertion_name}: {status}\n"
            f"  Expression: {self.expression}\n"
            f"  Checks: {self.total_checks} (pass: {self.pass_count}, "
            f"fail: {self.fail_count}, vacuous: {self.vacuous_count})"
        )


class TemporalOperator(Enum):
    """Temporal operators."""
    NEXT = "##"  # Next cycle
    DELAY = "##N"  # N cycles delay
    RANGE = "##[M:N]"  # Range of cycles
    THROUGHOUT = "throughout"
    UNTIL = "until"
    EVENTUALLY = "eventually"
    ALWAYS = "always"


@dataclass
class Assertion:
    """
    A temporal assertion.
    
    Syntax examples:
    - "rose(req) |-> ##[1:5] rose(ack)"  # Request implies ack within 1-5 cycles
    - "stable(data) throughout valid"     # Data stable while valid
    - "fell(busy) |=> rose(done)"         # Busy falling followed by done
    """
    name: str
    expression: str
    
    # Parsed components
    antecedent: Optional[str] = None
    consequent: Optional[str] = None
    operator: str = "|=>"
    
    # Timing constraints
    min_delay: int = 0
    max_delay: int = 0
    
    # Options
    cover_mode: bool = False  # True = cover property, False = assert
    
    def __post_init__(self):
        self._parse()
    
    def _parse(self):
        """Parse assertion expression."""
        # Split on implication operators
        if "|->" in self.expression:
            parts = self.expression.split("|->", 1)
            self.antecedent = parts[0].strip()
            self.consequent = parts[1].strip()
            self.operator = "|->"
        elif "|=>" in self.expression:
            parts = self.expression.split("|=>", 1)
            self.antecedent = parts[0].strip()
            self.consequent = parts[1].strip()
            self.operator = "|=>"
        else:
            # Simple property
            self.consequent = self.expression.strip()
        
        # Extract timing from consequent
        if self.consequent:
            timing_match = re.search(r'##\[(\d+):(\d+)\]', self.consequent)
            if timing_match:
                self.min_delay = int(timing_match.group(1))
                self.max_delay = int(timing_match.group(2))
            else:
                timing_match = re.search(r'##(\d+)', self.consequent)
                if timing_match:
                    self.min_delay = int(timing_match.group(1))
                    self.max_delay = self.min_delay


class AssertionChecker:
    """
    Check temporal assertions against waveforms.
    
    Supports a simplified SVA-like syntax:
    
    Events:
        rose(signal)  - Rising edge
        fell(signal)  - Falling edge
        high(signal)  - Signal is high
        low(signal)   - Signal is low
        changed(signal) - Any change
        stable(signal)  - No change
    
    Operators:
        |->  - Implication (same cycle)
        |=>  - Implication (next cycle)
        ##N  - N cycle delay
        ##[M:N] - Delay range
        throughout - Hold during
        until - Hold until
    """
    
    def __init__(self, parser: VCDParser):
        self.parser = parser
        self._cache: Dict[str, Any] = {}
        
        # Event evaluators
        self.event_funcs = {
            'rose': self._check_rose,
            'fell': self._check_fell,
            'high': self._check_high,
            'low': self._check_low,
            'changed': self._check_changed,
            'stable': self._check_stable,
        }
    
    def check(self, assertion: Union[str, Assertion],
              name: str = None) -> AssertionCheckResult:
        """
        Check an assertion against the waveform.
        
        Args:
            assertion: Assertion object or expression string
            name: Optional name (if expression string)
        
        Returns:
            AssertionCheckResult with pass/fail info
        """
        if isinstance(assertion, str):
            assertion = Assertion(name=name or "assertion", expression=assertion)
        
        result = AssertionCheckResult(
            assertion_name=assertion.name,
            expression=assertion.expression,
            result=AssertionResult.PASS
        )
        
        if assertion.antecedent:
            self._check_implication(assertion, result)
        else:
            self._check_property(assertion, result)
        
        # Determine overall result
        if result.fail_count > 0:
            result.result = AssertionResult.FAIL
        elif result.pass_count == 0 and result.vacuous_count > 0:
            result.result = AssertionResult.VACUOUS
        
        return result
    
    def _check_implication(self, assertion: Assertion, 
                           result: AssertionCheckResult):
        """Check implication-style assertion."""
        # Find all antecedent matches
        antecedent_times = self._find_event_times(assertion.antecedent)
        
        if not antecedent_times:
            result.vacuous_count = 1
            return
        
        # For each antecedent, check consequent
        for ant_time in antecedent_times:
            # Calculate check window
            if assertion.operator == "|->":
                start = ant_time + assertion.min_delay
                end = ant_time + assertion.max_delay if assertion.max_delay else ant_time
            else:  # |=>
                start = ant_time + 1 + assertion.min_delay
                end = ant_time + 1 + assertion.max_delay if assertion.max_delay else ant_time + 1
            
            # Check consequent in window
            consequent_times = self._find_event_times(
                assertion.consequent.split('##')[-1].strip(' []0123456789:'),
                start, end
            )
            
            if consequent_times:
                result.pass_count += 1
                result.matches.append(AssertionMatch(
                    result=AssertionResult.PASS,
                    start_time=ant_time,
                    end_time=consequent_times[0],
                    message=f"Consequent found at {consequent_times[0]}"
                ))
            else:
                result.fail_count += 1
                result.matches.append(AssertionMatch(
                    result=AssertionResult.FAIL,
                    start_time=ant_time,
                    end_time=end,
                    message=f"Consequent not found in [{start}:{end}]"
                ))
    
    def _check_property(self, assertion: Assertion,
                        result: AssertionCheckResult):
        """Check simple property (no implication)."""
        times = self._find_event_times(assertion.consequent)
        
        if assertion.cover_mode:
            # Cover: at least one occurrence
            if times:
                result.pass_count = len(times)
            else:
                result.fail_count = 1
        else:
            # Assert: property always holds
            result.pass_count = len(times) if times else 0
    
    def _find_event_times(self, event_expr: str,
                          start_time: int = None,
                          end_time: int = None) -> List[int]:
        """Find all times where event expression is true."""
        # Parse event expression
        match = re.match(r'(\w+)\(([^)]+)\)', event_expr.strip())
        if not match:
            return []
        
        func_name = match.group(1)
        signal_name = match.group(2).strip()
        
        if func_name not in self.event_funcs:
            return []
        
        try:
            return self.event_funcs[func_name](signal_name, start_time, end_time)
        except ValueError:
            # Signal not found
            return []
    
    def _check_rose(self, signal: str, 
                    start: int = None, end: int = None) -> List[int]:
        """Find rising edges."""
        try:
            values = self.parser.get_signal_values(signal)
        except ValueError:
            return []
        if not values:
            return []
        
        times = []
        prev_val = None
        
        for time, value in values:
            if start and time < start:
                prev_val = value
                continue
            if end and time > end:
                break
            
            if prev_val is not None:
                if self._is_low(prev_val) and self._is_high(value):
                    times.append(time)
            
            prev_val = value
        
        return times
    
    def _check_fell(self, signal: str,
                    start: int = None, end: int = None) -> List[int]:
        """Find falling edges."""
        try:
            values = self.parser.get_signal_values(signal)
        except ValueError:
            return []
        if not values:
            return []
        
        times = []
        prev_val = None
        
        for time, value in values:
            if start and time < start:
                prev_val = value
                continue
            if end and time > end:
                break
            
            if prev_val is not None:
                if self._is_high(prev_val) and self._is_low(value):
                    times.append(time)
            
            prev_val = value
        
        return times
    
    def _check_high(self, signal: str,
                    start: int = None, end: int = None) -> List[int]:
        """Find times when signal is high."""
        try:
            values = self.parser.get_signal_values(signal)
        except ValueError:
            return []
        if not values:
            return []
        
        return [time for time, value in values 
                if self._is_high(value)
                and (start is None or time >= start)
                and (end is None or time <= end)]
    
    def _check_low(self, signal: str,
                   start: int = None, end: int = None) -> List[int]:
        """Find times when signal is low."""
        try:
            values = self.parser.get_signal_values(signal)
        except ValueError:
            return []
        if not values:
            return []
        
        return [time for time, value in values
                if self._is_low(value)
                and (start is None or time >= start)
                and (end is None or time <= end)]
    
    def _check_changed(self, signal: str,
                       start: int = None, end: int = None) -> List[int]:
        """Find times when signal changed."""
        try:
            values = self.parser.get_signal_values(signal)
        except ValueError:
            return []
        if not values:
            return []
        
        times = []
        prev_val = None
        
        for time, value in values:
            if start and time < start:
                prev_val = value
                continue
            if end and time > end:
                break
            
            if prev_val is not None and value != prev_val:
                times.append(time)
            
            prev_val = value
        
        return times
    
    def _check_stable(self, signal: str,
                      start: int = None, end: int = None) -> List[int]:
        """Find times when signal is stable (no change)."""
        # Return all times except change times
        try:
            all_values = self.parser.get_signal_values(signal)
        except ValueError:
            return []
        change_times = set(self._check_changed(signal, start, end))
        
        return [time for time, _ in all_values
                if time not in change_times
                and (start is None or time >= start)
                and (end is None or time <= end)]
    
    def _is_high(self, value: str) -> bool:
        """Check if value is logic high."""
        if not value:
            return False
        return value in ('1', 'H', 'h')
    
    def _is_low(self, value: str) -> bool:
        """Check if value is logic low."""
        if not value:
            return False
        return value in ('0', 'L', 'l')


def check_assertions_from_file(parser: VCDParser, 
                                assertions_file: str) -> List[AssertionCheckResult]:
    """
    Check assertions defined in a file.
    
    File format (one assertion per line):
        name: expression
    
    Example:
        req_ack: rose(req) |-> ##[1:5] rose(ack)
        data_stable: stable(data) throughout valid
    
    Args:
        parser: VCD parser instance
        assertions_file: Path to assertions file
    
    Returns:
        List of check results
    """
    checker = AssertionChecker(parser)
    results = []
    
    with open(assertions_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if ':' in line:
                name, expr = line.split(':', 1)
                assertion = Assertion(name=name.strip(), expression=expr.strip())
            else:
                assertion = Assertion(name="assertion", expression=line)
            
            results.append(checker.check(assertion))
    
    return results


def generate_assertion_report(results: List[AssertionCheckResult],
                               output_path: str = None) -> str:
    """Generate assertion check report."""
    lines = [
        "# Assertion Check Report",
        "",
        "## Summary",
        "",
    ]
    
    total_pass = sum(1 for r in results if r.result == AssertionResult.PASS)
    total_fail = sum(1 for r in results if r.result == AssertionResult.FAIL)
    total_vacuous = sum(1 for r in results if r.result == AssertionResult.VACUOUS)
    
    lines.extend([
        f"- **Total Assertions:** {len(results)}",
        f"- **Passed:** {total_pass}",
        f"- **Failed:** {total_fail}",
        f"- **Vacuous:** {total_vacuous}",
        "",
        "## Details",
        "",
    ])
    
    for result in results:
        status_icon = {
            AssertionResult.PASS: "✓",
            AssertionResult.FAIL: "✗",
            AssertionResult.VACUOUS: "○",
        }.get(result.result, "?")
        
        lines.append(f"### {status_icon} {result.assertion_name}")
        lines.append(f"```")
        lines.append(result.expression)
        lines.append(f"```")
        lines.append(f"- Result: **{result.result.value.upper()}**")
        lines.append(f"- Checks: {result.total_checks}")
        
        if result.matches and result.fail_count > 0:
            lines.append("- Failures:")
            for match in result.matches[:5]:
                if match.result == AssertionResult.FAIL:
                    lines.append(f"  - Time {match.start_time}: {match.message}")
        
        lines.append("")
    
    report = "\n".join(lines)
    
    if output_path:
        from pathlib import Path
        Path(output_path).write_text(report)
    
    return report
