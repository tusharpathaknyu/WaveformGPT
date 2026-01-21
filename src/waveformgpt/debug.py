"""
Debug Utilities for Waveform Analysis.

Tools to help debug failing simulations:
- Root cause analysis
- Signal tracing
- Divergence detection
- Debug session management
"""

import os
import json
from typing import Optional, List, Dict, Any, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime


@dataclass
class DebugFinding:
    """A finding from debug analysis."""
    severity: str  # "critical", "warning", "info"
    category: str  # "timing", "protocol", "value", "state"
    signal: str
    time: int
    description: str
    context: Dict[str, Any] = field(default_factory=dict)
    suggested_fix: str = ""


@dataclass
class DebugSession:
    """A debug session with findings and analysis."""
    vcd_file: str
    start_time: datetime
    findings: List[DebugFinding] = field(default_factory=list)
    bookmarks: List[Dict] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    root_cause: Optional[str] = None


class WaveformDebugger:
    """
    Interactive waveform debugger.
    
    Usage:
        dbg = WaveformDebugger("simulation.vcd")
        
        # Find root cause of error
        dbg.trace_signal("error", target_value=1)
        
        # Analyze around failure point
        dbg.analyze_window(1523, window=100)
        
        # Get AI-powered debug suggestions
        dbg.explain_failure()
    """
    
    def __init__(self, vcd_file: str, use_llm: bool = True):
        from waveformgpt import VCDParser, WaveformChat
        
        self.parser = VCDParser(vcd_file)
        self.chat = WaveformChat(vcd_file, use_llm=use_llm)
        self.use_llm = use_llm
        
        self.session = DebugSession(
            vcd_file=vcd_file,
            start_time=datetime.now()
        )
    
    def trace_signal(self, signal: str, 
                     target_value: Any = 1,
                     direction: str = "backward") -> List[DebugFinding]:
        """
        Trace back to find what caused a signal to reach a value.
        
        Args:
            signal: Signal to trace
            target_value: The value we're investigating
            direction: "backward" to trace cause, "forward" to trace effect
        
        Returns:
            List of findings about signal dependencies
        """
        findings = []
        
        # Find when signal first reaches target value
        values = self.parser.get_signal_values(signal)
        target_time = None
        
        for time, value in values:
            if value == target_value:
                target_time = time
                break
        
        if target_time is None:
            findings.append(DebugFinding(
                severity="info",
                category="value",
                signal=signal,
                time=0,
                description=f"Signal never reaches value {target_value}"
            ))
            return findings
        
        findings.append(DebugFinding(
            severity="info",
            category="value",
            signal=signal,
            time=target_time,
            description=f"Signal first reaches {target_value} at time {target_time}"
        ))
        
        # Find what changed just before
        if direction == "backward":
            window_start = max(0, target_time - 100)
            all_signals = [s.full_name for s in self.parser.header.signals.values()]
            
            changes_before = []
            
            for sig in all_signals:
                if sig == signal:
                    continue
                    
                sig_values = self.parser.get_signal_values(sig)
                for time, value in sig_values:
                    if window_start <= time < target_time:
                        changes_before.append((sig, time, value))
            
            # Sort by time (closest to target first)
            changes_before.sort(key=lambda x: target_time - x[1])
            
            for sig, time, value in changes_before[:10]:
                findings.append(DebugFinding(
                    severity="info",
                    category="timing",
                    signal=sig,
                    time=time,
                    description=f"Changed to {value} at {time} ({target_time - time}ns before {signal}={target_value})"
                ))
        
        self.session.findings.extend(findings)
        return findings
    
    def find_divergence(self, golden_file: str) -> List[DebugFinding]:
        """
        Find where waveform diverges from golden reference.
        
        Returns first point of divergence for each signal.
        """
        from waveformgpt import VCDParser
        
        golden = VCDParser(golden_file)
        findings = []
        
        our_signals = {s.full_name for s in self.parser.header.signals.values()}
        golden_signals = {s.full_name for s in golden.header.signals.values()}
        
        for signal in our_signals:
            if signal not in golden_signals:
                continue
            
            actual = self.parser.get_signal_values(signal)
            expected = golden.get_signal_values(signal)
            
            # Build maps
            actual_map = {t: v for t, v in actual}
            expected_map = {t: v for t, v in expected}
            
            # Find first divergence
            all_times = sorted(set(actual_map.keys()) | set(expected_map.keys()))
            
            actual_val = None
            expected_val = None
            
            for t in all_times:
                if t in actual_map:
                    actual_val = actual_map[t]
                if t in expected_map:
                    expected_val = expected_map[t]
                
                if actual_val != expected_val:
                    findings.append(DebugFinding(
                        severity="critical",
                        category="value",
                        signal=signal,
                        time=t,
                        description=f"Divergence: expected {expected_val}, got {actual_val}",
                        context={
                            "expected": expected_val,
                            "actual": actual_val,
                            "golden_file": golden_file
                        }
                    ))
                    break
        
        self.session.findings.extend(findings)
        return findings
    
    def analyze_window(self, center_time: int, 
                       window: int = 100,
                       signals: List[str] = None) -> Dict[str, Any]:
        """
        Analyze all activity in a time window.
        
        Returns summary of signal changes and patterns.
        """
        start = max(0, center_time - window)
        end = center_time + window
        
        signals = signals or [s.full_name for s in self.parser.header.signals.values()]
        
        analysis = {
            "center_time": center_time,
            "window": (start, end),
            "signal_activity": {},
            "transitions": [],
            "summary": ""
        }
        
        for sig in signals:
            values = self.parser.get_signal_values(sig)
            
            changes = [(t, v) for t, v in values if start <= t <= end]
            
            if changes:
                analysis["signal_activity"][sig] = {
                    "changes": len(changes),
                    "values": changes
                }
                
                for t, v in changes:
                    analysis["transitions"].append({
                        "signal": sig,
                        "time": t,
                        "value": v
                    })
        
        # Sort transitions by time
        analysis["transitions"].sort(key=lambda x: x["time"])
        
        # Generate summary
        total_changes = sum(a["changes"] for a in analysis["signal_activity"].values())
        analysis["summary"] = f"{total_changes} signal changes in window [{start}, {end}]"
        
        # Bookmark this window
        self.bookmark(center_time, f"Analyzed window ±{window}ns")
        
        return analysis
    
    def explain_failure(self, error_signal: str = None,
                        error_time: int = None) -> str:
        """
        Use LLM to explain what caused a failure.
        """
        if not self.use_llm:
            return "LLM not enabled. Enable with use_llm=True for AI explanations."
        
        # Build context from findings
        context_parts = []
        
        if self.session.findings:
            context_parts.append("Previous findings:")
            for f in self.session.findings[-10:]:
                context_parts.append(f"- {f.description}")
        
        query = "Analyze this waveform and explain what might be causing failures. "
        
        if error_signal:
            query += f"Focus on signal '{error_signal}'. "
        if error_time:
            query += f"The error appears around time {error_time}. "
        
        if context_parts:
            query += "\n" + "\n".join(context_parts)
        
        response = self.chat.ask(query)
        
        explanation = response.content if hasattr(response, 'content') else str(response)
        
        # Save as root cause if looks definitive
        if "root cause" in explanation.lower() or "caused by" in explanation.lower():
            self.session.root_cause = explanation
        
        return explanation
    
    def find_glitches(self, signal: str, min_width: int = 1) -> List[DebugFinding]:
        """
        Find signal glitches (very short pulses).
        
        Args:
            signal: Signal to check
            min_width: Minimum pulse width to NOT be a glitch (in ns)
        """
        findings = []
        values = self.parser.get_signal_values(signal)
        
        for i in range(1, len(values) - 1):
            prev_time, prev_val = values[i - 1]
            curr_time, curr_val = values[i]
            next_time, next_val = values[i + 1]
            
            pulse_width = next_time - curr_time
            
            if pulse_width < min_width and prev_val == next_val:
                findings.append(DebugFinding(
                    severity="warning",
                    category="timing",
                    signal=signal,
                    time=curr_time,
                    description=f"Glitch detected: {pulse_width}ns pulse to {curr_val}",
                    context={
                        "pulse_width": pulse_width,
                        "glitch_value": curr_val,
                        "stable_value": prev_val
                    }
                ))
        
        self.session.findings.extend(findings)
        return findings
    
    def find_metastability(self, data_signal: str, 
                           clock_signal: str,
                           setup_time: int = 1,
                           hold_time: int = 1) -> List[DebugFinding]:
        """
        Find potential metastability issues (setup/hold violations).
        """
        findings = []
        
        data_values = self.parser.get_signal_values(data_signal)
        clock_values = self.parser.get_signal_values(clock_signal)
        
        # Find rising clock edges
        clock_edges = []
        for i in range(1, len(clock_values)):
            prev_time, prev_val = clock_values[i - 1]
            curr_time, curr_val = clock_values[i]
            
            if prev_val == 0 and curr_val == 1:
                clock_edges.append(curr_time)
        
        # Check data stability around each clock edge
        data_map = {t: v for t, v in data_values}
        
        for edge_time in clock_edges:
            # Check setup window
            setup_changes = []
            for t, v in data_values:
                if edge_time - setup_time <= t < edge_time:
                    setup_changes.append(t)
            
            if setup_changes:
                findings.append(DebugFinding(
                    severity="critical",
                    category="timing",
                    signal=data_signal,
                    time=edge_time,
                    description=f"Setup violation: data changed at {setup_changes[-1]}, edge at {edge_time}",
                    context={
                        "edge_time": edge_time,
                        "data_change_time": setup_changes[-1],
                        "margin": edge_time - setup_changes[-1]
                    },
                    suggested_fix="Add pipeline stage or adjust clock phase"
                ))
            
            # Check hold window
            hold_changes = []
            for t, v in data_values:
                if edge_time < t <= edge_time + hold_time:
                    hold_changes.append(t)
            
            if hold_changes:
                findings.append(DebugFinding(
                    severity="critical",
                    category="timing",
                    signal=data_signal,
                    time=edge_time,
                    description=f"Hold violation: data changed at {hold_changes[0]}, edge at {edge_time}",
                    context={
                        "edge_time": edge_time,
                        "data_change_time": hold_changes[0],
                        "margin": hold_changes[0] - edge_time
                    },
                    suggested_fix="Add delay on data path or clock"
                ))
        
        self.session.findings.extend(findings)
        return findings
    
    def bookmark(self, time: int, note: str = ""):
        """Add a bookmark at a specific time."""
        self.session.bookmarks.append({
            "time": time,
            "note": note,
            "created": datetime.now().isoformat()
        })
    
    def add_note(self, note: str):
        """Add a debug note to the session."""
        self.session.notes.append(note)
    
    def get_summary(self) -> str:
        """Get debug session summary."""
        lines = [
            "=" * 50,
            "DEBUG SESSION SUMMARY",
            "=" * 50,
            f"File: {self.session.vcd_file}",
            f"Duration: {datetime.now() - self.session.start_time}",
            "",
            f"Findings: {len(self.session.findings)}",
        ]
        
        # Group by severity
        by_severity = {}
        for f in self.session.findings:
            by_severity.setdefault(f.severity, []).append(f)
        
        for sev in ["critical", "warning", "info"]:
            if sev in by_severity:
                lines.append(f"  {sev.upper()}: {len(by_severity[sev])}")
        
        if self.session.root_cause:
            lines.extend([
                "",
                "Root Cause:",
                self.session.root_cause[:200] + "..." if len(self.session.root_cause) > 200 else self.session.root_cause
            ])
        
        if self.session.bookmarks:
            lines.extend([
                "",
                f"Bookmarks: {len(self.session.bookmarks)}"
            ])
            for bm in self.session.bookmarks[:5]:
                lines.append(f"  - t={bm['time']}: {bm['note']}")
        
        lines.append("=" * 50)
        return "\n".join(lines)
    
    def save_session(self, output_file: str = None):
        """Save debug session to file."""
        output_file = output_file or self.session.vcd_file.replace('.vcd', '_debug.json')
        
        data = {
            "vcd_file": self.session.vcd_file,
            "start_time": self.session.start_time.isoformat(),
            "findings": [
                {
                    "severity": f.severity,
                    "category": f.category,
                    "signal": f.signal,
                    "time": f.time,
                    "description": f.description,
                    "suggested_fix": f.suggested_fix
                }
                for f in self.session.findings
            ],
            "bookmarks": self.session.bookmarks,
            "notes": self.session.notes,
            "root_cause": self.session.root_cause
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return output_file


class StateMachineAnalyzer:
    """
    Analyze and debug state machines in waveforms.
    
    Usage:
        fsm = StateMachineAnalyzer("simulation.vcd", "state")
        
        # Learn states from waveform
        fsm.learn_states()
        
        # Find invalid transitions
        issues = fsm.find_issues()
        
        # Visualize state machine
        dot = fsm.to_graphviz()
    """
    
    def __init__(self, vcd_file: str, state_signal: str):
        from waveformgpt import VCDParser
        
        self.parser = VCDParser(vcd_file)
        self.state_signal = state_signal
        
        self.states: Set[Any] = set()
        self.transitions: List[Tuple[Any, Any, int]] = []  # (from, to, time)
        self.valid_transitions: Set[Tuple[Any, Any]] = set()
    
    def learn_states(self):
        """Learn states and transitions from waveform."""
        values = self.parser.get_signal_values(self.state_signal)
        
        prev_state = None
        
        for time, state in values:
            self.states.add(state)
            
            if prev_state is not None and prev_state != state:
                self.transitions.append((prev_state, state, time))
                self.valid_transitions.add((prev_state, state))
            
            prev_state = state
    
    def define_valid_transitions(self, transitions: List[Tuple[Any, Any]]):
        """Define which transitions are valid."""
        self.valid_transitions = set(transitions)
    
    def find_issues(self) -> List[DebugFinding]:
        """Find state machine issues."""
        findings = []
        
        # Find invalid transitions
        for from_state, to_state, time in self.transitions:
            if (from_state, to_state) not in self.valid_transitions:
                findings.append(DebugFinding(
                    severity="critical",
                    category="state",
                    signal=self.state_signal,
                    time=time,
                    description=f"Invalid transition: {from_state} -> {to_state}",
                    context={
                        "from_state": from_state,
                        "to_state": to_state
                    }
                ))
        
        # Find stuck states (no outgoing transitions)
        states_with_outgoing = set(t[0] for t in self.transitions)
        stuck_states = self.states - states_with_outgoing
        
        for state in stuck_states:
            # Find when we entered this state
            for from_state, to_state, time in self.transitions:
                if to_state == state:
                    findings.append(DebugFinding(
                        severity="warning",
                        category="state",
                        signal=self.state_signal,
                        time=time,
                        description=f"State {state} has no outgoing transitions (stuck state)",
                        context={"stuck_state": state}
                    ))
                    break
        
        return findings
    
    def get_time_in_states(self) -> Dict[Any, int]:
        """Calculate time spent in each state."""
        values = self.parser.get_signal_values(self.state_signal)
        end_time = self.parser.get_time_range()[1]
        
        time_in_state = {s: 0 for s in self.states}
        
        for i in range(len(values)):
            state_start, state = values[i]
            
            if i + 1 < len(values):
                state_end = values[i + 1][0]
            else:
                state_end = end_time
            
            time_in_state[state] += state_end - state_start
        
        return time_in_state
    
    def to_graphviz(self) -> str:
        """Generate Graphviz DOT representation."""
        lines = ["digraph FSM {"]
        lines.append("  rankdir=LR;")
        lines.append("  node [shape=circle];")
        
        # Add nodes
        for state in self.states:
            lines.append(f'  "{state}";')
        
        # Add edges with transition counts
        edge_counts = {}
        for from_s, to_s, _ in self.transitions:
            key = (from_s, to_s)
            edge_counts[key] = edge_counts.get(key, 0) + 1
        
        for (from_s, to_s), count in edge_counts.items():
            lines.append(f'  "{from_s}" -> "{to_s}" [label="{count}x"];')
        
        lines.append("}")
        return "\n".join(lines)
    
    def print_summary(self):
        """Print state machine summary."""
        print("\n" + "=" * 50)
        print(f"STATE MACHINE: {self.state_signal}")
        print("=" * 50)
        print(f"States: {len(self.states)}")
        print(f"Transitions: {len(self.transitions)}")
        print(f"Unique transitions: {len(self.valid_transitions)}")
        
        print("\nTime in states:")
        time_map = self.get_time_in_states()
        total = sum(time_map.values())
        
        for state, time in sorted(time_map.items(), key=lambda x: -x[1]):
            pct = (time / total * 100) if total > 0 else 0
            bar = "█" * int(pct / 5)
            print(f"  {state}: {time}ns ({pct:.1f}%) {bar}")
        
        print("=" * 50)
