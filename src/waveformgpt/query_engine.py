"""
Query Engine for WaveformGPT.

Translates structured queries into waveform analysis operations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from enum import Enum
import re

from waveformgpt.vcd_parser import VCDParser, Signal, ValueChange


class QueryType(Enum):
    """Types of queries supported."""
    FIND_EVENT = "find_event"          # When does X happen?
    FIND_PATTERN = "find_pattern"       # Find A followed by B
    COUNT = "count"                     # How many times does X happen?
    MEASURE = "measure"                 # Measure time between events
    STATISTICS = "statistics"           # Min/max/avg for timing
    COMPARE = "compare"                 # Compare two signals


@dataclass
class EventCondition:
    """Condition for matching signal events."""
    signal: str
    condition: str  # "rise", "fall", "high", "low", "change", "value"
    value: Optional[str] = None  # For value match


@dataclass
class TemporalConstraint:
    """Temporal constraint between events."""
    min_cycles: int = 0
    max_cycles: Optional[int] = None
    exact_cycles: Optional[int] = None


@dataclass
class Query:
    """Structured query representation."""
    type: QueryType
    events: List[EventCondition] = field(default_factory=list)
    temporal: Optional[TemporalConstraint] = None
    time_range: Optional[tuple] = None
    limit: Optional[int] = None


@dataclass 
class QueryMatch:
    """A single match result."""
    time: int
    end_time: Optional[int] = None
    signals: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """Complete query result."""
    query: Query
    matches: List[QueryMatch] = field(default_factory=list)
    statistics: Optional[Dict[str, float]] = None
    answer: str = ""
    
    @property
    def count(self) -> int:
        return len(self.matches)


class QueryEngine:
    """
    Executes queries against VCD waveform data.
    """
    
    def __init__(self, parser: VCDParser):
        self.parser = parser
        self._signal_cache: Dict[str, List[tuple]] = {}
    
    def execute(self, query: Query) -> QueryResult:
        """Execute a structured query."""
        if query.type == QueryType.FIND_EVENT:
            return self._find_event(query)
        elif query.type == QueryType.FIND_PATTERN:
            return self._find_pattern(query)
        elif query.type == QueryType.COUNT:
            return self._count_events(query)
        elif query.type == QueryType.MEASURE:
            return self._measure_timing(query)
        elif query.type == QueryType.STATISTICS:
            return self._calculate_statistics(query)
        else:
            return QueryResult(query=query, answer="Unsupported query type")
    
    def _get_signal_values(self, signal_name: str) -> List[tuple]:
        """Get cached signal values."""
        if signal_name not in self._signal_cache:
            self._signal_cache[signal_name] = self.parser.get_signal_values(signal_name)
        return self._signal_cache[signal_name]
    
    def _find_event(self, query: Query) -> QueryResult:
        """Find occurrences of a single event."""
        if not query.events:
            return QueryResult(query=query, answer="No event specified")
        
        event = query.events[0]
        matches = self._match_event(event, query.time_range)
        
        if query.limit:
            matches = matches[:query.limit]
        
        result = QueryResult(
            query=query,
            matches=matches,
            answer=f"Found {len(matches)} occurrence(s) of {event.signal} {event.condition}"
        )
        
        if matches:
            times = [m.time for m in matches]
            result.answer += f" at times: {times[:10]}"
            if len(times) > 10:
                result.answer += f"... and {len(times) - 10} more"
        
        return result
    
    def _match_event(self, event: EventCondition, 
                     time_range: Optional[tuple] = None) -> List[QueryMatch]:
        """Find all matches for an event condition."""
        matches = []
        values = self._get_signal_values(event.signal)
        
        prev_value = None
        for time, value in values:
            if time_range:
                if time < time_range[0]:
                    prev_value = value
                    continue
                if time > time_range[1]:
                    break
            
            match = self._check_condition(event, value, prev_value)
            if match:
                matches.append(QueryMatch(
                    time=time,
                    signals={event.signal: value}
                ))
            
            prev_value = value
        
        return matches
    
    def _check_condition(self, event: EventCondition, 
                         value: str, prev_value: Optional[str]) -> bool:
        """Check if a value change matches the event condition."""
        if event.condition == "rise":
            return prev_value == "0" and value == "1"
        elif event.condition == "fall":
            return prev_value == "1" and value == "0"
        elif event.condition == "high":
            return value == "1"
        elif event.condition == "low":
            return value == "0"
        elif event.condition == "change":
            return prev_value is not None and value != prev_value
        elif event.condition == "value":
            return value == event.value
        return False
    
    def _find_pattern(self, query: Query) -> QueryResult:
        """Find temporal pattern (A followed by B within N cycles)."""
        if len(query.events) < 2:
            return QueryResult(query=query, answer="Pattern requires at least 2 events")
        
        matches = []
        first_event = query.events[0]
        second_event = query.events[1]
        
        first_matches = self._match_event(first_event, query.time_range)
        second_values = self._get_signal_values(second_event.signal)
        
        for first_match in first_matches:
            # Look for second event after first
            found_second = self._find_next_event(
                second_event, 
                second_values,
                first_match.time,
                query.temporal
            )
            
            if found_second:
                matches.append(QueryMatch(
                    time=first_match.time,
                    end_time=found_second.time,
                    signals={
                        first_event.signal: first_match.signals[first_event.signal],
                        second_event.signal: found_second.signals[second_event.signal]
                    },
                    metadata={"delay": found_second.time - first_match.time}
                ))
        
        result = QueryResult(
            query=query,
            matches=matches,
            answer=f"Found {len(matches)} pattern matches"
        )
        
        return result
    
    def _find_next_event(self, event: EventCondition,
                         values: List[tuple],
                         after_time: int,
                         constraint: Optional[TemporalConstraint]) -> Optional[QueryMatch]:
        """Find the next occurrence of an event after a given time."""
        prev_value = None
        
        for time, value in values:
            if time <= after_time:
                prev_value = value
                continue
            
            # Check temporal constraint
            if constraint:
                delay = time - after_time
                if constraint.min_cycles and delay < constraint.min_cycles:
                    prev_value = value
                    continue
                if constraint.max_cycles and delay > constraint.max_cycles:
                    return None
                if constraint.exact_cycles and delay != constraint.exact_cycles:
                    prev_value = value
                    continue
            
            if self._check_condition(event, value, prev_value):
                return QueryMatch(time=time, signals={event.signal: value})
            
            prev_value = value
        
        return None
    
    def _count_events(self, query: Query) -> QueryResult:
        """Count occurrences of an event."""
        result = self._find_event(query)
        result.answer = f"Count: {len(result.matches)}"
        return result
    
    def _measure_timing(self, query: Query) -> QueryResult:
        """Measure timing between pattern occurrences."""
        pattern_result = self._find_pattern(query)
        
        if pattern_result.matches:
            delays = [m.metadata.get("delay", 0) for m in pattern_result.matches]
            pattern_result.statistics = {
                "min": min(delays),
                "max": max(delays),
                "avg": sum(delays) / len(delays),
                "count": len(delays)
            }
            pattern_result.answer = (
                f"Timing: min={min(delays)}, max={max(delays)}, "
                f"avg={sum(delays)/len(delays):.2f} ({len(delays)} samples)"
            )
        
        return pattern_result
    
    def _calculate_statistics(self, query: Query) -> QueryResult:
        """Calculate statistics for a signal or pattern."""
        if not query.events:
            return QueryResult(query=query, answer="No event specified")
        
        # For single event: count transitions
        event = query.events[0]
        matches = self._match_event(event, query.time_range)
        
        if len(matches) < 2:
            return QueryResult(
                query=query,
                matches=matches,
                answer="Not enough data for statistics"
            )
        
        # Calculate inter-event timing
        intervals = []
        for i in range(1, len(matches)):
            intervals.append(matches[i].time - matches[i-1].time)
        
        stats = {
            "count": len(matches),
            "min_interval": min(intervals),
            "max_interval": max(intervals),
            "avg_interval": sum(intervals) / len(intervals)
        }
        
        return QueryResult(
            query=query,
            matches=matches,
            statistics=stats,
            answer=(
                f"Statistics for {event.signal} {event.condition}: "
                f"{len(matches)} occurrences, interval min={min(intervals)}, "
                f"max={max(intervals)}, avg={sum(intervals)/len(intervals):.2f}"
            )
        )
    
    def find_signal_value_at(self, signal: str, time: int) -> Optional[str]:
        """Get the value of a signal at a specific time."""
        values = self._get_signal_values(signal)
        
        last_value = None
        for t, v in values:
            if t > time:
                break
            last_value = v
        
        return last_value
    
    def get_signal_changes_around(self, time: int, 
                                   window: int = 10,
                                   signals: Optional[List[str]] = None) -> Dict[str, List[tuple]]:
        """Get all signal changes within a time window."""
        result = {}
        
        signal_list = signals or [s.name for s in self.parser.header.signals.values()]
        
        for sig_name in signal_list:
            try:
                values = self._get_signal_values(sig_name)
                changes = [(t, v) for t, v in values if time - window <= t <= time + window]
                if changes:
                    result[sig_name] = changes
            except ValueError:
                continue
        
        return result
