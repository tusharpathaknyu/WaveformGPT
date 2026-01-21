"""
Natural Language Processing for WaveformGPT.

Translates natural language queries into structured Query objects.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import re
import json

from waveformgpt.query_engine import Query, QueryType, EventCondition, TemporalConstraint


# Patterns for parsing natural language queries
PATTERNS = {
    "when": r"when\s+(?:does|did|is|was)?\s*(\w+)\s+(go|went|become|became|rise|fall|change)",
    "find": r"find\s+(?:all\s+)?(?:times?\s+)?(?:when\s+)?(\w+)\s+(rises?|falls?|changes?|high|low)",
    "show": r"show\s+(?:me\s+)?(?:when\s+)?(\w+)\s+(rises?|falls?|goes?\s+high|goes?\s+low)",
    "count": r"(?:how\s+many|count)\s+(?:times?\s+)?(?:does\s+)?(\w+)\s+(rise|fall|change|transition)",
    "measure": r"(?:measure|what(?:'s| is))\s+(?:the\s+)?(?:time|delay|latency)\s+(?:between|from)\s+(\w+)\s+(?:and|to)\s+(\w+)",
    "pattern": r"(\w+)\s+(?:followed by|then|before)\s+(\w+)(?:\s+within\s+(\d+)\s+(?:cycles?|ns|clocks?))?",
    "average": r"(?:average|mean)\s+(?:time|delay|latency)\s+(?:between|from)\s+(\w+)\s+(?:and|to)\s+(\w+)",
    "value_at": r"(?:what(?:'s| is))\s+(?:the\s+)?(?:value of\s+)?(\w+)\s+at\s+(?:time\s+)?(\d+)",
}

CONDITION_MAP = {
    "rise": "rise", "rises": "rise", "rising": "rise", "rose": "rise", "go high": "rise", "goes high": "rise", "went high": "rise",
    "fall": "fall", "falls": "fall", "falling": "fall", "fell": "fall", "go low": "fall", "goes low": "fall", "went low": "fall",
    "change": "change", "changes": "change", "changed": "change", "transition": "change", "transitions": "change",
    "high": "high", "is high": "high", "was high": "high",
    "low": "low", "is low": "low", "was low": "low",
}


@dataclass
class ParsedQuery:
    """Result of parsing a natural language query."""
    query: Optional[Query]
    confidence: float
    raw_text: str
    interpretation: str
    

class NLParser:
    """
    Parses natural language queries into structured Query objects.
    Uses pattern matching and optional LLM enhancement.
    """
    
    def __init__(self, available_signals: Optional[List[str]] = None, use_llm: bool = False):
        self.available_signals = available_signals or []
        self.use_llm = use_llm
        self._llm_client = None
    
    def parse(self, text: str) -> ParsedQuery:
        """Parse natural language text into a Query."""
        text_lower = text.lower().strip()
        
        # Try pattern-based parsing first
        result = self._pattern_parse(text_lower)
        if result.query:
            return result
        
        # Fall back to LLM if enabled
        if self.use_llm:
            return self._llm_parse(text)
        
        return ParsedQuery(
            query=None,
            confidence=0.0,
            raw_text=text,
            interpretation="Could not parse query"
        )
    
    def _pattern_parse(self, text: str) -> ParsedQuery:
        """Pattern-based query parsing."""
        
        # Try "when" pattern
        match = re.search(PATTERNS["when"], text)
        if match:
            signal = match.group(1)
            condition = self._normalize_condition(match.group(2))
            return ParsedQuery(
                query=Query(
                    type=QueryType.FIND_EVENT,
                    events=[EventCondition(signal=signal, condition=condition)]
                ),
                confidence=0.8,
                raw_text=text,
                interpretation=f"Find when {signal} {condition}s"
            )
        
        # Try "find" pattern
        match = re.search(PATTERNS["find"], text)
        if match:
            signal = match.group(1)
            condition = self._normalize_condition(match.group(2))
            return ParsedQuery(
                query=Query(
                    type=QueryType.FIND_EVENT,
                    events=[EventCondition(signal=signal, condition=condition)]
                ),
                confidence=0.85,
                raw_text=text,
                interpretation=f"Find all {signal} {condition} events"
            )
        
        # Try "show" pattern  
        match = re.search(PATTERNS["show"], text)
        if match:
            signal = match.group(1)
            condition = self._normalize_condition(match.group(2))
            return ParsedQuery(
                query=Query(
                    type=QueryType.FIND_EVENT,
                    events=[EventCondition(signal=signal, condition=condition)]
                ),
                confidence=0.8,
                raw_text=text,
                interpretation=f"Show {signal} {condition} events"
            )
        
        # Try "count" pattern
        match = re.search(PATTERNS["count"], text)
        if match:
            signal = match.group(1)
            condition = self._normalize_condition(match.group(2))
            return ParsedQuery(
                query=Query(
                    type=QueryType.COUNT,
                    events=[EventCondition(signal=signal, condition=condition)]
                ),
                confidence=0.9,
                raw_text=text,
                interpretation=f"Count {signal} {condition} events"
            )
        
        # Try "measure" pattern (timing between two signals)
        match = re.search(PATTERNS["measure"], text)
        if match:
            signal1 = match.group(1)
            signal2 = match.group(2)
            return ParsedQuery(
                query=Query(
                    type=QueryType.MEASURE,
                    events=[
                        EventCondition(signal=signal1, condition="rise"),
                        EventCondition(signal=signal2, condition="rise")
                    ]
                ),
                confidence=0.85,
                raw_text=text,
                interpretation=f"Measure timing from {signal1} to {signal2}"
            )
        
        # Try "pattern" (A followed by B)
        match = re.search(PATTERNS["pattern"], text)
        if match:
            signal1 = match.group(1)
            signal2 = match.group(2)
            max_delay = int(match.group(3)) if match.group(3) else None
            return ParsedQuery(
                query=Query(
                    type=QueryType.FIND_PATTERN,
                    events=[
                        EventCondition(signal=signal1, condition="rise"),
                        EventCondition(signal=signal2, condition="rise")
                    ],
                    temporal=TemporalConstraint(max_cycles=max_delay) if max_delay else None
                ),
                confidence=0.8,
                raw_text=text,
                interpretation=f"Find {signal1} followed by {signal2}"
            )
        
        # Try "average" pattern
        match = re.search(PATTERNS["average"], text)
        if match:
            signal1 = match.group(1)
            signal2 = match.group(2)
            return ParsedQuery(
                query=Query(
                    type=QueryType.STATISTICS,
                    events=[
                        EventCondition(signal=signal1, condition="rise"),
                        EventCondition(signal=signal2, condition="rise")
                    ]
                ),
                confidence=0.85,
                raw_text=text,
                interpretation=f"Calculate average timing from {signal1} to {signal2}"
            )
        
        return ParsedQuery(
            query=None,
            confidence=0.0,
            raw_text=text,
            interpretation="No pattern matched"
        )
    
    def _normalize_condition(self, condition_text: str) -> str:
        """Normalize various condition phrasings to standard conditions."""
        condition_text = condition_text.lower().strip()
        return CONDITION_MAP.get(condition_text, "change")
    
    def _llm_parse(self, text: str) -> ParsedQuery:
        """Use LLM to parse complex queries."""
        try:
            from openai import OpenAI
            
            if self._llm_client is None:
                self._llm_client = OpenAI()
            
            prompt = self._build_llm_prompt(text)
            
            response = self._llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            query = self._result_to_query(result)
            
            return ParsedQuery(
                query=query,
                confidence=result.get("confidence", 0.7),
                raw_text=text,
                interpretation=result.get("interpretation", "LLM parsed query")
            )
            
        except Exception as e:
            return ParsedQuery(
                query=None,
                confidence=0.0,
                raw_text=text,
                interpretation=f"LLM parsing failed: {str(e)}"
            )
    
    def _build_llm_prompt(self, text: str) -> Dict[str, str]:
        """Build prompt for LLM query parsing."""
        signals_hint = ""
        if self.available_signals:
            signals_hint = f"\nAvailable signals: {', '.join(self.available_signals[:20])}"
        
        system = f"""You are a waveform query parser. Convert natural language queries about digital signals into structured JSON.

Output JSON format:
{{
    "query_type": "find_event|find_pattern|count|measure|statistics",
    "events": [
        {{"signal": "signal_name", "condition": "rise|fall|high|low|change", "value": null}}
    ],
    "temporal": {{"min_cycles": 0, "max_cycles": null}},
    "interpretation": "human-readable interpretation",
    "confidence": 0.0-1.0
}}

Condition meanings:
- rise: signal transitions from 0 to 1
- fall: signal transitions from 1 to 0
- high: signal is 1
- low: signal is 0
- change: signal value changes
{signals_hint}
"""
        
        return {
            "system": system,
            "user": f"Parse this waveform query: {text}"
        }
    
    def _result_to_query(self, result: Dict[str, Any]) -> Optional[Query]:
        """Convert LLM result to Query object."""
        try:
            query_type = QueryType(result["query_type"])
            
            events = []
            for e in result.get("events", []):
                events.append(EventCondition(
                    signal=e["signal"],
                    condition=e["condition"],
                    value=e.get("value")
                ))
            
            temporal = None
            if result.get("temporal"):
                t = result["temporal"]
                temporal = TemporalConstraint(
                    min_cycles=t.get("min_cycles", 0),
                    max_cycles=t.get("max_cycles")
                )
            
            return Query(
                type=query_type,
                events=events,
                temporal=temporal
            )
        except Exception:
            return None
