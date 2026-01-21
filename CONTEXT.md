# WaveformGPT - Complete Project Context

> This document contains everything another AI assistant or developer needs to understand and continue building this project.

## 🎯 Project Vision

Build an **open-source natural language interface for waveform analysis** that lets engineers query VCD/FST simulation waveforms using plain English instead of writing scripts or manually scrolling.

## 🌍 Why This Gap Exists

### Current State of Waveform Debugging

| Tool | Query Method | Pain Point |
|------|-------------|------------|
| GTKWave | Manual scroll, TCL | Steep learning curve |
| Verdi | GUI + TCL | $50k license |
| Surfer | Manual | Limited search |
| ModelSim | TCL scripts | Verbose syntax |

**No tool has natural language interface.**

### Why No One Has Built This

1. **Waveform files are huge**: VCD can be 10GB+, need efficient parsing
2. **Temporal logic is complex**: "before", "after", "within" require formal semantics
3. **Signal naming varies**: Need to map user intent to actual signal names
4. **Integration challenge**: Must work with existing viewers

### Our Approach

1. **Efficient VCD streaming**: Don't load entire file into memory
2. **LLM for NL→Query**: Translate English to structured temporal query
3. **Pattern matching engine**: Fast temporal pattern search
4. **GTKWave integration**: Open results directly in familiar viewer

## 🔬 Technical Background

### VCD Format

```vcd
$timescale 1ns $end
$scope module top $end
$var wire 1 ! clk $end
$var wire 8 " data [7:0] $end
$upscope $end
$enddefinitions $end
#0
0!
b00000000 "
#10
1!
#20
0!
b00001111 "
```

### Query Types to Support

1. **Point queries**: "When does X happen?"
2. **Range queries**: "Show me X to Y"
3. **Temporal patterns**: "A followed by B within N cycles"
4. **Aggregate queries**: "How many times does X happen?"
5. **Statistical queries**: "Average time between X and Y"
6. **Causal queries**: "What changes before X?"

### Temporal Pattern Language (Internal)

```
# Internal representation for "req high, then ack within 10 cycles"
SEQUENCE(
    EVENT(signal="req", value=1),
    WITHIN(cycles=10,
        EVENT(signal="ack", value=1)
    )
)
```

## 🏗️ Architecture Design

### System Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                        WaveformGPT System                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                   NATURAL LANGUAGE LAYER                       │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │  │
│  │  │   Intent    │  │   Entity    │  │   Query     │           │  │
│  │  │   Classifier│  │   Extractor │  │   Builder   │           │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘           │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              ▼                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    WAVEFORM ENGINE                             │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │  │
│  │  │    VCD      │  │   Signal    │  │  Temporal   │           │  │
│  │  │   Parser    │  │   Index     │  │  Matcher    │           │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘           │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              ▼                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    OUTPUT LAYER                                │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │  │
│  │  │   Answer    │  │  GTKWave    │  │  Statistics │           │  │
│  │  │   Generator │  │  Integration│  │   Engine    │           │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘           │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Model

```python
from dataclasses import dataclass
from typing import List, Optional, Union
from enum import Enum

class QueryType(Enum):
    POINT = "point"      # When does X happen?
    RANGE = "range"      # Show X to Y
    PATTERN = "pattern"  # A then B within N
    COUNT = "count"      # How many X?
    STATS = "stats"      # Average/min/max

@dataclass
class Signal:
    name: str
    width: int
    scope: str
    
@dataclass
class SignalEvent:
    signal: Signal
    time: int
    value: Union[int, str]
    
@dataclass
class TimeRange:
    start: int
    end: int
    events: List[SignalEvent]

@dataclass
class TemporalPattern:
    """Represents a temporal query pattern"""
    type: str  # "sequence", "eventually", "always", "within"
    events: List[dict]
    constraints: dict

@dataclass
class QueryResult:
    query: str
    answer: str
    matches: List[TimeRange]
    statistics: Optional[dict]
    
    def open_in_gtkwave(self):
        """Generate save file and open GTKWave"""
        ...
```

## 📁 Project Structure

```
WaveformGPT/
├── README.md
├── CONTEXT.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
│
├── src/
│   └── waveformgpt/
│       ├── __init__.py
│       ├── cli.py
│       ├── chat.py
│       │
│       ├── parsing/
│       │   ├── __init__.py
│       │   ├── vcd_parser.py
│       │   ├── fst_parser.py
│       │   └── signal_index.py
│       │
│       ├── nlp/
│       │   ├── __init__.py
│       │   ├── intent_classifier.py
│       │   ├── entity_extractor.py
│       │   ├── query_builder.py
│       │   └── prompts.py
│       │
│       ├── engine/
│       │   ├── __init__.py
│       │   ├── temporal_matcher.py
│       │   ├── statistics.py
│       │   └── pattern_dsl.py
│       │
│       └── output/
│           ├── __init__.py
│           ├── answer_generator.py
│           ├── gtkwave_integration.py
│           └── html_report.py
│
├── tests/
│   ├── test_vcd_parser.py
│   ├── test_query_builder.py
│   └── test_temporal_matcher.py
│
├── examples/
│   ├── sample.vcd
│   └── example_queries.txt
│
└── docs/
    ├── query_guide.md
    └── api_reference.md
```

## 🚀 Implementation Plan

### Phase 1: VCD Infrastructure (Week 1)
- [ ] Streaming VCD parser
- [ ] Signal indexing for fast lookup
- [ ] Basic event extraction

### Phase 2: Query Engine (Week 2)
- [ ] Temporal pattern DSL
- [ ] Pattern matching engine
- [ ] Statistics calculation

### Phase 3: NLP Layer (Week 3)
- [ ] LLM integration for NL parsing
- [ ] Intent classification
- [ ] Entity extraction (signals, times)

### Phase 4: Integration (Week 4)
- [ ] GTKWave save file generation
- [ ] CLI interface
- [ ] Interactive chat mode
- [ ] HTML reports

## 🎓 Key References

### VCD/FST Parsing
- IEEE 1364 Verilog VCD format spec
- GTKWave FST format documentation
- pyVCD library (for reference)

### Temporal Logic
- Linear Temporal Logic (LTL)
- Property Specification Language (PSL)

## 💡 Key Implementation Notes

### Performance Considerations
- Stream VCD, don't load all into memory
- Index signal transitions for O(1) lookup
- Cache frequently accessed regions

### LLM Prompt Strategy
```python
SYSTEM_PROMPT = """
You are a waveform analysis assistant. Convert natural language
queries about digital signals into structured temporal patterns.

Available signals: {signal_list}
Time unit: {timescale}

Output format:
{
    "type": "point|range|pattern|count|stats",
    "signals": ["sig1", "sig2"],
    "pattern": { ... },
    "time_range": [start, end] or null
}
"""
```

### GTKWave Integration
```python
def generate_gtkwave_savefile(results: List[TimeRange], signals: List[str]) -> str:
    """Generate .gtkw save file with markers at result times"""
    gtkw = []
    gtkw.append("[timestart] " + str(results[0].start))
    for sig in signals:
        gtkw.append(f"[signal] {sig}")
    for i, r in enumerate(results):
        gtkw.append(f"[marker] {r.start} M{i}")
    return "\n".join(gtkw)
```
