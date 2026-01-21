# WaveformGPT - Natural Language Waveform Query

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Query simulation waveforms in plain English**

WaveformGPT lets you ask questions about VCD/FST waveforms in natural language and get precise answers with highlighted regions.

## 🎯 Problem Statement

Debugging waveforms is tedious:
- Scroll through millions of cycles
- Manually set markers and cursors
- Write TCL scripts for complex queries
- Remember exact signal names and timing

**WaveformGPT makes it conversational:**

```
You: "Show me when req went high but ack didn't follow within 10 cycles"
WaveformGPT: Found 3 violations at cycles 1045, 8923, 15678 [Opens in viewer]

You: "Find the last write transaction before the error flag"
WaveformGPT: Write at cycle 4521, error at 4530. Here's the region...

You: "What's the average latency between valid and ready?"
WaveformGPT: 4.7 cycles (min: 1, max: 23, std: 2.1)
```

## 🚀 Key Features

- **Natural language queries**: No scripting required
- **Multi-format support**: VCD, FST, GHW waveforms
- **GTKWave integration**: Opens results directly in viewer
- **Temporal patterns**: Before/after, within N cycles, sequences
- **Statistics**: Min/max/avg/histogram for timing analysis
- **Bookmarks**: Save interesting regions for later
- **Batch mode**: Script complex analyses

## 📊 Query Examples

| Natural Language | What It Does |
|-----------------|--------------|
| "When does clock stop toggling?" | Find clock gating events |
| "Show all reset assertions" | Find `rst_n` low periods |
| "Find setup violations on data vs clock" | Timing analysis |
| "What signals change when error goes high?" | Root cause analysis |
| "Count transitions on busy signal" | Activity analysis |
| "Find the longest burst transaction" | Pattern matching |

## 🔧 Installation

```bash
pip install waveformgpt

# With GTKWave integration
pip install waveformgpt[gtkwave]
```

## 📖 Quick Start

```python
from waveformgpt import WaveformChat

# Load waveform
chat = WaveformChat("simulation.vcd")

# Ask questions
result = chat.query("When does the FIFO overflow?")
print(result.answer)
# "FIFO overflow detected at cycles: 10234, 45678, 89012"

# Get detailed timing
result = chat.query("What's the read latency distribution?")
print(result.histogram)

# Open in GTKWave at specific time
result = chat.query("Show me the first AXI error")
result.open_in_gtkwave()
```

## 🖥️ CLI Usage

```bash
# Interactive mode
waveformgpt chat simulation.vcd

# Single query
waveformgpt query simulation.vcd "Find protocol violations"

# Batch mode
waveformgpt batch simulation.vcd queries.txt -o report.html
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     WaveformGPT Pipeline                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  Natural │───▶│  Query   │───▶│ Waveform │              │
│  │ Language │    │  Parser  │    │  Engine  │              │
│  │  Query   │    │  (LLM)   │    │          │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│                        │               │                    │
│                        ▼               ▼                    │
│                  ┌──────────┐    ┌──────────┐              │
│                  │ Temporal │    │   VCD    │              │
│                  │ Pattern  │◀──▶│  Parser  │              │
│                  │  Matcher │    │          │              │
│                  └──────────┘    └──────────┘              │
│                        │                                    │
│                        ▼                                    │
│                  ┌──────────────────────────┐              │
│                  │  Results + Visualization │              │
│                  │  • Time ranges           │              │
│                  │  • Statistics            │              │
│                  │  • GTKWave markers       │              │
│                  └──────────────────────────┘              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
