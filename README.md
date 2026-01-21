# WaveformGPT 🌊🤖

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **Query simulation waveforms in plain English — now with LLM superpowers!**

WaveformGPT lets you ask questions about VCD/FST waveforms in natural language and get precise answers. Works offline with pattern matching, or use AI for unlimited flexibility.

## 🎯 The Problem

Debugging waveforms is tedious:
- 😵 Scroll through millions of cycles
- 🔍 Manually set markers and cursors  
- 📜 Write TCL scripts for complex queries
- 🤔 Remember exact signal names and timing

**WaveformGPT makes it conversational:**

```
You: "Show me when req went high but ack didn't follow within 10 cycles"
WaveformGPT: Found 3 violations at cycles 1045, 8923, 15678

You: "What's the average latency between valid and ready?"
WaveformGPT: 4.7 cycles (min: 1, max: 23, std: 2.1)

You: "Is there a glitch on the data bus?"  ← Requires LLM mode
WaveformGPT: Yes, detected 2 glitches at t=1523ns and t=8901ns...
```

## 🚀 Features

| Feature | Pattern Mode | LLM Mode |
|---------|:------------:|:--------:|
| Edge detection (rise/fall) | ✅ | ✅ |
| Signal counting | ✅ | ✅ |
| Timing measurements | ✅ | ✅ |
| Protocol violations | ✅ | ✅ |
| **Arbitrary questions** | ❌ | ✅ |
| **Complex analysis** | ❌ | ✅ |
| **Explanations** | ❌ | ✅ |
| Works offline | ✅ | Ollama only |
| Free | ✅ | Ollama only |

## 📦 Installation

```bash
pip install waveformgpt

# For LLM support (optional)
pip install openai anthropic
```

## 🎮 Quick Start

### Pattern-Based Mode (Offline)

```python
from waveformgpt import WaveformChat

# Load and query
chat = WaveformChat("simulation.vcd")
response = chat.ask("When does clk rise?")
print(response.content)
# → Found 1000 rising edges at t=10, 30, 50, 70...
```

### LLM-Powered Mode (Unlimited Questions)

```python
from waveformgpt import WaveformChat

# Set your API key
# export OPENAI_API_KEY="sk-..."
# OR export ANTHROPIC_API_KEY="sk-ant-..."

chat = WaveformChat("simulation.vcd", use_llm=True)

# Ask ANYTHING!
chat.ask("What's the clock frequency?")
chat.ask("Is there a handshake between req and ack?")
chat.ask("Describe the state machine behavior")
chat.ask("Find any protocol violations")
chat.ask("What caused the error flag to trigger?")
```

## 🤖 LLM Backend Options

### 1. OpenAI (Recommended)
```bash
export OPENAI_API_KEY="sk-..."
```
- Best accuracy
- Fast responses
- ~$0.01 per complex query

### 2. Anthropic Claude
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```
- Excellent reasoning
- Good at explanations

### 3. Ollama (Free & Private)
```bash
# Install Ollama
brew install ollama  # macOS
# or download from https://ollama.com

# Start server and download model
ollama serve
ollama pull llama3.1

# Now WaveformGPT auto-detects it!
```
- 100% free
- Runs locally (private)
- Works offline
- Good for basic queries

## 📊 What You Can Ask

### Pattern Mode (Always Works)
```
When does [signal] rise/fall?
How many times does [signal] change?
Show me [signal] followed by [signal2]
Find all [signal] events
What's the time between [signal1] and [signal2]?
```

### LLM Mode (Unlimited)
```
What's the clock duty cycle?
Is this following the AXI4 protocol correctly?
Why did the state machine get stuck?
What signals change when error goes high?
Summarize the transaction flow
Find race conditions between signals
What's the typical handshake latency?
Are there any glitches or hazards?
Explain what happens at t=1500ns
```

## 🏗️ Full Feature Set

- **VCD/FST Parsing**: Streaming for large files
- **Natural Language**: Pattern + LLM understanding
- **Protocol Checkers**: AXI4, Wishbone B4, SPI
- **Temporal Assertions**: SVA-like syntax
- **Waveform Comparison**: Diff two simulations
- **Visualization**: ASCII art + HTML rendering
- **Export**: CSV, JSON, WaveDrom, SystemVerilog, cocotb

## 🔧 Advanced Usage

### Protocol Checking
```python
from waveformgpt import AXI4Checker, VCDParser

parser = VCDParser("axi_transaction.vcd")
checker = AXI4Checker(parser, signal_prefix="axi_")
violations = checker.check_all()

for v in violations:
    print(f"{v.severity}: {v.message} at t={v.time}")
```

### Temporal Assertions
```python
from waveformgpt import AssertionChecker, VCDParser

parser = VCDParser("design.vcd")
checker = AssertionChecker(parser)

# Request must be followed by ack within 10 cycles
checker.check("rose(req) |-> ##[1:10] rose(ack)")
```

### Waveform Comparison
```python
from waveformgpt import WaveformComparator

comp = WaveformComparator("golden.vcd", "test.vcd")
result = comp.compare(signals=["data", "valid", "ready"])
print(f"Differences: {len(result.differences)}")
```

### ASCII Visualization
```python
from waveformgpt import ASCIIWaveform, VCDParser

parser = VCDParser("test.vcd")
viz = ASCIIWaveform(parser, width=60)
print(viz.render(["clk", "data", "valid"]))
```

Output:
```
clk:   ▁▁▀▀▀▁▁▁▀▀▀▁▁▁▀▀▀▁▁▁▀▀▀▁▁▁▀▀▀▁▁
data:  ▁▁▁▁▁▁▁▁▁████████▁▁▁▁▁▁▁▁████▁▁
valid: ▁▁▁▁▁▁▁▀▀▀▀▀▀▀▀▀▀▀▁▁▁▁▁▀▀▀▀▀▁▁▁
```

## 📁 Project Structure

```
WaveformGPT/
├── src/waveformgpt/
│   ├── chat.py          # Main chat interface
│   ├── llm_engine.py    # LLM backends (OpenAI, Claude, Ollama)
│   ├── vcd_parser.py    # VCD file parsing
│   ├── nl_parser.py     # Pattern-based NL parser
│   ├── query_engine.py  # Query execution
│   ├── analyzer.py      # Timing analysis
│   ├── protocols.py     # Protocol checkers
│   ├── assertions.py    # Temporal assertions
│   ├── compare.py       # Waveform comparison
│   ├── visualize.py     # ASCII/HTML visualization
│   └── export.py        # Multi-format export
├── demo/
│   ├── demo.vcd         # Sample waveform
│   ├── run_demo.py      # Basic demo
│   └── llm_demo.py      # LLM-powered demo
└── tests/               # Test suite
```

## 🎓 How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                     WaveformGPT Pipeline                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │   User   │───▶│  Pattern │───▶│  Query   │              │
│  │  Query   │    │  Parser  │    │  Engine  │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│        │               │               │                    │
│        │ (if failed)   │               │                    │
│        ▼               ▼               ▼                    │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │   LLM    │───▶│   Code   │───▶│   VCD    │              │
│  │ Backend  │    │   Gen    │    │  Parser  │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│        │                               │                    │
│        └───────────────┬───────────────┘                    │
│                        ▼                                    │
│              ┌──────────────────┐                          │
│              │     Response     │                          │
│              │  • Answer        │                          │
│              │  • Reasoning     │                          │
│              │  • Visualization │                          │
│              └──────────────────┘                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 🔌 Supported LLM Providers

| Provider | Model | Setup | Cost |
|----------|-------|-------|------|
| OpenAI | GPT-4o | `OPENAI_API_KEY` | ~$0.01/query |
| Anthropic | Claude 3.5 | `ANTHROPIC_API_KEY` | ~$0.01/query |
| Ollama | Llama 3.1 | Local server | Free |
| Ollama | Mistral | Local server | Free |
| Ollama | CodeLlama | Local server | Free |

## 🛠️ Development

```bash
# Clone
git clone https://github.com/tusharpathaknyu/WaveformGPT.git
cd WaveformGPT

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run demo
python demo/run_demo.py        # Pattern mode
python demo/llm_demo.py        # LLM mode
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🌟 Contributing

Contributions welcome! Please open an issue or PR.

---

**Made for the hardware verification community** 🔧⚡
