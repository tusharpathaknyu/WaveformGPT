# WaveformGPT 🌊🤖

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **Query simulation waveforms in plain English — with LLM, Voice & Live streaming!**

WaveformGPT lets you **talk to your waveforms**. Ask questions in natural language (or by voice!), analyze live simulations, and get AI-powered insights.

## 🎯 The Problem

Debugging waveforms is tedious:
- 😵 Scroll through millions of cycles
- 🔍 Manually set markers and cursors  
- 📜 Write TCL scripts for complex queries
- 🤔 Remember exact signal names and timing

**WaveformGPT makes it conversational:**

```
🎤 You: "Hey WaveformGPT, what's the clock frequency?"
🤖 WaveformGPT: "The clock frequency is 50MHz with a 50% duty cycle."

🎤 You: "Is there a handshake between req and ack?"
🤖 WaveformGPT: "Yes! At 40ns req goes high, ack acknowledges at 70ns..."

🎤 You: "Alert me when error goes high"  
🤖 WaveformGPT: [Monitoring...] 🚨 Alert! Error triggered at t=1523ns!
```

## 🚀 Features

| Feature | Pattern Mode | LLM Mode | Voice Mode | Live Mode |
|---------|:------------:|:--------:|:----------:|:---------:|
| Edge detection | ✅ | ✅ | ✅ | ✅ |
| Timing analysis | ✅ | ✅ | ✅ | ✅ |
| Arbitrary questions | ❌ | ✅ | ✅ | ✅ |
| Voice input/output | ❌ | ❌ | ✅ | ✅ |
| Live streaming | ❌ | ❌ | ❌ | ✅ |
| Real-time alerts | ❌ | ❌ | ❌ | ✅ |
| Works offline | ✅ | Ollama | ❌ | ✅ |

## 📦 Installation

```bash
pip install waveformgpt

# For LLM support
pip install openai anthropic

# For voice support  
pip install pyaudio  # macOS: brew install portaudio first

# For live WebSocket streaming
pip install websockets
```

## 🎮 Quick Start

### Basic Query Mode
```python
from waveformgpt import WaveformChat

chat = WaveformChat("simulation.vcd", use_llm=True)
chat.ask("What's the duty cycle of the clock?")
# → "The duty cycle is 50%. Calculated from 10ns high / 20ns period."
```

### 🎤 Voice Mode (NEW!)
```python
from waveformgpt import VoiceChat

# Talk to your waveforms!
voice = VoiceChat("simulation.vcd")
voice.start()  # Speak your questions, hear responses!
```

### ⚡ Live Streaming Mode (NEW!)
```python
from waveformgpt import LiveWaveformAnalyzer

# Monitor a running simulation
analyzer = LiveWaveformAnalyzer()
analyzer.connect("file:/path/to/simulation.vcd")  # Watch VCD file
# Or: analyzer.connect("ws://localhost:8765")      # WebSocket
# Or: analyzer.connect("sigrok://fx2lafw")         # Logic analyzer

# Set up alerts
analyzer.add_alert("error_detected", "error == '1'")
analyzer.add_alert("timeout", "req == '1' and ack == '0'")

analyzer.start()

# Query live data
analyzer.ask("What's happening right now?")
```

## 🎤 Voice Interface

Talk to your waveforms hands-free!

```bash
# Start voice session
python demo/voice_demo.py

# Or in code:
from waveformgpt import start_voice_session
start_voice_session("simulation.vcd")
```

**Voice Commands:**
- "What's the clock frequency?"
- "Describe the handshake protocol"
- "Are there any errors?"
- "Quit" (to exit)

## ⚡ Live Waveform Sources

| Source | URI Format | Use Case |
|--------|------------|----------|
| VCD File | `file:/path/to.vcd` | Watch simulation output |
| Named Pipe | `fifo:/tmp/pipe` | Custom integrations |
| WebSocket | `ws://host:port` | Browser/remote tools |
| Sigrok | `sigrok://driver` | Logic analyzers |

### Example: Live Simulation Monitoring

```python
from waveformgpt import LiveWaveformAnalyzer

analyzer = LiveWaveformAnalyzer()
analyzer.connect("file:/tmp/sim.vcd")

# Alert when something goes wrong
analyzer.add_alert("bus_error", "error == '1'")
analyzer.add_alert("stall", "valid == '1' and ready == '0'")

# Callback on any update
def on_update(signal, time, value):
    if signal == "state" and value == "ERROR":
        print(f"State machine error at t={time}!")

analyzer.on_update(on_update)
analyzer.start()
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

### 🔧 CI/CD Integration (NEW!)
```python
from waveformgpt import WaveformCI

# Run automated waveform checks in CI/CD pipelines
ci = WaveformCI("simulation.vcd", use_llm=True)

# Add checks
ci.add_check("clock_present", lambda chat: chat.ask("Is there a clock?"))
ci.add_check("no_errors", lambda chat: chat.ask("Any errors?"), 
             lambda r: "no" in r.lower())
ci.add_protocol_check("axi", signal_prefix="m_axi_")

# Run and get exit code
results = ci.run_all()
ci.generate_junit_xml(results, "results.xml")  # For Jenkins/GitHub Actions
exit_code = ci.print_report(results)
```

Use in GitHub Actions:
```yaml
- name: Analyze Waveforms
  run: |
    python -c "
    from waveformgpt.integrations import ci_check_waveform
    exit(ci_check_waveform('sim.vcd', [
        {'name': 'clock', 'question': 'Is clock present?'},
        {'name': 'errors', 'question': 'Any errors?'}
    ]))"
```

### 🔄 Regression Testing (NEW!)
```python
from waveformgpt import WaveformRegression, CoverageAnalyzer

# Compare waveforms for regressions
reg = WaveformRegression(tolerance_ns=1)
result = reg.compare("golden.vcd", "new.vcd")
print(result.summary)  # Shows differences

# Coverage analysis
cov = CoverageAnalyzer("simulation.vcd")
cov.add_point("fifo_full", "full", "value == 1")
cov.add_toggle_coverage("clk")
report = cov.analyze()
print(f"Coverage: {report.coverage_percent:.1f}%")
```

### 🔍 Debugging Tools (NEW!)
```python
from waveformgpt import WaveformDebugger, StateMachineAnalyzer

dbg = WaveformDebugger("simulation.vcd", use_llm=True)

# Trace signal to find root cause
dbg.trace_signal("error", target_value=1)

# Analyze window around failure
window = dbg.analyze_window(1523, window=100)

# Find glitches
glitches = dbg.find_glitches("data", min_width=2)

# Find setup/hold violations
violations = dbg.find_metastability("data", "clk", setup_time=1)

# AI-powered root cause analysis
dbg.explain_failure(error_signal="error", error_time=1523)

# State machine analysis
fsm = StateMachineAnalyzer("simulation.vcd", "state")
fsm.learn_states()
issues = fsm.find_issues()
fsm.print_summary()  # Shows time spent in each state
```

### 📣 Team Notifications (NEW!)
```python
from waveformgpt import SlackNotifier, GitHubIntegration

# Slack alerts
slack = SlackNotifier("https://hooks.slack.com/...")
slack.send_violation(
    title="Protocol Violation",
    details="AXI handshake timeout",
    vcd_file="sim.vcd", time=1523
)
slack.send_ci_results(results, "simulation.vcd")

# GitHub issue creation
gh = GitHubIntegration(token="ghp_...", repo="user/repo")
gh.create_violation_issue(
    violation_type="Setup Timing Violation",
    details="Data changed 0.5ns before clock edge",
    vcd_file="simulation.vcd"
)
```

### 📊 Report Generation (NEW!)
```python
from waveformgpt import ReportGenerator

report = ReportGenerator("simulation.vcd", title="Verification Report")
report.add_section("Overview", chat.ask("Summarize the simulation"))
report.add_section("Protocol", chat.ask("Any protocol violations?"))
report.add_ci_results(ci_results)
report.add_waveform(["clk", "data", "valid"], title="Key Signals")

report.generate_html("report.html")   # Beautiful HTML report
report.generate_markdown("report.md")  # Markdown for GitHub
```

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
│   ├── voice.py         # Voice interface (STT/TTS)
│   ├── live.py          # Live waveform streaming
│   ├── integrations.py  # CI/CD, Slack, GitHub, reports
│   ├── regression.py    # Regression & coverage testing
│   ├── debug.py         # Debugging & FSM analysis
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
│   ├── demo.vcd           # Sample waveform
│   ├── run_demo.py        # Basic demo
│   ├── llm_demo.py        # LLM-powered demo
│   ├── voice_demo.py      # Voice interface demo
│   ├── live_demo.py       # Live streaming demo
│   └── integration_demo.py # CI/CD & debug demo
├── .github/workflows/
│   └── waveform-ci.yml    # GitHub Actions example
└── tests/                 # Test suite
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
