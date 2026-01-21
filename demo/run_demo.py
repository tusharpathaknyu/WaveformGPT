#!/usr/bin/env python3
"""
WaveformGPT Interactive Demo
============================

This demo shows what WaveformGPT can do:
1. Parse VCD waveform files from digital simulations
2. Query waveforms using natural language
3. Visualize signals as ASCII waveforms
4. Analyze timing and protocols
5. Export to various formats
"""

import sys
sys.path.insert(0, 'src')

from waveformgpt import WaveformChat, VCDParser
from waveformgpt.visualize import ASCIIWaveform, WaveformStyle

def main():
    print("=" * 70)
    print("  WaveformGPT - Query Simulation Waveforms in Natural Language")
    print("=" * 70)
    print()
    
    # Load a VCD file
    vcd_path = "demo/demo.vcd"
    print(f"📂 Loading VCD file: {vcd_path}")
    print()
    
    parser = VCDParser(vcd_path)
    
    # Show what signals are in the waveform
    print("📊 Signals in this waveform:")
    print("-" * 40)
    for sig in parser.header.signals.values():
        width_str = f"[{sig.width-1}:0]" if sig.width > 1 else ""
        print(f"  • {sig.name}{width_str}")
    print()
    
    # Visualize some signals as ASCII waveforms
    print("🌊 ASCII Waveform Visualization:")
    print("-" * 70)
    
    style = WaveformStyle(width=65, high_char="▀", low_char="▁")
    ascii_wave = ASCIIWaveform(parser, style)
    
    for signal in ["clk", "reset", "req", "ack", "valid", "ready"]:
        try:
            waveform = ascii_wave.render_signal(signal, width=60)
            print(waveform)
        except Exception as e:
            pass
    print()
    
    # Now use natural language queries
    print("💬 Natural Language Queries:")
    print("-" * 70)
    
    chat = WaveformChat()
    chat.load_vcd(vcd_path)
    
    queries = [
        "When does req go high?",
        "How many times does clk rise?",
        "Find when ack goes high after req",
        "What signals are in this waveform?",
    ]
    
    for query in queries:
        print(f"\n❓ Query: \"{query}\"")
        response = chat.ask(query)
        print(f"📝 Answer: {response}")
    
    print()
    print("=" * 70)
    print("  This is WaveformGPT - Making waveform analysis accessible!")
    print("=" * 70)
    print()
    print("Use cases:")
    print("  • Debug RTL simulations by asking questions in plain English")
    print("  • Find timing issues: 'How long between req and ack?'")
    print("  • Protocol checking: Verify AXI, Wishbone, SPI transactions")
    print("  • Regression testing: Compare waveforms between runs")
    print("  • Documentation: Export to WaveDrom, markdown, CSV")
    print()

if __name__ == "__main__":
    main()
