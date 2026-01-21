#!/usr/bin/env python3
"""
WaveformGPT Live Waveform Demo - Real-time Analysis!

This demo shows how to monitor live waveforms with alerts.

Demo modes:
1. File watching - Simulates a VCD being written
2. WebSocket - Receive waveform data over network
3. FIFO - Read from named pipe

Usage:
    python demo/live_demo.py [mode]
    
    Modes: file, websocket, simulate
"""

import sys
import os
import time
import threading
import tempfile

# Add source to path for development  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def simulate_vcd_writer(filepath: str, stop_event: threading.Event):
    """Simulate a running simulation writing VCD data."""
    
    # Write initial VCD header
    with open(filepath, 'w') as f:
        f.write("""$version WaveformGPT Live Demo $end
$timescale 1ns $end
$scope module cpu $end
$var wire 1 ! clk $end
$var wire 1 " req $end
$var wire 1 # ack $end
$var wire 1 $ error $end
$var wire 8 % data [7:0] $end
$upscope $end
$enddefinitions $end
$dumpvars
0!
0"
0#
0$
b00000000 %
$end
""")
    
    print("📝 Simulating VCD writes...")
    
    # Simulate ongoing simulation
    t = 0
    clk = 0
    req = 0
    ack = 0
    error = 0
    data = 0
    
    while not stop_event.is_set():
        with open(filepath, 'a') as f:
            # Advance time
            t += 10
            f.write(f"#{t}\n")
            
            # Toggle clock
            clk = 1 - clk
            f.write(f"{clk}!\n")
            
            # Simulate req/ack handshake
            if t % 100 == 20:
                req = 1
                f.write('1"\n')
            elif t % 100 == 60:
                ack = 1
                f.write('1#\n')
            elif t % 100 == 80:
                req = 0
                f.write('0"\n')
            elif t % 100 == 0:
                ack = 0
                f.write('0#\n')
            
            # Occasionally trigger error
            if t % 500 == 250:
                error = 1
                f.write('1$\n')
                print(f"   💥 Error signal triggered at t={t}!")
            elif t % 500 == 260:
                error = 0
                f.write('0$\n')
            
            # Random data
            if t % 40 == 0:
                import random
                data = random.randint(0, 255)
                f.write(f"b{data:08b} %\n")
            
            f.flush()
        
        time.sleep(0.1)  # 100ms between updates
    
    print("📝 VCD writer stopped")


def demo_file_watching():
    """Demo: Watch a VCD file being written."""
    from waveformgpt.live import LiveWaveformAnalyzer
    
    print("\n" + "=" * 60)
    print("📁 Live File Watching Demo")
    print("=" * 60)
    print()
    
    # Create temp VCD file
    temp_vcd = tempfile.mktemp(suffix=".vcd")
    print(f"Creating temp VCD: {temp_vcd}")
    
    # Start VCD writer in background
    stop_writer = threading.Event()
    writer_thread = threading.Thread(
        target=simulate_vcd_writer,
        args=(temp_vcd, stop_writer)
    )
    writer_thread.start()
    
    # Give it a moment to write header
    time.sleep(0.5)
    
    # Create live analyzer
    analyzer = LiveWaveformAnalyzer()
    analyzer.connect(f"file:{temp_vcd}")
    
    # Add alerts
    analyzer.add_alert("error", "error == '1'")
    analyzer.add_alert("handshake", "req == '1' and ack == '1'")
    
    # Track updates
    update_count = [0]
    
    def on_update(signal, t, value):
        update_count[0] += 1
        if update_count[0] % 50 == 0:
            print(f"   📊 Received {update_count[0]} updates, latest: t={t}")
    
    analyzer.on_update(on_update)
    
    print("\nStarting live monitoring...")
    print("Alerts configured:")
    print("  - 'error': Triggers when error signal goes high")
    print("  - 'handshake': Triggers when req and ack are both high")
    print()
    print("Press Ctrl+C to stop\n")
    
    analyzer.start()
    
    try:
        while True:
            time.sleep(2)
            
            # Periodically show status
            status = analyzer.get_status()
            print(f"\n📈 Status: {len(status['signals'])} signals, "
                  f"time range: {status['time_range']}")
            
            # Ask LLM about what's happening
            if os.getenv("OPENAI_API_KEY"):
                try:
                    print("   🤔 Analyzing current activity...")
                    answer = analyzer.ask("What's happening right now?")
                    print(f"   🤖 {answer[:200]}..." if len(answer) > 200 else f"   🤖 {answer}")
                except Exception as e:
                    print(f"   (LLM not available: {e})")
            
    except KeyboardInterrupt:
        print("\n\nStopping...")
    
    finally:
        analyzer.stop()
        stop_writer.set()
        writer_thread.join()
        
        # Cleanup
        try:
            os.remove(temp_vcd)
        except:
            pass
    
    print("\n✓ Demo complete!")


def demo_websocket():
    """Demo: WebSocket server for receiving waveform data."""
    from waveformgpt.live import LiveWaveformAnalyzer
    
    print("\n" + "=" * 60)
    print("🌐 WebSocket Live Demo")
    print("=" * 60)
    print()
    print("This will start a WebSocket server on ws://localhost:8765")
    print()
    print("Send JSON messages like:")
    print('  {"signal": "clk", "time": 100, "value": "1"}')
    print()
    print("Example with wscat:")
    print('  wscat -c ws://localhost:8765')
    print('  > {"signal": "clk", "time": 100, "value": "1"}')
    print()
    
    try:
        import websockets
    except ImportError:
        print("❌ Install websockets: pip install websockets")
        return
    
    analyzer = LiveWaveformAnalyzer()
    analyzer.connect("ws://localhost:8765")
    
    def on_update(signal, t, value):
        print(f"  📥 Received: {signal}={value} at t={t}")
    
    analyzer.on_update(on_update)
    
    print("Starting WebSocket server...")
    print("Press Ctrl+C to stop\n")
    
    analyzer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    
    analyzer.stop()
    print("\n✓ Demo complete!")


def demo_simulate():
    """Quick simulation demo without external dependencies."""
    from waveformgpt.live import LiveWaveformBuffer
    
    print("\n" + "=" * 60)
    print("🎬 Simulated Live Waveform Demo")
    print("=" * 60)
    print()
    
    buffer = LiveWaveformBuffer()
    
    # Simulate some data
    print("Simulating waveform data...")
    for t in range(0, 200, 10):
        clk = (t // 10) % 2
        buffer.add("clk", t, str(clk))
        
        if t >= 40 and t < 80:
            buffer.add("req", t, "1")
        else:
            buffer.add("req", t, "0")
        
        if t >= 70 and t < 90:
            buffer.add("ack", t, "1")
        else:
            buffer.add("ack", t, "0")
    
    print(f"\n📊 Buffer status:")
    print(f"   Signals: {buffer.get_signals()}")
    print(f"   Time range: {buffer.get_time_range()}")
    
    print(f"\n📈 Sample data:")
    for sig in buffer.get_signals():
        data = buffer.get_signal(sig)[:10]
        print(f"   {sig}: {data}")
    
    # If LLM available, analyze
    if os.getenv("OPENAI_API_KEY"):
        print("\n🤖 LLM Analysis:")
        from waveformgpt.llm_engine import WaveformLLM, OpenAIBackend
        
        llm = WaveformLLM(OpenAIBackend())
        llm.set_waveform_context(
            signals=buffer.get_signals(),
            time_range=buffer.get_time_range(),
            time_unit="ns",
            sample_data={s: buffer.get_signal(s) for s in buffer.get_signals()}
        )
        
        response = llm.query("Describe the handshake between req and ack")
        print(f"   {response.answer}")
    
    print("\n✓ Demo complete!")


def main():
    print("=" * 60)
    print("⚡ WaveformGPT Live Waveform Demo")
    print("=" * 60)
    
    mode = sys.argv[1] if len(sys.argv) > 1 else "simulate"
    
    if mode == "file":
        demo_file_watching()
    elif mode == "websocket":
        demo_websocket()
    elif mode == "simulate":
        demo_simulate()
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python live_demo.py [file|websocket|simulate]")


if __name__ == "__main__":
    main()
