#!/usr/bin/env python3
"""
WaveformBuddy Demo - Test with sample images

This demo shows how WaveformBuddy correlates circuit and waveform analysis.
It works with image files instead of a live camera.

Usage:
    python buddy_demo.py                    # Interactive mode
    python buddy_demo.py circuit.jpg        # Load circuit first
    python buddy_demo.py circuit.jpg wave.jpg  # Load both
"""

import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from waveformgpt.buddy import WaveformBuddy, Capture, CaptureType


def create_test_circuit_image():
    """Create a simple test circuit diagram"""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None
    
    # Create a simple schematic-like image
    img = Image.new('RGB', (800, 400), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw some components
    # Resistor (zigzag)
    draw.line([(100, 200), (150, 200)], fill='black', width=2)
    for i in range(5):
        x = 150 + i * 20
        draw.line([(x, 200), (x+10, 180), (x+20, 200)], fill='black', width=2)
    draw.line([(250, 200), (300, 200)], fill='black', width=2)
    
    # Capacitor
    draw.line([(300, 200), (350, 200)], fill='black', width=2)
    draw.line([(350, 170), (350, 230)], fill='black', width=3)
    draw.line([(360, 170), (360, 230)], fill='black', width=3)
    draw.line([(360, 200), (400, 200)], fill='black', width=2)
    
    # Inductor (coils)
    draw.line([(400, 200), (420, 200)], fill='black', width=2)
    for i in range(4):
        x = 420 + i * 25
        draw.arc([(x, 185), (x+25, 215)], 0, 180, fill='black', width=2)
    draw.line([(520, 200), (550, 200)], fill='black', width=2)
    
    # Labels
    draw.text((170, 150), "R1: 1kΩ", fill='black')
    draw.text((340, 140), "C1: 100nF", fill='black')
    draw.text((440, 150), "L1: 10µH", fill='black')
    
    # Title
    draw.text((250, 30), "LC Filter Circuit", fill='black')
    draw.text((100, 350), "VIN", fill='blue')
    draw.text((550, 350), "VOUT", fill='green')
    
    # Ground symbol
    draw.line([(325, 250), (325, 280)], fill='black', width=2)
    draw.line([(310, 280), (340, 280)], fill='black', width=2)
    draw.line([(315, 290), (335, 290)], fill='black', width=2)
    draw.line([(320, 300), (330, 300)], fill='black', width=2)
    
    import io
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()


def create_test_waveform_image():
    """Create a test oscilloscope-like waveform image"""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    
    import math
    
    img = Image.new('RGB', (800, 400), 'black')
    draw = ImageDraw.Draw(img)
    
    # Grid
    for x in range(0, 800, 50):
        draw.line([(x, 0), (x, 400)], fill='#333333', width=1)
    for y in range(0, 400, 50):
        draw.line([(0, y), (800, y)], fill='#333333', width=1)
    
    # Center lines
    draw.line([(0, 200), (800, 200)], fill='#555555', width=1)
    draw.line([(400, 0), (400, 400)], fill='#555555', width=1)
    
    # Channel 1 - Input signal (yellow)
    points_ch1 = []
    for x in range(800):
        t = x / 50
        y = 100 + 80 * math.sin(t * 2)
        points_ch1.append((x, y))
    draw.line(points_ch1, fill='yellow', width=2)
    
    # Channel 2 - Output with ringing (cyan)
    points_ch2 = []
    for x in range(800):
        t = x / 50
        # Damped oscillation (ringing)
        ringing = 30 * math.exp(-t * 0.3) * math.sin(t * 8)
        y = 300 + 60 * math.sin(t * 2) + ringing
        points_ch2.append((x, y))
    draw.line(points_ch2, fill='cyan', width=2)
    
    # Labels
    draw.text((10, 10), "CH1: VIN  1V/div", fill='yellow')
    draw.text((10, 30), "CH2: VOUT 1V/div", fill='cyan')
    draw.text((650, 10), "Time: 10µs/div", fill='white')
    draw.text((10, 380), "Trigger: CH1 Rising", fill='white')
    
    import io
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()


def main():
    print("=" * 60)
    print("  WaveformBuddy Demo")
    print("  AI-Powered Hardware Debugging Companion")
    print("=" * 60)
    print()
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set!")
        print("   Set it with: export OPENAI_API_KEY='your-key'")
        print()
    
    # Initialize buddy
    buddy = WaveformBuddy(
        on_status=lambda s: print(f"[Status] {s}")
    )
    
    # Handle command line arguments
    if len(sys.argv) >= 2:
        circuit_path = sys.argv[1]
        print(f"Loading circuit from: {circuit_path}")
        with open(circuit_path, 'rb') as f:
            buddy.capture_circuit(f.read())
    else:
        # Create test circuit image
        print("Creating test circuit image...")
        circuit_data = create_test_circuit_image()
        if circuit_data:
            buddy.capture_circuit(circuit_data)
    
    if len(sys.argv) >= 3:
        waveform_path = sys.argv[2]
        print(f"Loading waveform from: {waveform_path}")
        with open(waveform_path, 'rb') as f:
            buddy.capture_waveform(f.read())
    else:
        # Create test waveform image
        print("Creating test waveform image...")
        waveform_data = create_test_waveform_image()
        if waveform_data:
            buddy.capture_waveform(waveform_data)
    
    # Interactive Q&A
    print()
    print("=" * 60)
    print("Ask me anything about the circuit and waveform!")
    print("Examples:")
    print("  - What's causing the ringing?")
    print("  - How can I reduce the overshoot?")
    print("  - What component values would you suggest?")
    print("  - Is this a stable output?")
    print()
    print("Type 'quit' to exit")
    print("=" * 60)
    print()
    
    while True:
        try:
            question = input("You: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if question.lower() == 'new circuit':
                circuit_data = create_test_circuit_image()
                if circuit_data:
                    buddy.capture_circuit(circuit_data)
                continue
            
            if question.lower() == 'new waveform':
                waveform_data = create_test_waveform_image()
                if waveform_data:
                    buddy.capture_waveform(waveform_data)
                continue
            
            # Ask the question
            response = buddy.ask(question)
            print(f"\nBuddy: {response}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
