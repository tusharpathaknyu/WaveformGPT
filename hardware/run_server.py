#!/usr/bin/env python3
"""
WaveformBuddy Server

Run this on your Mac to receive data from the ESP32 hardware
and communicate with the OpenAI API.

Usage:
    python run_server.py              # Start server on port 8080
    python run_server.py --port 9000  # Use different port
    python run_server.py --webcam     # Also enable webcam capture
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from waveformgpt.buddy import WaveformBuddy, ESP32Bridge


def get_local_ip():
    """Get the local IP address for ESP32 to connect to"""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def main():
    parser = argparse.ArgumentParser(description='WaveformBuddy Server')
    parser.add_argument('--port', type=int, default=8080, 
                        help='Port to listen on (default: 8080)')
    parser.add_argument('--webcam', action='store_true',
                        help='Enable webcam capture mode')
    parser.add_argument('--voice', action='store_true',
                        help='Enable local voice input mode')
    args = parser.parse_args()
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  ERROR: OPENAI_API_KEY environment variable not set!")
        print()
        print("   Set it with:")
        print("   export OPENAI_API_KEY='your-api-key'")
        print()
        print("   Or add to your ~/.zshrc:")
        print("   echo 'export OPENAI_API_KEY=\"your-key\"' >> ~/.zshrc")
        print()
        sys.exit(1)
    
    # Get local IP
    local_ip = get_local_ip()
    
    print()
    print("=" * 60)
    print("  🔧 WaveformBuddy Server")
    print("=" * 60)
    print()
    print(f"  Your Mac's IP: {local_ip}")
    print(f"  Server Port:   {args.port}")
    print()
    print("  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  ESP32 Configuration:")
    print("  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    print(f"  Update your ESP32 firmware with:")
    print(f"    const char* SERVER_URL = \"http://{local_ip}:{args.port}\";")
    print()
    print("=" * 60)
    print()
    
    # Create buddy instance
    def status_callback(msg):
        print(f"[Status] {msg}")
    
    buddy = WaveformBuddy(on_status=status_callback)
    
    # Start in different modes
    if args.voice:
        print("Starting in voice mode...")
        print("Use your Mac's microphone to talk.")
        print()
        buddy.start()
    else:
        # Start ESP32 bridge
        bridge = ESP32Bridge(buddy, port=args.port)
        
        if args.webcam:
            print("Webcam mode enabled - use 'c' for circuit, 'w' for waveform")
            import threading
            
            def webcam_input():
                while True:
                    cmd = input()
                    if cmd == 'c':
                        buddy.capture_circuit()
                    elif cmd == 'w':
                        buddy.capture_waveform()
                    elif cmd == 'q':
                        print("Shutting down...")
                        os._exit(0)
            
            threading.Thread(target=webcam_input, daemon=True).start()
        
        try:
            bridge.start()
        except KeyboardInterrupt:
            print("\nShutting down...")


if __name__ == "__main__":
    main()
