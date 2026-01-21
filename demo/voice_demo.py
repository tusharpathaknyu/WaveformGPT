#!/usr/bin/env python3
"""
WaveformGPT Voice Demo - Talk to Your Waveforms!

This demo shows how to use voice commands to analyze waveforms.

Requirements:
    pip install pyaudio openai

Usage:
    python demo/voice_demo.py

Then just speak your questions!
"""

import sys
import os

# Add source to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def check_requirements():
    """Check if required packages are installed."""
    missing = []
    
    try:
        import pyaudio
    except ImportError:
        missing.append("pyaudio")
    
    try:
        from openai import OpenAI
    except ImportError:
        missing.append("openai")
    
    if missing:
        print("❌ Missing required packages:")
        print(f"   pip install {' '.join(missing)}")
        print()
        print("On macOS, you may need:")
        print("   brew install portaudio")
        print("   pip install pyaudio")
        return False
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set")
        print("   export OPENAI_API_KEY='your-key-here'")
        return False
    
    return True


def main():
    print("=" * 60)
    print("🎙️  WaveformGPT Voice Demo")
    print("=" * 60)
    print()
    
    if not check_requirements():
        print()
        print("Running in text mode instead...")
        print()
        
        from waveformgpt import WaveformChat
        
        demo_vcd = os.path.join(os.path.dirname(__file__), "demo.vcd")
        chat = WaveformChat(demo_vcd, use_llm=True)
        
        print("Type your questions (or 'quit' to exit):")
        while True:
            try:
                q = input("\n📝 You: ").strip()
                if q.lower() in ('quit', 'exit', 'q'):
                    break
                if q:
                    r = chat.ask(q)
                    print(f"🤖 {r.content}")
            except (KeyboardInterrupt, EOFError):
                break
        return
    
    # Voice mode
    from waveformgpt.voice import VoiceChat, VoiceConfig
    
    demo_vcd = os.path.join(os.path.dirname(__file__), "demo.vcd")
    
    config = VoiceConfig(
        stt_provider="openai",
        tts_provider="openai",
        tts_voice="nova",  # Friendly voice
    )
    
    voice = VoiceChat(demo_vcd, config)
    
    print("Starting voice interface...")
    print("Speak naturally - I'll listen and respond!")
    print()
    
    voice.start()


if __name__ == "__main__":
    main()
