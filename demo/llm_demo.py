#!/usr/bin/env python3
"""
WaveformGPT LLM Demo - Natural Language Waveform Analysis

This demo shows how WaveformGPT can answer ANY question about your
waveforms using LLM (Large Language Model) integration.

Setup:
    # Option 1: OpenAI (recommended)
    export OPENAI_API_KEY="your-key-here"
    
    # Option 2: Anthropic Claude
    export ANTHROPIC_API_KEY="your-key-here"
    
    # Option 3: Local Ollama (free, private)
    ollama serve  # Start Ollama server
    ollama pull llama3.1  # Download model

Usage:
    python demo/llm_demo.py
"""

import sys
import os

# Add source to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from waveformgpt import WaveformChat


def check_api_keys():
    """Check which LLM backends are available."""
    available = []
    
    if os.getenv("OPENAI_API_KEY"):
        available.append("OpenAI (GPT-4o)")
    
    if os.getenv("ANTHROPIC_API_KEY"):
        available.append("Anthropic (Claude)")
    
    # Check for local Ollama
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        if resp.status_code == 200:
            available.append("Ollama (local)")
    except:
        pass
    
    return available


def main():
    print("=" * 60)
    print("🧠 WaveformGPT LLM-Powered Demo")
    print("=" * 60)
    print()
    
    # Check available backends
    backends = check_api_keys()
    
    if not backends:
        print("❌ No LLM backend available!")
        print()
        print("To use LLM-powered queries, set up one of:")
        print()
        print("  1. OpenAI (recommended):")
        print("     export OPENAI_API_KEY='sk-...'")
        print()
        print("  2. Anthropic Claude:")
        print("     export ANTHROPIC_API_KEY='sk-ant-...'")
        print()
        print("  3. Ollama (free, local, private):")
        print("     brew install ollama  # or download from ollama.com")
        print("     ollama serve")
        print("     ollama pull llama3.1")
        print()
        print("For now, running in pattern-based mode (limited queries)...")
        print()
        use_llm = False
    else:
        print(f"✓ Available LLM backends: {', '.join(backends)}")
        print()
        use_llm = True
    
    # Initialize chat
    demo_vcd = os.path.join(os.path.dirname(__file__), "demo.vcd")
    
    print("Loading waveform file...")
    chat = WaveformChat(demo_vcd, use_llm=use_llm)
    print()
    
    # Show signals
    print("Available signals:")
    for sig in chat.parser.header.signals.values():
        print(f"  - {sig.full_name}")
    print()
    
    # Example questions
    questions = [
        # Basic questions (work in both modes)
        "When does clk rise?",
        "How many times does req go high?",
        
        # Advanced questions (require LLM)
        "What's the clock frequency?",
        "Is there a handshake protocol between req and ack?",
        "What's the latency from req going high to ack responding?",
        "Are there any glitches on the data signal?",
        "Describe the state machine behavior",
    ]
    
    print("-" * 60)
    print("Example Queries:")
    print("-" * 60)
    
    for q in questions:
        print(f"\n📝 Question: {q}")
        print("-" * 40)
        
        response = chat.ask(q)
        print(response.content)
        print()
        
        # Only try basic questions if no LLM
        if not use_llm and questions.index(q) >= 1:
            print("\n💡 Enable LLM mode to ask more complex questions!")
            break
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("Interactive Mode - Ask anything about the waveform!")
    print("Type 'quit' to exit, 'help' for examples")
    print("=" * 60)
    
    while True:
        try:
            question = input("\n🎤 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        
        if not question:
            continue
        
        if question.lower() in ('quit', 'exit', 'q'):
            break
        
        response = chat.ask(question)
        print(f"\n🤖 WaveformGPT: {response.content}")
    
    print("\nGoodbye! 👋")


if __name__ == "__main__":
    main()
