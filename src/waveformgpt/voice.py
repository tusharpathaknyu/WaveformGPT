"""
Voice Interface for WaveformGPT.

Enables voice-based conversational waveform analysis:
- Speak your questions naturally
- Hear AI-generated responses
- Hands-free debugging workflow

Supports:
- OpenAI Whisper (speech-to-text)
- OpenAI TTS (text-to-speech)
- Local alternatives (vosk, pyttsx3)
"""

import os
import io
import tempfile
import threading
import queue
from typing import Optional, Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VoiceConfig:
    """Voice interface configuration."""
    stt_provider: str = "openai"  # "openai", "whisper_local", "vosk"
    tts_provider: str = "openai"  # "openai", "elevenlabs", "pyttsx3"
    tts_voice: str = "alloy"      # OpenAI: alloy, echo, fable, onyx, nova, shimmer
    language: str = "en"
    wake_word: Optional[str] = None  # Optional wake word like "hey waveform"
    auto_listen: bool = True      # Automatically listen after response


class SpeechToText:
    """Speech-to-text engine."""
    
    def __init__(self, provider: str = "openai"):
        self.provider = provider
        self._client = None
    
    def transcribe(self, audio_file: str) -> str:
        """Transcribe audio file to text."""
        if self.provider == "openai":
            return self._transcribe_openai(audio_file)
        elif self.provider == "whisper_local":
            return self._transcribe_whisper_local(audio_file)
        else:
            raise ValueError(f"Unknown STT provider: {self.provider}")
    
    def _transcribe_openai(self, audio_file: str) -> str:
        """Use OpenAI Whisper API."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        
        with open(audio_file, "rb") as f:
            transcript = self._client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="en"
            )
        return transcript.text
    
    def _transcribe_whisper_local(self, audio_file: str) -> str:
        """Use local Whisper model."""
        try:
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(audio_file)
            return result["text"]
        except ImportError:
            raise ImportError("Install whisper: pip install openai-whisper")


class TextToSpeech:
    """Text-to-speech engine."""
    
    def __init__(self, provider: str = "openai", voice: str = "alloy"):
        self.provider = provider
        self.voice = voice
        self._client = None
    
    def speak(self, text: str, output_file: Optional[str] = None) -> Optional[str]:
        """Convert text to speech and play or save."""
        if self.provider == "openai":
            return self._speak_openai(text, output_file)
        elif self.provider == "pyttsx3":
            return self._speak_pyttsx3(text, output_file)
        else:
            raise ValueError(f"Unknown TTS provider: {self.provider}")
    
    def _speak_openai(self, text: str, output_file: Optional[str] = None) -> str:
        """Use OpenAI TTS API."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        
        response = self._client.audio.speech.create(
            model="tts-1",
            voice=self.voice,
            input=text
        )
        
        if output_file is None:
            output_file = tempfile.mktemp(suffix=".mp3")
        
        response.stream_to_file(output_file)
        
        # Play the audio
        self._play_audio(output_file)
        
        return output_file
    
    def _speak_pyttsx3(self, text: str, output_file: Optional[str] = None) -> None:
        """Use local pyttsx3 for TTS."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            if output_file:
                engine.save_to_file(text, output_file)
                engine.runAndWait()
            else:
                engine.say(text)
                engine.runAndWait()
        except ImportError:
            raise ImportError("Install pyttsx3: pip install pyttsx3")
    
    def _play_audio(self, filepath: str):
        """Play audio file."""
        import platform
        import subprocess
        
        system = platform.system()
        if system == "Darwin":  # macOS
            subprocess.run(["afplay", filepath], check=False)
        elif system == "Linux":
            subprocess.run(["aplay", filepath], check=False)
        elif system == "Windows":
            subprocess.run(["start", filepath], shell=True, check=False)


class AudioRecorder:
    """Record audio from microphone."""
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self._recording = False
        self._frames = []
    
    def record(self, duration: float = 5.0) -> str:
        """Record audio for specified duration, return temp file path."""
        try:
            import pyaudio
            import wave
        except ImportError:
            raise ImportError("Install pyaudio: pip install pyaudio")
        
        p = pyaudio.PyAudio()
        
        stream = p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024
        )
        
        print("🎤 Listening...")
        frames = []
        
        for _ in range(int(self.sample_rate / 1024 * duration)):
            data = stream.read(1024)
            frames.append(data)
        
        print("✓ Got it!")
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Save to temp file
        temp_file = tempfile.mktemp(suffix=".wav")
        with wave.open(temp_file, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))
        
        return temp_file
    
    def record_until_silence(self, silence_threshold: float = 500, 
                             silence_duration: float = 1.5,
                             max_duration: float = 30.0) -> str:
        """Record until silence is detected."""
        try:
            import pyaudio
            import wave
            import struct
            import math
        except ImportError:
            raise ImportError("Install pyaudio: pip install pyaudio")
        
        p = pyaudio.PyAudio()
        
        stream = p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024
        )
        
        print("🎤 Listening... (speak now, I'll stop when you pause)")
        frames = []
        silent_chunks = 0
        chunks_per_second = self.sample_rate / 1024
        max_silent_chunks = int(silence_duration * chunks_per_second)
        max_chunks = int(max_duration * chunks_per_second)
        
        for i in range(max_chunks):
            data = stream.read(1024)
            frames.append(data)
            
            # Calculate RMS amplitude
            shorts = struct.unpack(f'{len(data)//2}h', data)
            rms = math.sqrt(sum(s**2 for s in shorts) / len(shorts))
            
            if rms < silence_threshold:
                silent_chunks += 1
                if silent_chunks > max_silent_chunks and len(frames) > chunks_per_second:
                    break
            else:
                silent_chunks = 0
        
        print("✓ Got it!")
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Save to temp file
        temp_file = tempfile.mktemp(suffix=".wav")
        with wave.open(temp_file, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))
        
        return temp_file


class VoiceChat:
    """
    Voice-enabled chat interface for WaveformGPT.
    
    Usage:
        from waveformgpt.voice import VoiceChat
        
        voice = VoiceChat("simulation.vcd")
        voice.start()  # Starts listening for voice commands
        
        # Or single query:
        response = voice.ask_voice()  # Records and processes
    """
    
    def __init__(self, vcd_file: str, config: Optional[VoiceConfig] = None):
        self.config = config or VoiceConfig()
        
        # Initialize components
        from waveformgpt import WaveformChat
        self.chat = WaveformChat(vcd_file, use_llm=True)
        
        self.stt = SpeechToText(self.config.stt_provider)
        self.tts = TextToSpeech(self.config.tts_provider, self.config.tts_voice)
        self.recorder = AudioRecorder()
        
        self._running = False
        self._thread = None
    
    def ask_voice(self) -> str:
        """Record a voice question and get a spoken response."""
        # Record audio
        audio_file = self.recorder.record_until_silence()
        
        try:
            # Transcribe
            print("🔄 Processing speech...")
            question = self.stt.transcribe(audio_file)
            print(f"📝 You said: {question}")
            
            # Get response
            response = self.chat.ask(question)
            answer = response.content
            
            # Clean up answer for speech (remove emojis, code blocks)
            answer_clean = self._clean_for_speech(answer)
            
            print(f"🤖 {answer_clean[:200]}..." if len(answer_clean) > 200 else f"🤖 {answer_clean}")
            
            # Speak response
            print("🔊 Speaking...")
            self.tts.speak(answer_clean)
            
            return answer
            
        finally:
            # Cleanup temp file
            try:
                os.remove(audio_file)
            except:
                pass
    
    def _clean_for_speech(self, text: str) -> str:
        """Clean text for TTS (remove emojis, markdown, etc.)."""
        import re
        
        # Remove emojis
        text = re.sub(r'[🤖📝💡✓✗▀▁]', '', text)
        
        # Remove markdown formatting
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # Italic
        text = re.sub(r'`([^`]+)`', r'\1', text)        # Code
        text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)  # Code blocks
        
        # Remove LaTeX
        text = re.sub(r'\\\[.*?\\\]', '', text, flags=re.DOTALL)
        text = re.sub(r'\$[^$]+\$', '', text)
        
        # Simplify for speech
        text = text.replace('\n\n', '. ')
        text = text.replace('\n', ' ')
        
        # Limit length for TTS
        if len(text) > 500:
            text = text[:500] + "... I can provide more details if you ask."
        
        return text.strip()
    
    def start(self):
        """Start continuous voice conversation loop."""
        print("=" * 60)
        print("🎙️  WaveformGPT Voice Mode")
        print("=" * 60)
        print(f"VCD: {self.chat.vcd_file}")
        print("Say 'quit' or 'exit' to stop")
        print("=" * 60)
        
        # Greeting
        greeting = f"Hello! I'm WaveformGPT. I've loaded your waveform with {len(list(self.chat.parser.header.signals.values()))} signals. Ask me anything about your simulation."
        print(f"🤖 {greeting}")
        self.tts.speak(greeting)
        
        self._running = True
        while self._running:
            try:
                answer = self.ask_voice()
                
                # Check for exit commands
                if any(word in answer.lower() for word in ['quit', 'exit', 'goodbye', 'bye']):
                    farewell = "Goodbye! Happy debugging!"
                    print(f"🤖 {farewell}")
                    self.tts.speak(farewell)
                    break
                    
            except KeyboardInterrupt:
                print("\n👋 Voice mode stopped.")
                break
            except Exception as e:
                print(f"Error: {e}")
                self.tts.speak("Sorry, I had trouble with that. Please try again.")
    
    def stop(self):
        """Stop voice conversation."""
        self._running = False


def start_voice_session(vcd_file: str, **kwargs):
    """Convenience function to start a voice session."""
    config = VoiceConfig(**kwargs)
    voice = VoiceChat(vcd_file, config)
    voice.start()
