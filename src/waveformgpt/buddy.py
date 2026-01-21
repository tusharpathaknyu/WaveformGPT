"""
WaveformBuddy - AI-Powered Hardware Debugging Companion

A voice-controlled gadget that:
1. Captures circuit photos (PCB, schematic, breadboard)
2. Captures oscilloscope waveforms
3. Correlates circuit ↔ waveform in conversation
4. Provides expert debugging assistance via voice

Hardware: ESP32-CAM + INMP441 mic + PAM8302 amp + speaker
Software: This Python server bridges hardware ↔ cloud LLM

Author: WaveformGPT Project
"""

import os
import io
import base64
import time
import json
import wave
import threading
import tempfile
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Callable
from enum import Enum

# Optional imports - graceful fallback
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class CaptureType(Enum):
    """Type of capture"""
    CIRCUIT = "circuit"
    WAVEFORM = "waveform"


@dataclass
class Capture:
    """A captured image with metadata"""
    type: CaptureType
    image_data: bytes
    timestamp: datetime = field(default_factory=datetime.now)
    description: str = ""
    
    def to_base64(self) -> str:
        return base64.b64encode(self.image_data).decode('utf-8')
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            f.write(self.image_data)


@dataclass 
class BuddyContext:
    """Maintains conversation context with captured images"""
    circuit_captures: List[Capture] = field(default_factory=list)
    waveform_captures: List[Capture] = field(default_factory=list)
    conversation_history: List[dict] = field(default_factory=list)
    
    def add_capture(self, capture: Capture):
        if capture.type == CaptureType.CIRCUIT:
            self.circuit_captures.append(capture)
        else:
            self.waveform_captures.append(capture)
    
    def get_latest_circuit(self) -> Optional[Capture]:
        return self.circuit_captures[-1] if self.circuit_captures else None
    
    def get_latest_waveform(self) -> Optional[Capture]:
        return self.waveform_captures[-1] if self.waveform_captures else None
    
    def clear(self):
        self.circuit_captures.clear()
        self.waveform_captures.clear()
        self.conversation_history.clear()


class WaveformBuddy:
    """
    AI-Powered Hardware Debugging Companion
    
    Voice commands:
    - "Capture circuit" - Take photo of circuit/schematic
    - "Capture waveform" - Take photo of oscilloscope
    - "What's wrong?" - Analyze waveform in context of circuit
    - "Watch live" - Continuous monitoring mode
    - "Clear context" - Reset conversation
    
    Usage:
        buddy = WaveformBuddy()
        buddy.start()  # Starts listening
        
        # Or programmatic use:
        buddy.capture_circuit(image_bytes)
        buddy.capture_waveform(image_bytes)
        response = buddy.ask("Why is there ringing on this signal?")
    """
    
    SYSTEM_PROMPT = """You are WaveformBuddy, an expert hardware debugging assistant. 
You have the ability to see:
1. Circuit images (schematics, PCBs, breadboards)
2. Oscilloscope waveforms

When analyzing:
- Correlate what you see in the waveform to the circuit topology
- Identify potential issues (ringing, noise, timing violations, etc.)
- Explain root causes referencing specific components
- Suggest fixes with component values when possible

Be concise but thorough. Speak like a senior EE helping a colleague debug.
If you see a circuit, remember its topology for waveform analysis.
If you see a waveform, relate anomalies back to the circuit."""

    VOICE_COMMANDS = {
        "capture circuit": "circuit",
        "take circuit": "circuit",
        "capture schematic": "circuit",
        "capture board": "circuit",
        "capture waveform": "waveform",
        "take waveform": "waveform",
        "capture scope": "waveform",
        "capture oscilloscope": "waveform",
        "watch live": "live",
        "start watching": "live",
        "monitor": "live",
        "stop": "stop",
        "stop watching": "stop",
        "clear": "clear",
        "clear context": "clear",
        "reset": "clear",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        tts_model: str = "tts-1",
        voice: str = "onyx",
        camera_index: int = 0,
        sample_rate: int = 16000,
        on_status: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize WaveformBuddy.
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Vision model to use
            tts_model: Text-to-speech model
            voice: TTS voice (alloy, echo, fable, onyx, nova, shimmer)
            camera_index: Camera device index (0 = default webcam)
            sample_rate: Audio sample rate for voice input
            on_status: Callback for status updates (for display/LED)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.tts_model = tts_model
        self.voice = voice
        self.camera_index = camera_index
        self.sample_rate = sample_rate
        self.on_status = on_status or (lambda s: print(f"[Status] {s}"))
        
        # Context
        self.context = BuddyContext()
        
        # State
        self.is_listening = False
        self.is_watching = False
        self._watch_thread = None
        self._listen_thread = None
        
        # OpenAI client
        if HAS_OPENAI and self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None
            
        # Camera
        self.camera = None
        
    def _status(self, msg: str):
        """Update status"""
        if self.on_status:
            self.on_status(msg)
    
    # =========================================================================
    # CAPTURE METHODS
    # =========================================================================
    
    def capture_from_camera(self) -> Optional[bytes]:
        """Capture image from webcam/camera"""
        if not HAS_CV2:
            self._status("OpenCV not installed")
            return None
            
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self._status("Cannot open camera")
            return None
            
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            self._status("Failed to capture")
            return None
            
        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()
    
    def capture_circuit(self, image_data: Optional[bytes] = None) -> Capture:
        """
        Capture circuit image.
        
        Args:
            image_data: Image bytes, or None to capture from camera
            
        Returns:
            Capture object
        """
        self._status("Capturing circuit...")
        
        if image_data is None:
            image_data = self.capture_from_camera()
            if image_data is None:
                raise RuntimeError("Failed to capture from camera")
        
        capture = Capture(
            type=CaptureType.CIRCUIT,
            image_data=image_data,
            description="Circuit capture"
        )
        self.context.add_capture(capture)
        
        # Get quick description from LLM
        if self.client:
            response = self._call_vision(
                "Briefly describe this circuit in one sentence. What components do you see?",
                [capture]
            )
            capture.description = response
            self._status(f"Circuit: {response[:50]}...")
            self._speak(f"Got it. I see {response}")
        else:
            self._status("Circuit captured")
            
        return capture
    
    def capture_waveform(self, image_data: Optional[bytes] = None) -> Capture:
        """
        Capture waveform from oscilloscope.
        
        Args:
            image_data: Image bytes, or None to capture from camera
            
        Returns:
            Capture object with analysis
        """
        self._status("Capturing waveform...")
        
        if image_data is None:
            image_data = self.capture_from_camera()
            if image_data is None:
                raise RuntimeError("Failed to capture from camera")
        
        capture = Capture(
            type=CaptureType.WAVEFORM,
            image_data=image_data,
            description="Waveform capture"
        )
        self.context.add_capture(capture)
        
        # Analyze in context of circuit
        if self.client:
            circuit = self.context.get_latest_circuit()
            captures = [capture]
            if circuit:
                captures.insert(0, circuit)
                prompt = "Analyze this waveform in context of the circuit I showed you earlier. What do you observe?"
            else:
                prompt = "Analyze this oscilloscope waveform. What do you observe?"
            
            response = self._call_vision(prompt, captures)
            capture.description = response
            self._status("Waveform analyzed")
            self._speak(response)
        else:
            self._status("Waveform captured")
            
        return capture
    
    # =========================================================================
    # CONVERSATION METHODS
    # =========================================================================
    
    def ask(self, question: str) -> str:
        """
        Ask a question about the captured circuit/waveform.
        
        Args:
            question: Natural language question
            
        Returns:
            AI response
        """
        if not self.client:
            return "OpenAI client not available"
        
        # Build image list - most recent circuit + waveform
        captures = []
        circuit = self.context.get_latest_circuit()
        waveform = self.context.get_latest_waveform()
        
        if circuit:
            captures.append(circuit)
        if waveform:
            captures.append(waveform)
        
        response = self._call_vision(question, captures)
        
        # Add to conversation history
        self.context.conversation_history.append({
            "role": "user",
            "content": question
        })
        self.context.conversation_history.append({
            "role": "assistant", 
            "content": response
        })
        
        return response
    
    def _call_vision(self, prompt: str, captures: List[Capture]) -> str:
        """Make a vision API call with images"""
        if not self.client:
            return "API client not available"
        
        # Build message content
        content = []
        
        # Add images
        for capture in captures:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{capture.to_base64()}",
                    "detail": "high"
                }
            })
        
        # Add text prompt
        content.append({
            "type": "text",
            "text": prompt
        })
        
        # Build messages with history
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        
        # Add conversation history (last 10 exchanges)
        for msg in self.context.conversation_history[-20:]:
            messages.append(msg)
        
        messages.append({"role": "user", "content": content})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"
    
    # =========================================================================
    # VOICE METHODS
    # =========================================================================
    
    def _speak(self, text: str):
        """Convert text to speech and play"""
        if not self.client or not HAS_PYAUDIO:
            print(f"[Buddy] {text}")
            return
        
        try:
            response = self.client.audio.speech.create(
                model=self.tts_model,
                voice=self.voice,
                input=text
            )
            
            # Save to temp file and play
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(response.content)
                temp_path = f.name
            
            # Play with system command (cross-platform)
            import subprocess
            import sys
            
            if sys.platform == "darwin":  # macOS
                subprocess.run(["afplay", temp_path], check=True)
            elif sys.platform == "win32":
                subprocess.run(["start", temp_path], shell=True)
            else:  # Linux
                subprocess.run(["mpg123", "-q", temp_path], check=True)
                
            os.unlink(temp_path)
            
        except Exception as e:
            print(f"[Buddy] {text}")
            print(f"[TTS Error] {e}")
    
    def _listen(self) -> Optional[str]:
        """Listen for voice input and transcribe"""
        if not self.client or not HAS_PYAUDIO:
            return None
        
        self._status("Listening...")
        
        try:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=1024
            )
            
            frames = []
            silence_threshold = 500
            silence_count = 0
            max_silence = 30  # ~1.5 seconds of silence to stop
            
            # Record until silence
            for _ in range(int(self.sample_rate / 1024 * 10)):  # Max 10 seconds
                data = stream.read(1024, exception_on_overflow=False)
                frames.append(data)
                
                # Simple silence detection
                import struct
                samples = struct.unpack(f'{len(data)//2}h', data)
                amplitude = max(abs(s) for s in samples)
                
                if amplitude < silence_threshold:
                    silence_count += 1
                    if silence_count > max_silence:
                        break
                else:
                    silence_count = 0
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Save to WAV
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wf = wave.open(f.name, 'wb')
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(frames))
                wf.close()
                temp_path = f.name
            
            # Transcribe with Whisper
            with open(temp_path, 'rb') as f:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f
                )
            
            os.unlink(temp_path)
            self._status("Processing...")
            
            return transcript.text.lower().strip()
            
        except Exception as e:
            self._status(f"Listen error: {e}")
            return None
    
    def _process_voice_command(self, text: str) -> bool:
        """
        Process voice input for commands.
        
        Returns:
            True if it was a command, False if it's a question
        """
        # Check for commands
        for phrase, action in self.VOICE_COMMANDS.items():
            if phrase in text:
                if action == "circuit":
                    self.capture_circuit()
                    return True
                elif action == "waveform":
                    self.capture_waveform()
                    return True
                elif action == "live":
                    self.start_watching()
                    return True
                elif action == "stop":
                    self.stop_watching()
                    return True
                elif action == "clear":
                    self.context.clear()
                    self._speak("Context cleared. Ready for new captures.")
                    return True
        
        return False
    
    # =========================================================================
    # LIVE WATCHING MODE
    # =========================================================================
    
    def start_watching(self, fps: float = 1.0, duration: float = 30.0):
        """
        Start live waveform monitoring.
        
        Args:
            fps: Frames per second to capture
            duration: How long to watch (seconds)
        """
        if self.is_watching:
            return
            
        self.is_watching = True
        self._speak("Starting live monitoring. I'll alert you to any issues.")
        
        def watch_loop():
            frames = []
            start_time = time.time()
            frame_interval = 1.0 / fps
            
            while self.is_watching and (time.time() - start_time) < duration:
                # Capture frame
                image_data = self.capture_from_camera()
                if image_data:
                    frames.append(Capture(
                        type=CaptureType.WAVEFORM,
                        image_data=image_data
                    ))
                
                # Every 5 frames, do batch analysis
                if len(frames) >= 5:
                    self._analyze_batch(frames)
                    frames = []
                
                time.sleep(frame_interval)
            
            # Final analysis
            if frames:
                self._analyze_batch(frames)
            
            self.is_watching = False
            self._status("Watching stopped")
        
        self._watch_thread = threading.Thread(target=watch_loop, daemon=True)
        self._watch_thread.start()
    
    def stop_watching(self):
        """Stop live monitoring"""
        self.is_watching = False
        self._speak("Stopping live monitoring.")
    
    def _analyze_batch(self, frames: List[Capture]):
        """Analyze a batch of frames for changes/anomalies"""
        if not self.client or not frames:
            return
        
        # Send first and last frame for comparison
        captures = [frames[0], frames[-1]]
        circuit = self.context.get_latest_circuit()
        if circuit:
            captures.insert(0, circuit)
        
        prompt = """These are two waveform captures from continuous monitoring.
Compare them and report ONLY if you see:
- Significant changes between frames
- Anomalies (glitches, noise spikes, timing issues)
- Anything concerning

If everything looks stable and normal, just say "Stable."
Be brief - this is real-time monitoring."""

        response = self._call_vision(prompt, captures)
        
        # Only speak if there's something interesting
        if "stable" not in response.lower():
            self._speak(response)
        else:
            self._status("Stable...")
    
    # =========================================================================
    # MAIN LOOP
    # =========================================================================
    
    def start(self):
        """Start the voice-controlled assistant loop"""
        if not self.client:
            print("Error: OpenAI client not available. Set OPENAI_API_KEY.")
            return
        
        self.is_listening = True
        self._speak("WaveformBuddy ready. Say 'capture circuit' or 'capture waveform' to begin.")
        
        while self.is_listening:
            # Listen for voice input
            text = self._listen()
            if not text:
                continue
            
            print(f"[Heard] {text}")
            
            # Check for quit
            if "goodbye" in text or "quit" in text or "exit" in text:
                self._speak("Goodbye!")
                break
            
            # Process command or question
            if not self._process_voice_command(text):
                # It's a question - answer it
                response = self.ask(text)
                self._speak(response)
    
    def stop(self):
        """Stop the assistant"""
        self.is_listening = False
        self.is_watching = False


class ESP32Bridge:
    """
    Bridge to connect ESP32 hardware to WaveformBuddy.
    
    The ESP32 sends:
    - Images via HTTP POST
    - Audio via WebSocket
    
    This server receives them and forwards to WaveformBuddy.
    """
    
    def __init__(self, buddy: WaveformBuddy, port: int = 8080):
        self.buddy = buddy
        self.port = port
        
    def start(self):
        """Start the HTTP server for ESP32 communication"""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import json
        
        buddy = self.buddy
        
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                
                if self.path == '/circuit':
                    buddy.capture_circuit(post_data)
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'OK')
                    
                elif self.path == '/waveform':
                    buddy.capture_waveform(post_data)
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'OK')
                    
                elif self.path == '/ask':
                    question = post_data.decode('utf-8')
                    response = buddy.ask(question)
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(response.encode('utf-8'))
                    
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def log_message(self, format, *args):
                buddy._status(f"ESP32: {args[0]}")
        
        server = HTTPServer(('0.0.0.0', self.port), Handler)
        print(f"ESP32 Bridge listening on port {self.port}")
        server.serve_forever()


# =============================================================================
# DEMO / CLI
# =============================================================================

def demo_with_files():
    """Demo using image files instead of camera"""
    import sys
    
    buddy = WaveformBuddy()
    
    print("=" * 60)
    print("WaveformBuddy Demo")
    print("=" * 60)
    
    # Check for command line args
    if len(sys.argv) >= 2:
        circuit_path = sys.argv[1]
        with open(circuit_path, 'rb') as f:
            buddy.capture_circuit(f.read())
    
    if len(sys.argv) >= 3:
        waveform_path = sys.argv[2]
        with open(waveform_path, 'rb') as f:
            buddy.capture_waveform(f.read())
    
    # Interactive mode
    print("\nEnter questions (or 'quit' to exit):")
    while True:
        try:
            question = input("\nYou: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            if not question:
                continue
                
            response = buddy.ask(question)
            print(f"\nBuddy: {response}")
            
        except KeyboardInterrupt:
            break
    
    print("\nGoodbye!")


if __name__ == "__main__":
    demo_with_files()
