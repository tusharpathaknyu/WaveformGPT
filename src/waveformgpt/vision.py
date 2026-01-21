"""
Vision-Based Waveform Analysis.

Let the LLM actually SEE your waveforms - like having an expert
look at your screen and tell you what's happening.

Uses GPT-4 Vision / Claude Vision to analyze waveform images directly.
"""

import os
import io
import base64
import tempfile
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VisionAnalysis:
    """Result from visual waveform analysis."""
    description: str
    observations: List[str]
    issues_found: List[str]
    suggestions: List[str]
    raw_response: str


class WaveformVision:
    """
    Let the LLM SEE your waveforms.
    
    Renders waveforms as images and sends them to vision-capable LLMs
    for analysis. Like having an expert look at your oscilloscope.
    
    Usage:
        vision = WaveformVision("simulation.vcd")
        
        # Let AI see and describe what's happening
        vision.look("What do you see?")
        
        # Ask specific questions about what it sees
        vision.look("Is the handshake protocol working correctly?")
        
        # Watch a time window
        vision.look_at(1500, 2000, "What went wrong here?")
        
        # Live monitoring - take periodic snapshots
        vision.watch(interval=1.0)  # Analyze every second
    """
    
    def __init__(self, vcd_file: str, 
                 provider: str = "openai",
                 signals: List[str] = None):
        """
        Args:
            vcd_file: Path to VCD file
            provider: Vision provider ("openai" or "anthropic")
            signals: Signals to display (None = auto-select important ones)
        """
        self.vcd_file = vcd_file
        self.provider = provider
        self.selected_signals = signals
        
        from waveformgpt import VCDParser
        self.parser = VCDParser(vcd_file)
        
        # Auto-select signals if not specified
        if self.selected_signals is None:
            all_sigs = [s.full_name for s in self.parser.header.signals.values()]
            # Prioritize common important signals
            priority = ['clk', 'clock', 'rst', 'reset', 'valid', 'ready', 
                       'error', 'data', 'addr', 'req', 'ack', 'done']
            selected = []
            for p in priority:
                for s in all_sigs:
                    if p in s.lower() and s not in selected:
                        selected.append(s)
                        if len(selected) >= 8:
                            break
            # Fill remaining with first signals
            for s in all_sigs:
                if s not in selected:
                    selected.append(s)
                if len(selected) >= 12:
                    break
            self.selected_signals = selected
        
        self._client = None
    
    def render_image(self, time_range: Tuple[int, int] = None,
                     signals: List[str] = None,
                     width: int = 1200,
                     height: int = 600) -> bytes:
        """
        Render waveform as PNG image.
        
        Returns PNG bytes that can be sent to vision LLM.
        """
        signals = signals or self.selected_signals
        
        # Try matplotlib first, fall back to PIL
        try:
            return self._render_matplotlib(time_range, signals, width, height)
        except ImportError:
            return self._render_pil(time_range, signals, width, height)
    
    def _render_matplotlib(self, time_range, signals, width, height) -> bytes:
        """Render using matplotlib for high-quality output."""
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Get time range
        if time_range is None:
            time_range = self.parser.get_time_range()
        
        fig, axes = plt.subplots(len(signals), 1, figsize=(width/100, height/100), 
                                  sharex=True)
        if len(signals) == 1:
            axes = [axes]
        
        fig.suptitle(f'Waveform: {Path(self.vcd_file).name}', fontsize=12)
        
        for ax, sig_name in zip(axes, signals):
            values = self.parser.get_signal_values(sig_name)
            
            if values:
                times = []
                vals = []
                
                for t, v in values:
                    if time_range[0] <= t <= time_range[1]:
                        # Convert to numeric for plotting
                        if isinstance(v, str):
                            if v in ('0', '1'):
                                v = int(v)
                            elif v.startswith('b'):
                                try:
                                    v = int(v[1:], 2)
                                except:
                                    v = 0
                            else:
                                v = 0
                        times.append(t)
                        vals.append(v)
                
                if times:
                    # Step plot for digital signals
                    ax.step(times, vals, where='post', linewidth=1.5)
            
            ax.set_ylabel(sig_name.split('.')[-1], fontsize=8, rotation=0, 
                         ha='right', va='center')
            ax.set_xlim(time_range)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
        
        axes[-1].set_xlabel('Time (ns)', fontsize=9)
        
        plt.tight_layout()
        
        # Save to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    
    def _render_pil(self, time_range, signals, width, height) -> bytes:
        """Render using PIL for lightweight output."""
        from PIL import Image, ImageDraw, ImageFont
        
        if time_range is None:
            time_range = self.parser.get_time_range()
        
        # Create image
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Layout
        margin = 50
        sig_height = (height - 2 * margin) // len(signals)
        time_scale = (width - 2 * margin) / max(1, time_range[1] - time_range[0])
        
        # Draw each signal
        for i, sig_name in enumerate(signals):
            y_base = margin + i * sig_height + sig_height // 2
            
            # Label
            label = sig_name.split('.')[-1][:10]
            draw.text((5, y_base - 5), label, fill='black')
            
            # Get values
            values = self.parser.get_signal_values(sig_name)
            
            prev_x = margin
            prev_val = 0
            
            for t, v in values:
                if t < time_range[0]:
                    continue
                if t > time_range[1]:
                    break
                
                x = margin + int((t - time_range[0]) * time_scale)
                
                # Convert value
                if isinstance(v, str):
                    if v in ('0', '1'):
                        v = int(v)
                    else:
                        v = 0
                
                # Draw step
                y_high = y_base - sig_height // 3
                y_low = y_base + sig_height // 3
                
                y_prev = y_high if prev_val else y_low
                y_curr = y_high if v else y_low
                
                # Horizontal line to transition
                draw.line([(prev_x, y_prev), (x, y_prev)], fill='blue', width=2)
                # Vertical transition
                draw.line([(x, y_prev), (x, y_curr)], fill='blue', width=2)
                
                prev_x = x
                prev_val = v
            
            # Draw to end
            y_prev = y_base - sig_height // 3 if prev_val else y_base + sig_height // 3
            draw.line([(prev_x, y_prev), (width - margin, y_prev)], fill='blue', width=2)
        
        # Save to bytes
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        return buf.read()
    
    def look(self, question: str = "Describe what you see in this waveform.",
             time_range: Tuple[int, int] = None,
             signals: List[str] = None) -> VisionAnalysis:
        """
        Show the waveform to the LLM and ask a question.
        
        Args:
            question: What to ask about the waveform
            time_range: Time window to show (None = full simulation)
            signals: Signals to include (None = auto-selected)
        
        Returns:
            VisionAnalysis with LLM's observations
        """
        # Render waveform image
        image_bytes = self.render_image(time_range, signals)
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Build prompt
        system_prompt = """You are an expert digital hardware engineer analyzing waveform diagrams.

When looking at waveforms, analyze:
1. Clock signals - frequency, duty cycle, stability
2. Handshake protocols - valid/ready, request/acknowledge patterns
3. Data flow - when data changes relative to clocks
4. Timing relationships - setup/hold, propagation delays
5. Error conditions - glitches, stuck signals, protocol violations
6. State machines - state transitions, stuck states

Be specific about what you observe at what times."""

        user_prompt = f"""Look at this waveform from a digital simulation.

{question}

Please provide:
1. A brief description of what's happening
2. Key observations (timing, patterns, relationships)
3. Any issues or concerns you notice
4. Suggestions for the engineer"""

        # Send to vision LLM
        if self.provider == "openai":
            response = self._analyze_openai(system_prompt, user_prompt, image_b64)
        elif self.provider == "anthropic":
            response = self._analyze_anthropic(system_prompt, user_prompt, image_b64)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
        
        # Parse response
        return self._parse_response(response)
    
    def _analyze_openai(self, system: str, user: str, image_b64: str) -> str:
        """Send image to GPT-4 Vision."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        
        response = self._client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1500
        )
        
        return response.choices[0].message.content
    
    def _analyze_anthropic(self, system: str, user: str, image_b64: str) -> str:
        """Send image to Claude Vision."""
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        
        response = self._client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system=system,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b64
                            }
                        },
                        {
                            "type": "text",
                            "text": user
                        }
                    ]
                }
            ]
        )
        
        return response.content[0].text
    
    def _parse_response(self, response: str) -> VisionAnalysis:
        """Parse LLM response into structured analysis."""
        observations = []
        issues = []
        suggestions = []
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            lower = line.lower()
            if 'observation' in lower or 'key observation' in lower:
                current_section = 'obs'
            elif 'issue' in lower or 'concern' in lower or 'problem' in lower:
                current_section = 'issue'
            elif 'suggestion' in lower or 'recommend' in lower:
                current_section = 'sug'
            elif line.startswith(('-', '•', '*', '1', '2', '3', '4', '5')):
                # Extract bullet point content
                content = line.lstrip('-•*0123456789.) ')
                if current_section == 'obs':
                    observations.append(content)
                elif current_section == 'issue':
                    issues.append(content)
                elif current_section == 'sug':
                    suggestions.append(content)
        
        return VisionAnalysis(
            description=response[:500] if len(response) > 500 else response,
            observations=observations,
            issues_found=issues,
            suggestions=suggestions,
            raw_response=response
        )
    
    def look_at(self, start_time: int, end_time: int, 
                question: str = "What's happening in this time window?") -> VisionAnalysis:
        """Look at a specific time window."""
        return self.look(question, time_range=(start_time, end_time))
    
    def compare_regions(self, region1: Tuple[int, int], region2: Tuple[int, int],
                       question: str = "Compare these two time regions") -> str:
        """Compare two time regions visually."""
        # Render both regions
        img1 = self.render_image(region1)
        img2 = self.render_image(region2)
        
        # Combine images vertically
        from PIL import Image
        
        i1 = Image.open(io.BytesIO(img1))
        i2 = Image.open(io.BytesIO(img2))
        
        combined = Image.new('RGB', (i1.width, i1.height + i2.height + 20), 'white')
        combined.paste(i1, (0, 0))
        combined.paste(i2, (0, i1.height + 20))
        
        buf = io.BytesIO()
        combined.save(buf, format='PNG')
        buf.seek(0)
        
        image_b64 = base64.b64encode(buf.read()).decode('utf-8')
        
        prompt = f"""These are two time regions from the same simulation:
- Top: t={region1[0]} to t={region1[1]}
- Bottom: t={region2[0]} to t={region2[1]}

{question}

Identify differences, changes in behavior, or anomalies between the two regions."""

        if self.provider == "openai":
            return self._analyze_openai("You are a digital hardware engineer.", prompt, image_b64)
        else:
            return self._analyze_anthropic("You are a digital hardware engineer.", prompt, image_b64)
    
    def save_snapshot(self, output_path: str,
                     time_range: Tuple[int, int] = None,
                     signals: List[str] = None):
        """Save waveform image to file."""
        image_bytes = self.render_image(time_range, signals)
        with open(output_path, 'wb') as f:
            f.write(image_bytes)
        return output_path


class LiveVisionMonitor:
    """
    Real-time visual monitoring of waveforms.
    
    Takes periodic snapshots and has AI analyze them for anomalies.
    
    Usage:
        monitor = LiveVisionMonitor("simulation.vcd")
        monitor.add_watch("error", "Alert me when you see error go high")
        monitor.start()  # Begins monitoring
    """
    
    def __init__(self, vcd_file: str, 
                 interval: float = 2.0,
                 provider: str = "openai"):
        """
        Args:
            vcd_file: VCD file to monitor
            interval: Seconds between analysis frames
            provider: Vision LLM provider
        """
        self.vision = WaveformVision(vcd_file, provider)
        self.interval = interval
        self.watches: List[Dict] = []
        self._running = False
        self._last_time = 0
    
    def add_watch(self, name: str, condition: str):
        """
        Add a condition to watch for.
        
        The AI will be asked to check for this condition in each frame.
        """
        self.watches.append({"name": name, "condition": condition})
    
    def analyze_frame(self, window_size: int = 100) -> Dict[str, Any]:
        """Analyze the current frame."""
        # Get current end time
        _, end_time = self.vision.parser.get_time_range()
        
        # Only analyze new data
        if end_time <= self._last_time:
            return {"status": "no_new_data"}
        
        # Analyze window
        start = max(0, end_time - window_size)
        
        # Build question from watches
        watch_questions = "\n".join([
            f"- {w['name']}: {w['condition']}"
            for w in self.watches
        ])
        
        question = f"""Quickly analyze this waveform snapshot.

Check for:
{watch_questions}

Report any anomalies or conditions that were triggered."""

        result = self.vision.look(question, time_range=(start, end_time))
        self._last_time = end_time
        
        return {
            "status": "analyzed",
            "time_range": (start, end_time),
            "analysis": result
        }
    
    def start(self, callback=None):
        """
        Start continuous monitoring.
        
        Args:
            callback: Function to call with each analysis result
        """
        import time
        
        print(f"👁️  Starting visual monitoring (every {self.interval}s)")
        print(f"   Watches: {[w['name'] for w in self.watches]}")
        
        self._running = True
        
        try:
            while self._running:
                result = self.analyze_frame()
                
                if result["status"] == "analyzed":
                    print(f"\n[t={result['time_range'][1]}] Analysis:")
                    analysis = result["analysis"]
                    
                    if analysis.issues_found:
                        print("  ⚠️  Issues found:")
                        for issue in analysis.issues_found:
                            print(f"      - {issue}")
                    else:
                        print("  ✓ No issues detected")
                    
                    if callback:
                        callback(result)
                
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            print("\n👁️  Monitoring stopped")
    
    def stop(self):
        """Stop monitoring."""
        self._running = False
