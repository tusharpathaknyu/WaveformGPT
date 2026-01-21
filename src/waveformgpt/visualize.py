"""
Waveform Visualization Engine.

Provides ASCII art, matplotlib plots, and interactive visualization.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import math

from waveformgpt.vcd_parser import VCDParser, Signal


@dataclass
class WaveformStyle:
    """Styling options for waveform rendering."""
    width: int = 80
    height: int = 3
    high_char: str = "█"
    low_char: str = "_"
    rise_char: str = "/"
    fall_char: str = "\\"
    unknown_char: str = "?"
    highz_char: str = "-"
    bus_chars: Tuple[str, str] = ("╱", "╲")
    time_unit: str = "ns"


class ASCIIWaveform:
    """
    Generate ASCII art waveform diagrams.
    
    Produces text-based waveforms suitable for terminal display,
    documentation, and chat interfaces.
    """
    
    def __init__(self, parser: VCDParser, style: WaveformStyle = None):
        self.parser = parser
        self.style = style or WaveformStyle()
    
    def render_signal(self, 
                      signal_name: str,
                      start_time: int = None,
                      end_time: int = None,
                      width: int = None) -> str:
        """
        Render a single signal as ASCII waveform.
        
        Args:
            signal_name: Signal to render
            start_time: Start of time window
            end_time: End of time window
            width: Character width of output
        
        Returns:
            ASCII art string
        """
        try:
            values = self.parser.get_signal_values(signal_name)
        except ValueError:
            return f"{signal_name}: (no data)"
        if not values:
            return f"{signal_name}: (no data)"
        
        # Get signal metadata
        signals = self.parser.search_signals(f"^{signal_name}$")
        if not signals:
            # Try exact name match
            signals = [s for s in self.parser.search_signals(".*") if s.name == signal_name]
        sig_width = signals[0].width if signals else 1
        
        # Apply time window
        if start_time is not None:
            values = [(t, v) for t, v in values if t >= start_time]
        if end_time is not None:
            values = [(t, v) for t, v in values if t <= end_time]
        
        if not values:
            return f"{signal_name}: (no data in range)"
        
        width = width or self.style.width
        
        if sig_width == 1:
            return self._render_bit_signal(signal_name, values, width)
        else:
            return self._render_bus_signal(signal_name, values, width, sig_width)
    
    def _render_bit_signal(self, name: str, values: List[Tuple[int, str]], 
                           width: int) -> str:
        """Render single-bit signal."""
        if not values:
            return f"{name}: (empty)"
        
        time_range = values[-1][0] - values[0][0]
        if time_range == 0:
            time_range = 1
        
        scale = (width - len(name) - 3) / time_range
        output = [" "] * (width - len(name) - 2)
        
        prev_val = None
        prev_pos = 0
        
        for time, value in values:
            pos = int((time - values[0][0]) * scale)
            pos = min(pos, len(output) - 1)
            
            if prev_val is not None:
                # Draw horizontal line
                char = self._get_level_char(prev_val)
                for i in range(prev_pos, pos):
                    if i < len(output):
                        output[i] = char
            
            # Draw transition
            if prev_val is not None and prev_val != value:
                if pos < len(output):
                    if self._is_high(value) and not self._is_high(prev_val):
                        output[pos] = self.style.rise_char
                    elif not self._is_high(value) and self._is_high(prev_val):
                        output[pos] = self.style.fall_char
            
            prev_val = value
            prev_pos = pos
        
        # Fill remaining
        if prev_val is not None:
            char = self._get_level_char(prev_val)
            for i in range(prev_pos, len(output)):
                output[i] = char
        
        return f"{name}: {''.join(output)}"
    
    def _render_bus_signal(self, name: str, values: List[Tuple[int, str]], 
                           width: int, bus_width: int) -> str:
        """Render multi-bit bus signal."""
        if not values:
            return f"{name}: (empty)"
        
        time_range = values[-1][0] - values[0][0]
        if time_range == 0:
            time_range = 1
        
        avail_width = width - len(name) - 3
        scale = avail_width / time_range
        
        # Top and bottom lines
        top = [" "] * avail_width
        mid = [" "] * avail_width
        bot = [" "] * avail_width
        
        prev_val = None
        prev_pos = 0
        
        for time, value in values:
            pos = int((time - values[0][0]) * scale)
            pos = min(pos, avail_width - 1)
            
            if prev_val is not None and prev_pos < pos:
                # Draw bus lines
                for i in range(prev_pos + 1, pos):
                    if i < avail_width:
                        top[i] = "─"
                        bot[i] = "─"
                
                # Show value in middle
                hex_val = self._format_bus_value(prev_val, bus_width)
                val_pos = prev_pos + 1
                for c in hex_val:
                    if val_pos < pos and val_pos < avail_width:
                        mid[val_pos] = c
                        val_pos += 1
            
            # Draw transition markers
            if prev_val is not None and prev_val != value:
                if pos > 0:
                    top[pos - 1] = "╲"
                    bot[pos - 1] = "╱"
                if pos < avail_width:
                    top[pos] = "╱"
                    bot[pos] = "╲"
            
            prev_val = value
            prev_pos = pos
        
        # Fill remaining
        if prev_val is not None:
            for i in range(prev_pos + 1, avail_width):
                top[i] = "─"
                bot[i] = "─"
            
            hex_val = self._format_bus_value(prev_val, bus_width)
            val_pos = prev_pos + 1
            for c in hex_val:
                if val_pos < avail_width:
                    mid[val_pos] = c
                    val_pos += 1
        
        name_pad = " " * len(name)
        return f"{name_pad}  {''.join(top)}\n{name}: {''.join(mid)}\n{name_pad}  {''.join(bot)}"
    
    def render_multiple(self, 
                         signals: List[str],
                         start_time: int = None,
                         end_time: int = None,
                         width: int = None) -> str:
        """Render multiple signals aligned."""
        lines = []
        
        # Calculate name width for alignment
        max_name_len = max(len(s) for s in signals) if signals else 0
        
        for sig in signals:
            padded_name = sig.rjust(max_name_len)
            waveform = self.render_signal(sig, start_time, end_time, width)
            # Replace signal name with padded version
            waveform = waveform.replace(f"{sig}:", f"{padded_name}:", 1)
            lines.append(waveform)
        
        # Add time axis
        if signals:
            values = self.parser.get_signal_values(signals[0])
            if values:
                t_start = start_time if start_time else values[0][0]
                t_end = end_time if end_time else values[-1][0]
                axis = self._render_time_axis(t_start, t_end, width or self.style.width, max_name_len)
                lines.append(axis)
        
        return "\n".join(lines)
    
    def _render_time_axis(self, t_start: int, t_end: int, 
                          width: int, name_width: int) -> str:
        """Render time axis."""
        axis_width = width - name_width - 3
        
        # Create tick marks
        axis = [" "] * axis_width
        labels = []
        
        num_ticks = min(5, axis_width // 10)
        for i in range(num_ticks + 1):
            pos = int(i * (axis_width - 1) / num_ticks)
            time = t_start + (t_end - t_start) * i / num_ticks
            axis[pos] = "|"
            labels.append((pos, f"{int(time)}"))
        
        # Add labels
        label_line = [" "] * axis_width
        for pos, label in labels:
            for j, c in enumerate(label):
                if pos + j < axis_width:
                    label_line[pos + j] = c
        
        prefix = " " * (name_width + 2)
        return f"{prefix}{''.join(axis)}\n{prefix}{''.join(label_line)}"
    
    def _get_level_char(self, value: str) -> str:
        """Get character for signal level."""
        if not value:
            return self.style.unknown_char
        
        if value in ('x', 'X'):
            return self.style.unknown_char
        elif value in ('z', 'Z'):
            return self.style.highz_char
        elif self._is_high(value):
            return self.style.high_char
        else:
            return self.style.low_char
    
    def _is_high(self, value: str) -> bool:
        """Check if value is logic high."""
        if not value:
            return False
        return value in ('1', 'H', 'h')
    
    def _format_bus_value(self, value: str, width: int) -> str:
        """Format bus value as hex."""
        if not value:
            return "?"
        
        if value.startswith('b'):
            # Binary value
            binary = value[1:]
            if 'x' in binary.lower() or 'z' in binary.lower():
                return "??"
            try:
                int_val = int(binary, 2)
                hex_len = (width + 3) // 4
                return f"{int_val:0{hex_len}x}"
            except ValueError:
                return "??"
        
        return value[:8]  # Truncate long values


def render_to_html(parser: VCDParser, 
                   signals: List[str],
                   start_time: int = None,
                   end_time: int = None,
                   output_path: str = None) -> str:
    """
    Render waveforms to interactive HTML.
    
    Args:
        parser: VCD parser instance
        signals: List of signal names to render
        start_time: Optional start time
        end_time: Optional end time
        output_path: Optional path to save HTML
    
    Returns:
        HTML string
    """
    html_template = '''<!DOCTYPE html>
<html>
<head>
    <title>WaveformGPT Viewer</title>
    <style>
        body {{ font-family: monospace; background: #1e1e1e; color: #d4d4d4; padding: 20px; }}
        .signal-row {{ display: flex; margin: 2px 0; }}
        .signal-name {{ width: 150px; text-align: right; padding-right: 10px; color: #569cd6; }}
        .waveform {{ flex: 1; height: 30px; position: relative; }}
        .wave-bit {{ height: 100%; position: relative; }}
        .wave-high {{ position: absolute; top: 5px; height: 10px; background: #4ec9b0; }}
        .wave-low {{ position: absolute; bottom: 5px; height: 10px; background: #4ec9b0; }}
        .wave-trans {{ position: absolute; width: 2px; background: #4ec9b0; height: 100%; }}
        .timeline {{ display: flex; border-top: 1px solid #404040; margin-top: 10px; padding-top: 5px; }}
        .time-label {{ flex: 1; text-align: center; color: #808080; }}
        h1 {{ color: #4ec9b0; }}
    </style>
</head>
<body>
    <h1>WaveformGPT Waveform Viewer</h1>
    <div id="waveforms">
        {waveforms}
    </div>
    <div class="timeline" id="timeline">
        {timeline}
    </div>
</body>
</html>
'''
    
    waveform_html = []
    
    for sig_name in signals:
        try:
            values = parser.get_signal_values(sig_name)
        except ValueError:
            continue
            
        if not values:
            continue
        
        # Apply time window
        if start_time is not None:
            values = [(t, v) for t, v in values if t >= start_time]
        if end_time is not None:
            values = [(t, v) for t, v in values if t <= end_time]
        
        if not values:
            continue
        
        t_min = values[0][0]
        t_max = values[-1][0]
        t_range = t_max - t_min if t_max > t_min else 1
        
        wave_divs = []
        prev_val = None
        prev_pos = 0
        
        for time, value in values:
            pos_pct = ((time - t_min) / t_range) * 100
            
            if prev_val is not None:
                width_pct = pos_pct - prev_pos
                is_high = value in ('1', 'H', 'h')
                css_class = "wave-high" if prev_val in ('1', 'H', 'h') else "wave-low"
                wave_divs.append(
                    f'<div class="{css_class}" style="left:{prev_pos}%;width:{width_pct}%"></div>'
                )
                
                if prev_val != value:
                    wave_divs.append(
                        f'<div class="wave-trans" style="left:{pos_pct}%"></div>'
                    )
            
            prev_val = value
            prev_pos = pos_pct
        
        waveform_html.append(f'''
        <div class="signal-row">
            <div class="signal-name">{sig_name}</div>
            <div class="waveform">
                <div class="wave-bit">
                    {''.join(wave_divs)}
                </div>
            </div>
        </div>
        ''')
    
    # Timeline
    if signals:
        try:
            values = parser.get_signal_values(signals[0])
            t_min = start_time if start_time else values[0][0]
            t_max = end_time if end_time else values[-1][0]
            
            timeline_html = []
            for i in range(5):
                t = t_min + (t_max - t_min) * i / 4
                timeline_html.append(f'<div class="time-label">{int(t)}</div>')
            timeline = ''.join(timeline_html)
        except (ValueError, IndexError):
            timeline = ''
    else:
        timeline = ''
    
    html = html_template.format(
        waveforms=''.join(waveform_html),
        timeline=timeline
    )
    
    if output_path:
        Path(output_path).write_text(html)
    
    return html


try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.collections import LineCollection
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def render_to_matplotlib(parser: VCDParser,
                          signals: List[str],
                          start_time: int = None,
                          end_time: int = None,
                          output_path: str = None,
                          figsize: Tuple[int, int] = (12, 6)):
    """
    Render waveforms using matplotlib.
    
    Args:
        parser: VCD parser instance
        signals: List of signal names
        start_time: Optional start time
        end_time: Optional end time
        output_path: Optional path to save figure
        figsize: Figure size tuple
    
    Returns:
        matplotlib Figure object
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    
    fig, axes = plt.subplots(len(signals), 1, figsize=figsize, sharex=True)
    if len(signals) == 1:
        axes = [axes]
    
    fig.patch.set_facecolor('#1e1e1e')
    
    for ax, sig_name in zip(axes, signals):
        ax.set_facecolor('#1e1e1e')
        
        values = parser.get_signal_values(sig_name)
        if not values:
            continue
        
        # Apply time window
        if start_time is not None:
            values = [(t, v) for t, v in values if t >= start_time]
        if end_time is not None:
            values = [(t, v) for t, v in values if t <= end_time]
        
        if not values:
            continue
        
        # Get signal width
        sig_list = parser.search_signals(f"^{sig_name}$")
        sig_width = sig_list[0].width if sig_list else 1
        
        if sig_width == 1:
            _plot_bit_signal(ax, values, sig_name)
        else:
            _plot_bus_signal(ax, values, sig_name, sig_width)
        
        ax.set_ylabel(sig_name, color='#569cd6', rotation=0, ha='right', va='center')
        ax.tick_params(colors='#808080')
        ax.spines['bottom'].set_color('#404040')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#404040')
    
    axes[-1].set_xlabel('Time', color='#808080')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, facecolor='#1e1e1e', edgecolor='none', dpi=150)
    
    return fig


def _plot_bit_signal(ax, values: List[Tuple[int, str]], name: str):
    """Plot single-bit signal."""
    times = []
    levels = []
    
    for time, value in values:
        times.append(time)
        if value in ('1', 'H', 'h'):
            levels.append(1)
        elif value in ('0', 'L', 'l'):
            levels.append(0)
        else:
            levels.append(0.5)
    
    # Step plot for digital signal
    ax.step(times, levels, where='post', color='#4ec9b0', linewidth=1.5)
    ax.fill_between(times, levels, step='post', alpha=0.3, color='#4ec9b0')
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['0', '1'])


def _plot_bus_signal(ax, values: List[Tuple[int, str]], name: str, width: int):
    """Plot multi-bit bus signal."""
    for i, (time, value) in enumerate(values[:-1]):
        next_time = values[i + 1][0]
        
        # Draw bus outline
        ax.plot([time, next_time], [1, 1], color='#4ec9b0', linewidth=1.5)
        ax.plot([time, next_time], [0, 0], color='#4ec9b0', linewidth=1.5)
        
        # Draw transition
        ax.plot([next_time, next_time], [0, 1], color='#4ec9b0', linewidth=1.5)
        
        # Add value label
        hex_val = _format_hex(value, width)
        mid_time = (time + next_time) / 2
        ax.text(mid_time, 0.5, hex_val, ha='center', va='center', 
                color='#d4d4d4', fontsize=8)
    
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([])


def _format_hex(value: str, width: int) -> str:
    """Format value as hex string."""
    if value.startswith('b'):
        binary = value[1:]
        if 'x' in binary.lower() or 'z' in binary.lower():
            return 'XX'
        try:
            return f"{int(binary, 2):X}"
        except ValueError:
            return '??'
    return value[:4]
