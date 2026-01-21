"""
GTKWave Integration for WaveformGPT.

Generates save files and can optionally launch GTKWave.
"""

from pathlib import Path
from typing import List, Optional
import subprocess
import shutil


def generate_savefile(vcd_file: str,
                      output_path: str,
                      signals: List[str],
                      markers: Optional[List[int]] = None,
                      zoom: Optional[float] = None) -> str:
    """
    Generate a GTKWave save file (.gtkw).
    
    Args:
        vcd_file: Path to the VCD file
        output_path: Path for the output .gtkw file
        signals: List of signal names to display
        markers: Time positions for markers
        zoom: Zoom level (pixels per time unit)
    
    Returns:
        Status message
    """
    vcd_path = Path(vcd_file).resolve()
    output = Path(output_path)
    
    lines = [
        "[*]",
        "[*] WaveformGPT Generated Save File",
        "[*]",
        f"[dumpfile] \"{vcd_path}\"",
        "[dumpfile_mtime] \"0\"",
        "[dumpfile_size] 0",
        "[savefile] \"{}\"".format(output.resolve()),
    ]
    
    # Zoom settings
    if zoom:
        lines.append(f"[timestart] 0")
        lines.append(f"[size] 1920 1080")
        lines.append(f"[pos] 0 0")
    
    # Signal traces
    lines.append("[treeopen] tb.")
    lines.append("[sst_width] 250")
    lines.append("[signals_width] 200")
    lines.append("[sst_expanded] 1")
    
    for sig in signals:
        # GTKWave format for signal
        lines.append(f"@22")  # Signal type (analog/digital)
        lines.append(f"{sig}")
    
    # Markers
    if markers:
        for i, time in enumerate(markers[:26]):  # A-Z markers
            marker_name = chr(ord('A') + i)
            lines.append(f"[marker{marker_name}] {time}")
    
    lines.append("[*]")
    lines.append("[*] End of save file")
    
    # Write file
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    
    return f"Generated GTKWave save file: {output}"


def launch_gtkwave(vcd_file: str, savefile: Optional[str] = None) -> bool:
    """
    Launch GTKWave with the given VCD file.
    
    Args:
        vcd_file: Path to VCD file
        savefile: Optional path to .gtkw save file
    
    Returns:
        True if launched successfully
    """
    # Find GTKWave executable
    gtkwave_cmd = shutil.which("gtkwave")
    
    if not gtkwave_cmd:
        # Try common locations
        common_paths = [
            "/Applications/gtkwave.app/Contents/Resources/bin/gtkwave",  # macOS
            "/usr/bin/gtkwave",
            "/usr/local/bin/gtkwave",
        ]
        for path in common_paths:
            if Path(path).exists():
                gtkwave_cmd = path
                break
    
    if not gtkwave_cmd:
        print("GTKWave not found. Please install GTKWave and add it to PATH.")
        return False
    
    cmd = [gtkwave_cmd, vcd_file]
    if savefile:
        cmd.extend(["-a", savefile])
    
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        print(f"Failed to launch GTKWave: {e}")
        return False


def generate_surfer_config(vcd_file: str,
                           output_path: str,
                           signals: List[str],
                           cursors: Optional[List[int]] = None) -> str:
    """
    Generate a Surfer waveform viewer config file.
    
    Surfer is a modern alternative to GTKWave with better performance.
    https://gitlab.com/nickelized/surfer
    
    Args:
        vcd_file: Path to VCD file
        output_path: Path for output config
        signals: Signals to display
        cursors: Cursor time positions
    
    Returns:
        Status message
    """
    import json
    
    config = {
        "source": str(Path(vcd_file).resolve()),
        "signals": [{"path": sig} for sig in signals],
        "cursors": [{"time": t} for t in (cursors or [])],
        "view": {
            "zoom_to_fit": True
        }
    }
    
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2))
    
    return f"Generated Surfer config: {output}"
