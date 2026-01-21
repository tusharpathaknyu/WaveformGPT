"""
Waveform Export Utilities.

Export waveforms to various formats: CSV, JSON, Markdown tables,
SystemVerilog constraints, and more.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import json

from waveformgpt.vcd_parser import VCDParser


@dataclass
class ExportOptions:
    """Options for waveform export."""
    signals: Optional[List[str]] = None
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    time_step: Optional[int] = None  # Sample at intervals
    include_header: bool = True
    timestamp_format: str = "decimal"  # "decimal", "hex", "scientific"


def export_to_csv(parser: VCDParser,
                   output_path: str,
                   options: ExportOptions = None) -> str:
    """
    Export waveform data to CSV format.
    
    Args:
        parser: VCD parser instance
        output_path: Path to output CSV file
        options: Export options
    
    Returns:
        Path to created file
    """
    options = options or ExportOptions()
    
    # Determine signals to export
    if options.signals:
        signals = options.signals
    else:
        signals = [s.name for s in parser.search_signals(".*")]
    
    # Collect all unique timestamps
    all_times = set()
    signal_data = {}
    
    for sig_name in signals:
        values = parser.get_signal_values(sig_name)
        if values:
            signal_data[sig_name] = dict(values)
            for time, _ in values:
                if options.start_time and time < options.start_time:
                    continue
                if options.end_time and time > options.end_time:
                    continue
                all_times.add(time)
    
    all_times = sorted(all_times)
    
    # Apply time stepping if requested
    if options.time_step:
        if all_times:
            start = all_times[0]
            end = all_times[-1]
            all_times = list(range(start, end + 1, options.time_step))
    
    # Generate CSV
    lines = []
    
    if options.include_header:
        lines.append("time," + ",".join(signals))
    
    # Track current values
    current_values = {sig: "" for sig in signals}
    
    for time in all_times:
        row = [_format_time(time, options.timestamp_format)]
        
        for sig_name in signals:
            if sig_name in signal_data and time in signal_data[sig_name]:
                current_values[sig_name] = signal_data[sig_name][time]
            row.append(current_values[sig_name])
        
        lines.append(",".join(row))
    
    csv_content = "\n".join(lines)
    Path(output_path).write_text(csv_content)
    
    return output_path


def export_to_json(parser: VCDParser,
                    output_path: str,
                    options: ExportOptions = None) -> str:
    """
    Export waveform data to JSON format.
    
    Args:
        parser: VCD parser instance
        output_path: Path to output JSON file
        options: Export options
    
    Returns:
        Path to created file
    """
    options = options or ExportOptions()
    
    # Determine signals
    if options.signals:
        signals = options.signals
    else:
        signals = [s.name for s in parser.search_signals(".*")]
    
    # Build data structure
    data = {
        "metadata": {
            "timescale": parser.timescale,
            "version": parser.version,
            "date": parser.date,
        },
        "signals": {}
    }
    
    for sig_name in signals:
        sig_list = parser.search_signals(f"^{sig_name}$")
        if sig_list:
            sig = sig_list[0]
            values = parser.get_signal_values(sig_name)
            
            # Apply time window
            if options.start_time or options.end_time:
                values = [
                    (t, v) for t, v in values
                    if (options.start_time is None or t >= options.start_time)
                    and (options.end_time is None or t <= options.end_time)
                ]
            
            data["signals"][sig_name] = {
                "path": sig.path,
                "width": sig.width,
                "type": sig.var_type,
                "changes": [{"time": t, "value": v} for t, v in values]
            }
    
    json_content = json.dumps(data, indent=2)
    Path(output_path).write_text(json_content)
    
    return output_path


def export_to_markdown(parser: VCDParser,
                        signals: List[str] = None,
                        start_time: int = None,
                        end_time: int = None,
                        max_columns: int = 10) -> str:
    """
    Export waveform data to Markdown table.
    
    Args:
        parser: VCD parser instance
        signals: Signals to include
        start_time: Start of time window
        end_time: End of time window
        max_columns: Maximum time columns
    
    Returns:
        Markdown table string
    """
    if signals is None:
        signals = [s.name for s in parser.search_signals(".*")][:5]  # Limit signals
    
    # Collect timestamps
    all_times = set()
    signal_data = {}
    
    for sig_name in signals:
        values = parser.get_signal_values(sig_name)
        if values:
            signal_data[sig_name] = dict(values)
            for time, _ in values:
                if start_time and time < start_time:
                    continue
                if end_time and time > end_time:
                    continue
                all_times.add(time)
    
    all_times = sorted(all_times)
    
    # Limit columns
    if len(all_times) > max_columns:
        step = len(all_times) // max_columns
        all_times = all_times[::step][:max_columns]
    
    # Build table
    lines = []
    
    # Header
    header = "| Signal | " + " | ".join(str(t) for t in all_times) + " |"
    separator = "|--------|" + "|".join("-" * 6 for _ in all_times) + "|"
    lines.append(header)
    lines.append(separator)
    
    # Data rows
    current_values = {sig: "" for sig in signals}
    
    for sig_name in signals:
        row = [sig_name]
        
        for time in all_times:
            if sig_name in signal_data:
                # Find value at this time
                for t in sorted(signal_data[sig_name].keys()):
                    if t <= time:
                        current_values[sig_name] = signal_data[sig_name][t]
                    else:
                        break
            
            value = current_values[sig_name]
            # Format for display
            if value.startswith('b'):
                try:
                    int_val = int(value[1:], 2)
                    value = f"0x{int_val:X}"
                except ValueError:
                    pass
            
            row.append(value[:6])  # Truncate long values
        
        lines.append("| " + " | ".join(row) + " |")
    
    return "\n".join(lines)


def export_to_wavedrom(parser: VCDParser,
                        signals: List[str] = None,
                        start_time: int = None,
                        end_time: int = None,
                        clock_signal: str = None) -> str:
    """
    Export to WaveDrom JSON format for documentation.
    
    Args:
        parser: VCD parser instance
        signals: Signals to include
        start_time: Start time
        end_time: End time
        clock_signal: Optional clock signal for reference
    
    Returns:
        WaveDrom JSON string
    """
    if signals is None:
        signals = [s.name for s in parser.search_signals(".*")][:8]
    
    wavedrom = {
        "signal": [],
        "head": {"text": "Waveform"},
        "config": {"hscale": 1}
    }
    
    # Determine time points
    all_times = set()
    signal_data = {}
    
    for sig_name in signals:
        values = parser.get_signal_values(sig_name)
        if values:
            signal_data[sig_name] = values
            for time, _ in values:
                if start_time and time < start_time:
                    continue
                if end_time and time > end_time:
                    continue
                all_times.add(time)
    
    all_times = sorted(all_times)[:20]  # Limit for readability
    
    # Generate wave strings
    for sig_name in signals:
        if sig_name not in signal_data:
            continue
        
        values = signal_data[sig_name]
        sig_list = parser.search_signals(f"^{sig_name}$")
        is_bus = sig_list and sig_list[0].width > 1
        
        wave = ""
        data = []
        current_val = None
        
        for i, time in enumerate(all_times):
            # Find value at this time
            val = None
            for t, v in values:
                if t <= time:
                    val = v
                else:
                    break
            
            if is_bus:
                if val != current_val:
                    wave += "="
                    # Format value
                    if val and val.startswith('b'):
                        try:
                            int_val = int(val[1:], 2)
                            data.append(f"0x{int_val:X}")
                        except ValueError:
                            data.append(val)
                    else:
                        data.append(val or "?")
                else:
                    wave += "."
            else:
                if val in ('1', 'H', 'h'):
                    wave += "1" if val != current_val else "."
                elif val in ('0', 'L', 'l'):
                    wave += "0" if val != current_val else "."
                elif val and 'x' in val.lower():
                    wave += "x"
                else:
                    wave += "."
            
            current_val = val
        
        signal_entry = {"name": sig_name, "wave": wave}
        if data:
            signal_entry["data"] = data
        
        wavedrom["signal"].append(signal_entry)
    
    return json.dumps(wavedrom, indent=2)


def export_to_systemverilog(parser: VCDParser,
                              signal: str,
                              output_path: str = None) -> str:
    """
    Generate SystemVerilog stimulus from waveform.
    
    Args:
        parser: VCD parser instance
        signal: Signal to export
        output_path: Optional output file path
    
    Returns:
        SystemVerilog code string
    """
    values = parser.get_signal_values(signal)
    if not values:
        return f"// No data for signal: {signal}"
    
    sig_list = parser.search_signals(f"^{signal}$")
    width = sig_list[0].width if sig_list else 1
    
    lines = [
        f"// Generated by WaveformGPT",
        f"// Signal: {signal}",
        f"",
        f"initial begin",
    ]
    
    prev_time = 0
    for time, value in values:
        delay = time - prev_time
        
        # Format value
        if value.startswith('b'):
            formatted = f"{width}'b{value[1:]}"
        elif value in ('0', '1'):
            formatted = f"1'b{value}"
        else:
            formatted = f"1'b{value}"
        
        if delay > 0:
            lines.append(f"    #{delay};")
        lines.append(f"    {signal} = {formatted};")
        
        prev_time = time
    
    lines.extend([
        f"end",
        f"",
    ])
    
    code = "\n".join(lines)
    
    if output_path:
        Path(output_path).write_text(code)
    
    return code


def export_to_cocotb(parser: VCDParser,
                      signals: List[str],
                      output_path: str = None) -> str:
    """
    Generate cocotb test stimulus from waveform.
    
    Args:
        parser: VCD parser instance
        signals: Signals to export
        output_path: Optional output file path
    
    Returns:
        Python cocotb code string
    """
    lines = [
        "# Generated by WaveformGPT",
        "import cocotb",
        "from cocotb.triggers import Timer",
        "",
        "@cocotb.test()",
        "async def replay_waveform(dut):",
        '    """Replay captured waveform."""',
        "",
    ]
    
    # Collect all events
    events = []
    for sig_name in signals:
        values = parser.get_signal_values(sig_name)
        if values:
            for time, value in values:
                events.append((time, sig_name, value))
    
    # Sort by time
    events.sort(key=lambda x: x[0])
    
    prev_time = 0
    for time, sig_name, value in events:
        delay = time - prev_time
        
        if delay > 0:
            lines.append(f"    await Timer({delay}, units='ns')")
        
        # Format value assignment
        safe_name = sig_name.replace('.', '_').replace('[', '_').replace(']', '')
        
        if value.startswith('b'):
            int_val = int(value[1:], 2) if 'x' not in value.lower() else 0
            lines.append(f"    dut.{safe_name}.value = {int_val}")
        elif value in ('0', '1'):
            lines.append(f"    dut.{safe_name}.value = {value}")
        
        prev_time = time
    
    lines.append("")
    
    code = "\n".join(lines)
    
    if output_path:
        Path(output_path).write_text(code)
    
    return code


def _format_time(time: int, fmt: str) -> str:
    """Format timestamp according to format option."""
    if fmt == "hex":
        return f"0x{time:X}"
    elif fmt == "scientific":
        return f"{time:.2e}"
    else:
        return str(time)
