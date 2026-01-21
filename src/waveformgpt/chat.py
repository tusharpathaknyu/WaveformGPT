"""
Chat Interface for WaveformGPT.

Provides an interactive conversational interface for waveform queries.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

from waveformgpt.vcd_parser import VCDParser
from waveformgpt.query_engine import QueryEngine, QueryResult
from waveformgpt.nl_parser import NLParser


@dataclass
class Message:
    """Chat message."""
    role: str  # "user" or "assistant"
    content: str
    result: Optional[QueryResult] = None


class WaveformChat:
    """
    Interactive chat interface for waveform queries.
    
    Usage:
        chat = WaveformChat("simulation.vcd")
        
        response = chat.ask("When does the clock rise?")
        print(response.content)
        
        response = chat.ask("Show me all reset events")
        print(response.result.matches)
    """
    
    def __init__(self, vcd_file: Optional[str] = None, use_llm: bool = False):
        self.vcd_file = vcd_file
        self.use_llm = use_llm
        
        self.parser: Optional[VCDParser] = None
        self.engine: Optional[QueryEngine] = None
        self.nl_parser: Optional[NLParser] = None
        
        self.history: List[Message] = []
        
        if vcd_file:
            self.load_vcd(vcd_file)
    
    def load_vcd(self, filepath: str) -> str:
        """Load a VCD file for querying."""
        try:
            self.vcd_file = filepath
            self.parser = VCDParser(filepath)
            self.engine = QueryEngine(self.parser)
            
            # Get signal list for NL parser
            signals = [s.full_name for s in self.parser.header.signals.values()]
            self.nl_parser = NLParser(
                available_signals=signals,
                use_llm=self.use_llm
            )
            
            time_range = self.parser.get_time_range()
            
            msg = (
                f"Loaded VCD file: {Path(filepath).name}\n"
                f"  Signals: {len(signals)}\n"
                f"  Time range: {time_range[0]} to {time_range[1]} {self.parser.header.timescale_unit}\n"
            )
            
            return msg
            
        except Exception as e:
            return f"Error loading VCD: {str(e)}"
    
    def ask(self, question: str) -> Message:
        """
        Ask a natural language question about the waveform.
        
        Args:
            question: Natural language query
            
        Returns:
            Message with response and any query results
        """
        # Record user message
        self.history.append(Message(role="user", content=question))
        
        # Check if VCD is loaded
        if self.parser is None:
            response = Message(
                role="assistant",
                content="Please load a VCD file first using: load <filepath>"
            )
            self.history.append(response)
            return response
        
        # Handle special commands
        if question.lower().startswith("load "):
            filepath = question[5:].strip()
            result = self.load_vcd(filepath)
            response = Message(role="assistant", content=result)
            self.history.append(response)
            return response
        
        if question.lower() in ("signals", "list signals", "show signals"):
            return self._list_signals()
        
        if question.lower().startswith("help"):
            return self._show_help()
        
        # Parse natural language query
        parsed = self.nl_parser.parse(question)
        
        if parsed.query is None:
            response = Message(
                role="assistant",
                content=f"I couldn't understand that query. {parsed.interpretation}\n\nTry questions like:\n"
                        "- When does clk rise?\n"
                        "- How many times does reset fall?\n"
                        "- Show me data_valid followed by data_ready"
            )
            self.history.append(response)
            return response
        
        # Execute query
        try:
            result = self.engine.execute(parsed.query)
            
            content = self._format_result(result, parsed.interpretation)
            
            response = Message(
                role="assistant",
                content=content,
                result=result
            )
            
        except Exception as e:
            response = Message(
                role="assistant",
                content=f"Error executing query: {str(e)}"
            )
        
        self.history.append(response)
        return response
    
    def _list_signals(self) -> Message:
        """List all available signals."""
        if not self.parser:
            return Message(role="assistant", content="No VCD loaded")
        
        signals = sorted([s.full_name for s in self.parser.header.signals.values()])
        
        # Group by scope
        scopes: Dict[str, List[str]] = {}
        for sig in signals:
            parts = sig.rsplit('.', 1)
            scope = parts[0] if len(parts) > 1 else "(root)"
            name = parts[-1]
            if scope not in scopes:
                scopes[scope] = []
            scopes[scope].append(name)
        
        lines = ["Available signals:\n"]
        for scope, names in sorted(scopes.items()):
            lines.append(f"\n{scope}:")
            for name in names[:20]:
                lines.append(f"  - {name}")
            if len(names) > 20:
                lines.append(f"  ... and {len(names) - 20} more")
        
        response = Message(role="assistant", content="\n".join(lines))
        self.history.append(response)
        return response
    
    def _show_help(self) -> Message:
        """Show help message."""
        help_text = """
WaveformGPT Query Examples:

**Finding Events:**
- "When does clk rise?"
- "Find all reset falling edges"
- "Show me when data_valid goes high"

**Counting:**
- "How many times does ack rise?"
- "Count clock transitions"

**Timing Analysis:**
- "Measure time between req and ack"
- "What's the average latency from valid to ready?"

**Pattern Matching:**
- "Find request followed by acknowledge within 10 cycles"
- "Show me write_enable then write_done"

**Special Commands:**
- `signals` - List all signals
- `load <file>` - Load a VCD file
- `help` - Show this help

**Tips:**
- Signal names are matched flexibly (partial names work)
- Use 'rise', 'fall', 'high', 'low', 'change' for conditions
"""
        response = Message(role="assistant", content=help_text)
        self.history.append(response)
        return response
    
    def _format_result(self, result: QueryResult, interpretation: str) -> str:
        """Format query result as readable text."""
        lines = [f"Query: {interpretation}", ""]
        
        if result.answer:
            lines.append(result.answer)
            lines.append("")
        
        if result.statistics:
            lines.append("Statistics:")
            for key, value in result.statistics.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.2f}")
                else:
                    lines.append(f"  {key}: {value}")
            lines.append("")
        
        if result.matches:
            lines.append(f"Matches ({len(result.matches)} total):")
            for i, match in enumerate(result.matches[:10]):
                time_str = f"t={match.time}"
                if match.end_time:
                    time_str += f" to t={match.end_time}"
                signals_str = ", ".join(f"{k}={v}" for k, v in match.signals.items())
                lines.append(f"  {i+1}. {time_str}: {signals_str}")
            
            if len(result.matches) > 10:
                lines.append(f"  ... and {len(result.matches) - 10} more matches")
        
        return "\n".join(lines)
    
    def export_gtkwave_savefile(self, filepath: str, 
                                 signals: Optional[List[str]] = None,
                                 markers: Optional[List[int]] = None) -> str:
        """
        Export a GTKWave save file for viewing results.
        
        Args:
            filepath: Output .gtkw file path
            signals: Signals to include (default: all from last query)
            markers: Time markers to add
        """
        from waveformgpt.gtkwave import generate_savefile
        
        if not self.parser:
            return "No VCD loaded"
        
        # Get signals from last query result
        if signals is None and self.history:
            for msg in reversed(self.history):
                if msg.result and msg.result.matches:
                    signals = list(msg.result.matches[0].signals.keys())
                    break
        
        if signals is None:
            signals = [s.full_name for s in list(self.parser.header.signals.values())[:10]]
        
        # Get markers from last result
        if markers is None and self.history:
            for msg in reversed(self.history):
                if msg.result and msg.result.matches:
                    markers = [m.time for m in msg.result.matches[:20]]
                    break
        
        return generate_savefile(
            vcd_file=self.vcd_file,
            output_path=filepath,
            signals=signals,
            markers=markers or []
        )
