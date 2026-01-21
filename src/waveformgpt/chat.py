"""
Chat Interface for WaveformGPT.

Provides an interactive conversational interface for waveform queries.
Supports both pattern-based parsing and LLM-powered natural language understanding.
"""

from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import os

from waveformgpt.vcd_parser import VCDParser
from waveformgpt.query_engine import QueryEngine, QueryResult
from waveformgpt.nl_parser import NLParser


@dataclass
class Message:
    """Chat message."""
    role: str  # "user" or "assistant"
    content: str
    result: Optional[QueryResult] = None
    reasoning: Optional[str] = None
    code: Optional[str] = None


class WaveformChat:
    """
    Interactive chat interface for waveform queries.
    
    Supports two modes:
    1. Pattern-based (default): Uses regex patterns - fast, works offline
    2. LLM-powered: Uses AI to understand any question - more flexible
    
    Usage:
        # Pattern-based (works offline)
        chat = WaveformChat("simulation.vcd")
        
        # LLM-powered (requires API key)
        chat = WaveformChat("simulation.vcd", use_llm=True)
        
        response = chat.ask("When does the clock rise?")
        print(response.content)
        
        # Ask anything with LLM mode!
        response = chat.ask("Is there a glitch on the data bus?")
        response = chat.ask("What's the duty cycle of the clock?")
        response = chat.ask("Find any protocol violations")
    """
    
    def __init__(self, vcd_file: Optional[str] = None, 
                 use_llm: bool = False,
                 llm_provider: str = "auto",
                 llm_api_key: Optional[str] = None):
        """
        Initialize chat interface.
        
        Args:
            vcd_file: Path to VCD file to load
            use_llm: Enable LLM-powered queries (requires API key)
            llm_provider: 'openai', 'anthropic', 'ollama', or 'auto'
            llm_api_key: API key (or use environment variables)
        """
        self.vcd_file = vcd_file
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        
        self.parser: Optional[VCDParser] = None
        self.engine: Optional[QueryEngine] = None
        self.nl_parser: Optional[NLParser] = None
        self.llm_engine = None
        
        self.history: List[Message] = []
        
        # Set API key if provided
        if llm_api_key:
            if llm_provider == "openai" or llm_provider == "auto":
                os.environ["OPENAI_API_KEY"] = llm_api_key
            elif llm_provider == "anthropic":
                os.environ["ANTHROPIC_API_KEY"] = llm_api_key
        
        # Initialize LLM engine if enabled
        if use_llm:
            self._init_llm()
        
        if vcd_file:
            self.load_vcd(vcd_file)
    
    def _init_llm(self):
        """Initialize the LLM engine."""
        try:
            from waveformgpt.llm_engine import WaveformLLM, get_llm_backend
            
            backend = get_llm_backend(self.llm_provider)
            self.llm_engine = WaveformLLM(backend)
        except Exception as e:
            print(f"Warning: Could not initialize LLM: {e}")
            print("Falling back to pattern-based parsing.")
            self.use_llm = False
    
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
            
            # Configure LLM context if enabled
            if self.llm_engine:
                # Get sample data for context
                sample_data = {}
                for sig_name in signals[:10]:  # First 10 signals
                    try:
                        values = self.parser.get_signal_values(sig_name)[:10]
                        sample_data[sig_name] = values
                    except:
                        pass
                
                self.llm_engine.set_waveform_context(
                    signals=signals,
                    time_range=time_range,
                    time_unit=self.parser.header.timescale_unit or "ns",
                    sample_data=sample_data
                )
            
            msg = (
                f"✓ Loaded VCD file: {Path(filepath).name}\n"
                f"  Signals: {len(signals)}\n"
                f"  Time range: {time_range[0]} to {time_range[1]} {self.parser.header.timescale_unit}\n"
                f"  Mode: {'🤖 LLM-powered' if self.use_llm else '⚡ Pattern-based'}\n"
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
        
        # Check if we should use LLM:
        # 1. Pattern matching returned no query
        # 2. Low confidence from pattern matcher
        # 3. User explicitly enabled LLM and question seems complex
        use_llm_for_this = False
        
        if parsed.query is None:
            use_llm_for_this = True
        elif parsed.confidence < 0.5:
            use_llm_for_this = True
        elif self.use_llm and self._is_complex_question(question):
            use_llm_for_this = True
        
        if use_llm_for_this and self.use_llm and self.llm_engine:
            return self._ask_llm(question)
        
        if parsed.query is None:
            response = Message(
                role="assistant",
                content=f"I couldn't understand that query. {parsed.interpretation}\n\n"
                        "💡 Tip: Enable LLM mode for more flexible queries:\n"
                        "   chat = WaveformChat('file.vcd', use_llm=True)\n\n"
                        "Pattern-based examples:\n"
                        "- When does clk rise?\n"
                        "- How many times does reset fall?\n"
                        "- Show me data_valid followed by data_ready"
            )
            self.history.append(response)
            return response
        
        # Execute query
        try:
            result = self.engine.execute(parsed.query)
            
            # If result is empty/failed and LLM is available, try LLM
            if self.use_llm and self.llm_engine:
                if not result.matches and not result.statistics:
                    return self._ask_llm(question)
            
            content = self._format_result(result, parsed.interpretation)
            
            response = Message(
                role="assistant",
                content=content,
                result=result
            )
            
        except Exception as e:
            # On error, try LLM if available
            if self.use_llm and self.llm_engine:
                return self._ask_llm(question)
            
            response = Message(
                role="assistant",
                content=f"Error executing query: {str(e)}"
            )
        
        self.history.append(response)
        return response
    
    def _is_complex_question(self, question: str) -> bool:
        """
        Detect if a question is too complex for pattern matching.
        These questions should go directly to the LLM.
        """
        q = question.lower()
        
        # Keywords that indicate complex analysis needs
        complex_keywords = [
            'describe', 'explain', 'summarize', 'analyze', 'why',
            'what is happening', 'what does', 'how does',
            'duty cycle', 'frequency', 'period',
            'state machine', 'fsm', 'behavior',
            'glitch', 'hazard', 'race condition',
            'protocol', 'violation', 'error',
            'relationship', 'correlation', 'cause',
            'debug', 'diagnose', 'troubleshoot',
            'compare', 'difference', 'similar',
            'optimize', 'improve', 'suggest',
        ]
        
        for keyword in complex_keywords:
            if keyword in q:
                return True
        
        return False
    
    def _ask_llm(self, question: str) -> Message:
        """
        Use LLM to answer a question that pattern matching couldn't handle.
        """
        try:
            # Create a data provider function for code execution
            def get_signal_data(signal_name: str) -> List[Tuple[int, str]]:
                return self.parser.get_signal_values(signal_name)
            
            # Query the LLM
            llm_response = self.llm_engine.query(
                question=question,
                signal_data_provider=get_signal_data
            )
            
            # Build response message
            content = f"🤖 {llm_response.answer}"
            
            if llm_response.reasoning:
                content += f"\n\n📝 Reasoning: {llm_response.reasoning}"
            
            response = Message(
                role="assistant",
                content=content,
                reasoning=llm_response.reasoning,
                code=llm_response.code
            )
            
        except Exception as e:
            response = Message(
                role="assistant",
                content=f"LLM query failed: {str(e)}\n\nFalling back to help message.\n"
                        "Try simpler pattern-based questions or check your API key."
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
