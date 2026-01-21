"""
WaveformGPT - Query simulation waveforms in natural language.

A powerful toolkit for analyzing digital waveforms using natural language
queries, temporal assertions, protocol checking, and visualization.
"""

from waveformgpt.chat import WaveformChat
from waveformgpt.vcd_parser import VCDParser, Signal, ValueChange
from waveformgpt.query_engine import QueryEngine, QueryType, Query, QueryResult
from waveformgpt.analyzer import (
    WaveformAnalyzer,
    TimingStats,
    SignalActivity,
    GlitchInfo,
    ProtocolViolation,
)
from waveformgpt.protocols import (
    ProtocolChecker,
    AXI4Checker,
    WishboneChecker,
    SPIChecker,
    auto_detect_protocol,
    create_checker,
)
from waveformgpt.assertions import (
    AssertionChecker,
    Assertion,
    AssertionResult,
    AssertionCheckResult,
    check_assertions_from_file,
)
from waveformgpt.compare import (
    WaveformComparator,
    ComparisonResult,
    ComparisonOptions,
    DifferenceType,
)
from waveformgpt.visualize import (
    ASCIIWaveform,
    WaveformStyle,
    render_to_html,
    render_to_matplotlib,
)
from waveformgpt.export import (
    export_to_csv,
    export_to_json,
    export_to_markdown,
    export_to_wavedrom,
    export_to_systemverilog,
    export_to_cocotb,
)

# Optional LLM imports (require API keys)
try:
    from waveformgpt.llm_engine import (
        WaveformLLM,
        LLMBackend,
        OpenAIBackend,
        AnthropicBackend,
        OllamaBackend,
        get_llm_backend,
        LLMResponse,
    )
    _HAS_LLM = True
except ImportError:
    _HAS_LLM = False

# Optional Voice imports (require pyaudio)
try:
    from waveformgpt.voice import (
        VoiceChat,
        VoiceConfig,
        SpeechToText,
        TextToSpeech,
        AudioRecorder,
        start_voice_session,
    )
    _HAS_VOICE = True
except ImportError:
    _HAS_VOICE = False

# Live waveform streaming
from waveformgpt.live import (
    LiveWaveformAnalyzer,
    LiveWaveformBuffer,
    VCDFileWatcher,
    FIFOSource,
    WebSocketSource,
    create_live_session,
)

__version__ = "0.4.0"
__all__ = [
    # Core
    "WaveformChat",
    "VCDParser",
    "Signal",
    "ValueChange",
    "QueryEngine",
    "QueryType",
    "Query",
    "QueryResult",
    # Analysis
    "WaveformAnalyzer",
    "TimingStats",
    "SignalActivity",
    "GlitchInfo",
    "ProtocolViolation",
    # Protocols
    "ProtocolChecker",
    "AXI4Checker",
    "WishboneChecker",
    "SPIChecker",
    "auto_detect_protocol",
    "create_checker",
    # Assertions
    "AssertionChecker",
    "Assertion",
    "AssertionResult",
    "AssertionCheckResult",
    "check_assertions_from_file",
    # Comparison
    "WaveformComparator",
    "ComparisonResult",
    "ComparisonOptions",
    "DifferenceType",
    # Visualization
    "ASCIIWaveform",
    "WaveformStyle",
    "render_to_html",
    "render_to_matplotlib",
    # Export
    "export_to_csv",
    "export_to_json",
    "export_to_markdown",
    "export_to_wavedrom",
    "export_to_systemverilog",
    "export_to_cocotb",
    # LLM (optional)
    "WaveformLLM",
    "LLMBackend",
    "OpenAIBackend",
    "AnthropicBackend", 
    "OllamaBackend",
    "get_llm_backend",
    "LLMResponse",
    # Voice (optional)
    "VoiceChat",
    "VoiceConfig",
    "SpeechToText",
    "TextToSpeech",
    "AudioRecorder",
    "start_voice_session",
    # Live Streaming
    "LiveWaveformAnalyzer",
    "LiveWaveformBuffer",
    "VCDFileWatcher",
    "FIFOSource",
    "WebSocketSource",
    "create_live_session",
]
