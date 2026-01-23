"""
WaveformGPT - Query simulation waveforms in natural language.

A powerful toolkit for analyzing digital waveforms using natural language
queries, temporal assertions, protocol checking, and visualization.

v2.0 adds:
- Oscilloscope image extraction
- DSP-based waveform analysis
- Circuit optimization via Bayesian optimization
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

# Vision-based analysis (LLM sees waveforms as images)
from waveformgpt.vision import (
    WaveformVision,
    VisionAnalysis,
    LiveVisionMonitor,
)

# Real-world integrations
from waveformgpt.integrations import (
    # Simulation runners
    VerilatorRunner,
    IcarusRunner,
    CocotbRunner,
    SimulationResult,
    # CI/CD
    WaveformCI,
    CICheckResult,
    # GTKWave
    GTKWaveIntegration,
    # Notifications
    SlackNotifier,
    GitHubIntegration,
    # Reports
    ReportGenerator,
    # Convenience
    run_simulation_and_analyze,
    ci_check_waveform,
)

# Regression testing
from waveformgpt.regression import (
    WaveformRegression,
    RegressionResult,
    SignalDiff,
    CoverageAnalyzer,
    CoverageResult,
    GoldenModel,
    TestSuite,
)

# Debug utilities
from waveformgpt.debug import (
    WaveformDebugger,
    DebugSession,
    DebugFinding,
    StateMachineAnalyzer,
)

# WaveformBuddy - Hardware debugging companion
try:
    from waveformgpt.buddy import (
        WaveformBuddy,
        BuddyContext,
        Capture,
        CaptureType,
        ESP32Bridge,
    )
    _HAS_BUDDY = True
except ImportError:
    _HAS_BUDDY = False

# Circuit Optimizer v2.0 (DSP + Bayesian Optimization)
try:
    from waveformgpt.circuit_optimizer import (
        CircuitOptimizer,
        DSPAnalyzer,
        RuleBasedDiagnostic,
        WaveformFeatures,
        CircuitFix,
        WaveformProblem,
    )
    from waveformgpt.spice_simulator import (
        CircuitSimulator,
        AnalyticalSimulator,
        format_component,
    )
    from waveformgpt.waveformgpt_v2 import WaveformGPT as WaveformGPTv2
    _HAS_CIRCUIT_OPTIMIZER = True
except ImportError:
    _HAS_CIRCUIT_OPTIMIZER = False

# Image Extractor (requires OpenCV)
try:
    from waveformgpt.image_extractor import (
        WaveformImageExtractor,
        QuickExtractor,
        ExtractedWaveform,
    )
    _HAS_IMAGE_EXTRACTOR = True
except ImportError:
    _HAS_IMAGE_EXTRACTOR = False

# CNN Classifier (requires numpy, optional PyTorch)
try:
    from waveformgpt.waveform_cnn import (
        WaveformClassifier,
        WaveformClass,
        ClassificationResult,
        NumpyWaveformClassifier,
        SyntheticDataGenerator,
    )
    _HAS_CNN = True
except ImportError:
    _HAS_CNN = False

# Enhanced CNN v2.1 (ResNet + Attention)
try:
    from waveformgpt.enhanced_cnn import (
        EnhancedWaveformCNN,
        EnhancedWaveformClassifier,
        WaveformAugmentor,
        ResidualBlock,
        SelfAttention1D,
        MultiScaleConv,
    )
    _HAS_ENHANCED_CNN = True
except ImportError:
    _HAS_ENHANCED_CNN = False

# Database storage v2.1
try:
    from waveformgpt.database import (
        WaveformDatabase,
        WaveformRecord,
        AnalysisRecord,
    )
    _HAS_DATABASE = True
except ImportError:
    _HAS_DATABASE = False

# Local LLM v2.1 (Ollama integration)
try:
    from waveformgpt.local_llm import (
        WaveformLLM as LocalWaveformLLM,
        OllamaClient,
        RuleBasedExplainer,
    )
    _HAS_LOCAL_LLM = True
except ImportError:
    _HAS_LOCAL_LLM = False

# Advanced Analysis v2.1 (frequency, anomaly, trend)
try:
    from waveformgpt.advanced_analysis import (
        AdvancedAnalyzer,
        FrequencyAnalyzer,
        FrequencyAnalysis,
        SpectrogramResult,
        AnomalyDetector,
        AnomalyResult,
        TrendAnalyzer,
        TrendAnalysis,
        WaveletAnalyzer,
        WaveletDecomposition,
        PatternMatcher,
        StatisticalAnalysis,
    )
    _HAS_ADVANCED = True
except ImportError:
    _HAS_ADVANCED = False

# Data Pipeline v2.2 (collection, labeling, dataset)
try:
    from waveformgpt.data_pipeline import (
        DataPipeline,
        DatasetManager,
        WaveformSample,
        ProblemLabel,
        ConfidenceLevel,
        WaveformSource,
        ESP32Collector,
        ActiveLearner,
        DatasetStats,
    )
    _HAS_DATA_PIPELINE = True
except ImportError:
    _HAS_DATA_PIPELINE = False

# Community Dataset v2.2 (shareable, versioned)
try:
    from waveformgpt.community_dataset import (
        CommunityDataset,
        CommunityDatasetBuilder,
        ContributionManager,
        BenchmarkRunner,
    )
    _HAS_COMMUNITY = True
except ImportError:
    _HAS_COMMUNITY = False

# Oscilloscope Integration v2.2 (Rigol, Tektronix, etc.)
try:
    from waveformgpt.oscilloscope import (
        Oscilloscope,
        RigolOscilloscope,
        TektronixOscilloscope,
        WaveformCapture,
        OscilloscopeDataCollector,
    )
    _HAS_OSCILLOSCOPE = True
except ImportError:
    _HAS_OSCILLOSCOPE = False

__version__ = "2.2.0"
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
    # Vision Analysis
    "WaveformVision",
    "VisionAnalysis", 
    "LiveVisionMonitor",
    # Integrations
    "VerilatorRunner",
    "IcarusRunner",
    "CocotbRunner",
    "SimulationResult",
    "WaveformCI",
    "CICheckResult",
    "GTKWaveIntegration",
    "SlackNotifier",
    "GitHubIntegration",
    "ReportGenerator",
    "run_simulation_and_analyze",
    "ci_check_waveform",
    # Regression
    "WaveformRegression",
    "RegressionResult",
    "SignalDiff",
    "CoverageAnalyzer",
    "CoverageResult",
    "GoldenModel",
    "TestSuite",
    # Debug
    "WaveformDebugger",
    "DebugSession",
    "DebugFinding",
    "StateMachineAnalyzer",
    # WaveformBuddy Hardware Companion
    "WaveformBuddy",
    "BuddyContext",
    "Capture",
    "CaptureType",
    "ESP32Bridge",
    # v2.0 Circuit Optimizer
    "CircuitOptimizer",
    "DSPAnalyzer",
    "RuleBasedDiagnostic",
    "WaveformFeatures",
    "CircuitFix",
    "WaveformProblem",
    "CircuitSimulator",
    "AnalyticalSimulator",
    "format_component",
    "WaveformGPTv2",
    # v2.0 Image Extraction
    "WaveformImageExtractor",
    "QuickExtractor",
    "ExtractedWaveform",
    # v2.0 CNN Classifier
    "WaveformClassifier",
    "WaveformClass",
    "ClassificationResult",
    "NumpyWaveformClassifier",
    "SyntheticDataGenerator",
    # v2.1 Enhanced CNN
    "EnhancedWaveformCNN",
    "EnhancedWaveformClassifier",
    "WaveformAugmentor",
    "ResidualBlock",
    "SelfAttention1D",
    "MultiScaleConv",
    # v2.1 Database
    "WaveformDatabase",
    "WaveformRecord",
    "AnalysisRecord",
    # v2.1 Local LLM
    "LocalWaveformLLM",
    "OllamaClient",
    "RuleBasedExplainer",
    # v2.1 Advanced Analysis
    "AdvancedAnalyzer",
    "FrequencyAnalyzer",
    "FrequencyAnalysis",
    "SpectrogramResult",
    "AnomalyDetector",
    "AnomalyResult",
    "TrendAnalyzer",
    "TrendAnalysis",
    "WaveletAnalyzer",
    "WaveletDecomposition",
    "PatternMatcher",
    "StatisticalAnalysis",
    # v2.2 Data Pipeline
    "DataPipeline",
    "DatasetManager",
    "WaveformSample",
    "ProblemLabel",
    "ConfidenceLevel",
    "WaveformSource",
    "ESP32Collector",
    "ActiveLearner",
    "DatasetStats",
    # v2.2 Community Dataset
    "CommunityDataset",
    "CommunityDatasetBuilder",
    "ContributionManager",
    "BenchmarkRunner",
    # v2.2 Oscilloscope Integration
    "Oscilloscope",
    "RigolOscilloscope",
    "TektronixOscilloscope",
    "WaveformCapture",
    "OscilloscopeDataCollector",
]
