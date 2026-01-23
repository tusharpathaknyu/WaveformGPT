"""
WaveformGPT REST API Server

A production-ready FastAPI server for waveform analysis, classification, and optimization.

Features:
- /analyze: DSP analysis with measurements and diagnosis
- /classify: CNN-based problem classification
- /optimize: Bayesian optimization for component tuning
- /simulate: Circuit simulation with various topologies
- /extract: Extract waveform from oscilloscope images
- /ws/stream: WebSocket for real-time analysis

Run with: uvicorn waveformgpt.api.server:app --reload
"""

import io
import base64
import time
import hashlib
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import json

import numpy as np

# FastAPI imports
try:
    from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    print("FastAPI not installed. Run: pip install fastapi uvicorn python-multipart")

# Our modules
try:
    from waveformgpt.waveformgpt_v2 import WaveformGPT
    from waveformgpt.waveform_cnn import WaveformClassifier, WaveformClass, SyntheticDataGenerator
    from waveformgpt.circuit_optimizer import DSPAnalyzer, RuleBasedDiagnostic, WaveformFeatures
    from waveformgpt.spice_simulator import CircuitSimulator, format_component
except ImportError:
    from waveformgpt_v2 import WaveformGPT
    from waveform_cnn import WaveformClassifier, WaveformClass, SyntheticDataGenerator
    from circuit_optimizer import DSPAnalyzer, RuleBasedDiagnostic, WaveformFeatures
    from spice_simulator import CircuitSimulator, format_component


# =============================================================================
# Pydantic Models for Request/Response
# =============================================================================

class WaveformInput(BaseModel):
    """Input waveform data"""
    samples: List[float] = Field(..., description="Array of voltage samples")
    sample_rate: float = Field(1e6, description="Sample rate in Hz")
    unit: str = Field("V", description="Voltage unit (V, mV, uV)")
    
class AnalysisResponse(BaseModel):
    """Response from /analyze endpoint"""
    measurements: Dict[str, Any]
    diagnosis: str
    problems: List[str]
    fixes: List[Dict[str, Any]]
    processing_time_ms: float

class ClassificationResponse(BaseModel):
    """Response from /classify endpoint"""
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]
    diagnosis: str
    processing_time_ms: float

class OptimizationRequest(BaseModel):
    """Request for /optimize endpoint"""
    target_waveform: List[float] = Field(..., description="Target waveform to match")
    circuit_type: str = Field("rlc", description="Circuit topology: rc, rl, rlc")
    iterations: int = Field(100, description="Number of optimization iterations")
    sample_rate: float = Field(1e6, description="Sample rate in Hz")

class OptimizationResponse(BaseModel):
    """Response from /optimize endpoint"""
    optimal_components: Dict[str, str]
    raw_values: Dict[str, float]
    mse: float
    simulated_waveform: List[float]
    processing_time_ms: float

class SimulationRequest(BaseModel):
    """Request for /simulate endpoint"""
    circuit_type: str = Field("rlc", description="Circuit type: rc, rl, rlc, buck")
    components: Dict[str, float] = Field(..., description="Component values")
    sample_rate: float = Field(1e6)
    duration: float = Field(100e-6)

class SimulationResponse(BaseModel):
    """Response from /simulate endpoint"""
    waveform: List[float]
    time_points: List[float]
    circuit_type: str
    components: Dict[str, str]
    processing_time_ms: float

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    cnn_loaded: bool
    gpu_available: bool
    uptime_seconds: float


# =============================================================================
# In-Memory Cache
# =============================================================================

class WaveformCache:
    """Simple in-memory cache with LRU eviction"""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Dict] = {}
        self.max_size = max_size
        self.access_order: List[str] = []
    
    def _hash_waveform(self, samples: List[float]) -> str:
        """Create hash of waveform for caching"""
        data = np.array(samples).tobytes()
        return hashlib.md5(data).hexdigest()[:16]
    
    def get(self, samples: List[float]) -> Optional[Dict]:
        """Get cached result"""
        key = self._hash_waveform(samples)
        if key in self.cache:
            # Move to end (most recent)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def set(self, samples: List[float], result: Dict):
        """Cache a result"""
        key = self._hash_waveform(samples)
        
        # Evict if full
        while len(self.cache) >= self.max_size:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = result
        self.access_order.append(key)
    
    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        self.access_order.clear()


# =============================================================================
# API Application
# =============================================================================

if HAS_FASTAPI:
    
    # Create app
    app = FastAPI(
        title="WaveformGPT API",
        description="AI-powered waveform analysis, classification, and circuit optimization",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Global state
    _start_time = time.time()
    _gpt: Optional[WaveformGPT] = None
    _classifier: Optional[WaveformClassifier] = None
    _cache = WaveformCache()
    
    
    def get_gpt() -> WaveformGPT:
        """Lazy-load WaveformGPT instance"""
        global _gpt
        if _gpt is None:
            _gpt = WaveformGPT()
        return _gpt
    
    def get_classifier() -> WaveformClassifier:
        """Lazy-load CNN classifier"""
        global _classifier
        if _classifier is None:
            _classifier = WaveformClassifier()
        return _classifier
    
    
    # =========================================================================
    # Endpoints
    # =========================================================================
    
    @app.get("/", response_model=HealthResponse)
    async def root():
        """Health check and API info"""
        classifier = get_classifier()
        
        # Check for GPU
        gpu_available = False
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except:
            pass
        
        return HealthResponse(
            status="healthy",
            version="2.0.0",
            cnn_loaded=classifier.use_cnn if hasattr(classifier, 'use_cnn') else False,
            gpu_available=gpu_available,
            uptime_seconds=time.time() - _start_time
        )
    
    
    @app.post("/analyze", response_model=AnalysisResponse)
    async def analyze_waveform(data: WaveformInput):
        """
        Analyze a waveform and get measurements + diagnosis.
        
        Returns rise time, overshoot, THD, ringing, and recommended fixes.
        """
        start = time.time()
        
        # Check cache
        cached = _cache.get(data.samples)
        if cached:
            cached['processing_time_ms'] = (time.time() - start) * 1000
            cached['cached'] = True
            return cached
        
        # Convert to numpy
        waveform = np.array(data.samples, dtype=np.float32)
        
        # Scale if not in volts
        if data.unit == "mV":
            waveform /= 1000
        elif data.unit == "uV":
            waveform /= 1e6
        
        # Analyze
        gpt = get_gpt()
        gpt.sample_rate = data.sample_rate
        gpt.dsp.sample_rate = data.sample_rate
        
        result = gpt.analyze(waveform)
        
        # Build response
        response = {
            "measurements": {
                "vpp": result['features'].vpp,
                "vmax": result['features'].vmax,
                "vmin": result['features'].vmin,
                "vrms": result['features'].vrms,
                "dc_offset": result['features'].dc_offset,
                "rise_time_us": result['features'].rise_time * 1e6,
                "fall_time_us": result['features'].fall_time * 1e6,
                "overshoot_pct": result['features'].overshoot_pct,
                "undershoot_pct": result['features'].undershoot_pct,
                "ringing_freq_hz": result['features'].ringing_freq,
                "settling_time_us": result['features'].settling_time * 1e6,
                "frequency_hz": result['features'].frequency,
                "duty_cycle_pct": result['features'].duty_cycle * 100,
                "noise_rms": result['features'].noise_rms,
                "thd_pct": result['features'].thd_pct,
            },
            "diagnosis": result['summary'],
            "problems": [p.name for p in result['problems']],
            "fixes": [
                {
                    "problem": f.problem.name,
                    "severity": f.severity,
                    "action": f.action,
                    "suggested_value": f.suggested_value,
                    "explanation": f.fix,
                }
                for f in result['fixes']
            ],
            "processing_time_ms": (time.time() - start) * 1000
        }
        
        # Cache result
        _cache.set(data.samples, response)
        
        return response
    
    
    @app.post("/classify", response_model=ClassificationResponse)
    async def classify_waveform(data: WaveformInput):
        """
        Classify waveform problem type using CNN.
        
        Classes: NORMAL, OVERSHOOT, RINGING, NOISE, CLIPPING, SLOW_RISE, DISTORTION, DC_OFFSET
        """
        start = time.time()
        
        waveform = np.array(data.samples, dtype=np.float32)
        
        classifier = get_classifier()
        result = classifier.classify(waveform)
        
        return ClassificationResponse(
            predicted_class=result.predicted_class.name,
            confidence=result.confidence,
            all_probabilities=result.all_probabilities,
            diagnosis=classifier.get_diagnosis(waveform),
            processing_time_ms=(time.time() - start) * 1000
        )
    
    
    @app.post("/optimize", response_model=OptimizationResponse)
    async def optimize_circuit(request: OptimizationRequest):
        """
        Find optimal component values to match a target waveform.
        
        Uses Bayesian optimization (requires scikit-optimize).
        """
        start = time.time()
        
        target = np.array(request.target_waveform, dtype=np.float32)
        
        gpt = get_gpt()
        gpt.sample_rate = request.sample_rate
        
        try:
            result = gpt.optimize_to_target(
                target,
                circuit_type=request.circuit_type,
                n_iterations=request.iterations
            )
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail="Optimization requires scikit-optimize. Install with: pip install scikit-optimize"
            )
        
        # Simulate with optimal values
        simulated = gpt.simulate_circuit(
            circuit_type=request.circuit_type,
            **{k: v for k, v in result.items() if k in ['R', 'L', 'C']}
        )
        
        # Format component values
        formatted = {}
        for key in ['R', 'L', 'C']:
            if key in result:
                unit = {'R': 'Ω', 'L': 'H', 'C': 'F'}[key]
                formatted[key] = format_component(result[key], unit)
        
        return OptimizationResponse(
            optimal_components=formatted,
            raw_values={k: v for k, v in result.items() if k in ['R', 'L', 'C']},
            mse=result['mse'],
            simulated_waveform=simulated.tolist(),
            processing_time_ms=(time.time() - start) * 1000
        )
    
    
    @app.post("/simulate", response_model=SimulationResponse)
    async def simulate_circuit(request: SimulationRequest):
        """
        Simulate a circuit with given components.
        
        Supported types: rc, rl, rlc, buck
        """
        start = time.time()
        
        simulator = CircuitSimulator(
            sample_rate=request.sample_rate,
            duration=request.duration
        )
        
        # Run simulation
        try:
            if request.circuit_type == "rc":
                waveform = simulator.simulate_rc(
                    R=request.components.get("R", 1000),
                    C=request.components.get("C", 1e-9)
                )
            elif request.circuit_type == "rl":
                waveform = simulator.simulate_rl(
                    R=request.components.get("R", 100),
                    L=request.components.get("L", 10e-6)
                )
            elif request.circuit_type == "rlc":
                waveform = simulator.simulate_rlc(
                    R=request.components.get("R", 100),
                    L=request.components.get("L", 10e-6),
                    C=request.components.get("C", 1e-9)
                )
            else:
                raise HTTPException(400, f"Unknown circuit type: {request.circuit_type}")
        except Exception as e:
            raise HTTPException(500, f"Simulation error: {str(e)}")
        
        # Format components
        formatted = {}
        for key, value in request.components.items():
            unit = {'R': 'Ω', 'L': 'H', 'C': 'F'}.get(key, '')
            formatted[key] = format_component(value, unit) if unit else str(value)
        
        t = np.linspace(0, request.duration, len(waveform))
        
        return SimulationResponse(
            waveform=waveform.tolist(),
            time_points=t.tolist(),
            circuit_type=request.circuit_type,
            components=formatted,
            processing_time_ms=(time.time() - start) * 1000
        )
    
    
    @app.post("/extract")
    async def extract_from_image(
        file: UploadFile = File(...),
        channel: Optional[str] = None,
        volts_per_div: Optional[float] = None,
        time_per_div: Optional[float] = None
    ):
        """
        Extract waveform data from an oscilloscope screenshot.
        
        Supports common scope colors: yellow, cyan, magenta, green
        """
        start = time.time()
        
        gpt = get_gpt()
        if gpt.image_extractor is None:
            raise HTTPException(
                status_code=503,
                detail="Image extraction requires OpenCV. Install with: pip install opencv-python"
            )
        
        # Read image
        contents = await file.read()
        
        # Save temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        
        try:
            result = gpt.analyze_image(
                tmp_path,
                channel=channel,
                volts_per_div=volts_per_div,
                time_per_div=time_per_div
            )
            
            return {
                "waveform": result['waveform'].tolist(),
                "measurements": {
                    "vpp": result['features'].vpp,
                    "rise_time_us": result['features'].rise_time * 1e6,
                    "overshoot_pct": result['features'].overshoot_pct,
                },
                "diagnosis": result['summary'],
                "processing_time_ms": (time.time() - start) * 1000
            }
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    
    @app.post("/batch/analyze")
    async def batch_analyze(waveforms: List[WaveformInput]):
        """Analyze multiple waveforms in batch"""
        results = []
        for wf in waveforms:
            result = await analyze_waveform(wf)
            results.append(result)
        return {"results": results, "count": len(results)}
    
    
    @app.post("/compare")
    async def compare_waveforms(
        waveform_a: WaveformInput,
        waveform_b: WaveformInput
    ):
        """Compare two waveforms and return differences"""
        start = time.time()
        
        a = np.array(waveform_a.samples)
        b = np.array(waveform_b.samples)
        
        # Resample if different lengths
        if len(a) != len(b):
            target_len = max(len(a), len(b))
            a = np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(a)), a)
            b = np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(b)), b)
        
        # Compute metrics
        mse = float(np.mean((a - b) ** 2))
        mae = float(np.mean(np.abs(a - b)))
        correlation = float(np.corrcoef(a, b)[0, 1])
        max_diff = float(np.max(np.abs(a - b)))
        
        # Analyze both
        gpt = get_gpt()
        analysis_a = gpt.analyze(a)
        analysis_b = gpt.analyze(b)
        
        return {
            "similarity": {
                "mse": mse,
                "mae": mae,
                "correlation": correlation,
                "max_difference": max_diff,
                "similar": correlation > 0.95 and mse < 0.1
            },
            "waveform_a": {
                "problems": [p.name for p in analysis_a['problems']],
                "summary": analysis_a['summary']
            },
            "waveform_b": {
                "problems": [p.name for p in analysis_b['problems']],
                "summary": analysis_b['summary']
            },
            "processing_time_ms": (time.time() - start) * 1000
        }
    
    
    @app.delete("/cache")
    async def clear_cache():
        """Clear the analysis cache"""
        _cache.clear()
        return {"status": "cleared", "message": "Cache cleared successfully"}
    
    
    @app.get("/stats")
    async def get_stats():
        """Get API statistics"""
        return {
            "cache_size": len(_cache.cache),
            "uptime_seconds": time.time() - _start_time,
            "cnn_model": "loaded" if get_classifier().use_cnn else "numpy-fallback"
        }
    
    
    # =========================================================================
    # WebSocket for Real-time Streaming
    # =========================================================================
    
    @app.websocket("/ws/stream")
    async def websocket_stream(websocket: WebSocket):
        """
        WebSocket endpoint for real-time waveform analysis.
        
        Send JSON with 'samples' array, receive analysis results.
        """
        await websocket.accept()
        
        gpt = get_gpt()
        classifier = get_classifier()
        
        try:
            while True:
                # Receive data
                data = await websocket.receive_json()
                
                if "samples" not in data:
                    await websocket.send_json({"error": "Missing 'samples' field"})
                    continue
                
                waveform = np.array(data["samples"], dtype=np.float32)
                
                # Quick classification
                cls_result = classifier.classify(waveform)
                
                # Quick measurements
                features = gpt.dsp.extract_features(waveform)
                
                # Send result
                await websocket.send_json({
                    "classification": cls_result.predicted_class.name,
                    "confidence": cls_result.confidence,
                    "vpp": features.vpp,
                    "overshoot": features.overshoot_pct,
                    "thd": features.thd_pct,
                    "timestamp": time.time()
                })
                
        except WebSocketDisconnect:
            pass


else:
    # Dummy app if FastAPI not installed
    app = None


# =============================================================================
# CLI Entry Point
# =============================================================================

def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the API server"""
    if not HAS_FASTAPI:
        print("FastAPI not installed. Install with:")
        print("  pip install fastapi uvicorn python-multipart")
        return
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    WaveformGPT API v2.0                       ║
╠══════════════════════════════════════════════════════════════╣
║  Endpoints:                                                   ║
║    POST /analyze    - Analyze waveform, get measurements      ║
║    POST /classify   - CNN classification (8 problem types)    ║
║    POST /optimize   - Bayesian component optimization         ║
║    POST /simulate   - Circuit simulation                      ║
║    POST /extract    - Extract from oscilloscope image         ║
║    POST /compare    - Compare two waveforms                   ║
║    WS   /ws/stream  - Real-time streaming analysis            ║
║                                                               ║
║  Docs: http://{host}:{port}/docs                                   ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    uvicorn.run(
        "waveformgpt.api.server:app",
        host=host,
        port=port,
        reload=reload
    )


if __name__ == "__main__":
    start_server(reload=True)
