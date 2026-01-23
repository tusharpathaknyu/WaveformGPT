"""
WaveformGPT Local LLM Integration

Uses Ollama for natural language explanations without API costs.
Falls back to rule-based templates if Ollama not available.

Supported models:
- llama3.2 (3B, fastest)
- llama3.1 (8B, better quality)
- mistral (7B, good balance)
- phi3 (3.8B, efficient)
- qwen2 (7B, good for technical)

Install Ollama: curl -fsSL https://ollama.com/install.sh | sh
Pull a model: ollama pull llama3.2
"""

import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import re

# HTTP client
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    try:
        import requests as httpx
        HAS_HTTPX = True
    except ImportError:
        HAS_HTTPX = False


# =============================================================================
# LLM Response Types
# =============================================================================

@dataclass
class LLMExplanation:
    """Natural language explanation from LLM"""
    summary: str
    detailed_explanation: str
    recommended_actions: List[str]
    confidence: float
    model_used: str
    tokens_used: int
    latency_ms: float


@dataclass
class ConversationMessage:
    """Chat message"""
    role: str  # 'user', 'assistant', 'system'
    content: str


# =============================================================================
# Prompt Templates
# =============================================================================

SYSTEM_PROMPT = """You are WaveformGPT, an expert analog circuit and signal analysis assistant.
You analyze waveform measurements and provide clear, actionable advice for fixing circuit problems.

Your expertise includes:
- Oscilloscope waveform interpretation
- Circuit debugging (overshoot, ringing, noise, distortion)
- Component selection (R, L, C values)
- Signal integrity issues
- Power supply design
- EMI/EMC troubleshooting

Always be concise and practical. Provide specific component values when possible.
Format responses with clear sections for Summary, Analysis, and Recommended Actions."""


ANALYSIS_PROMPT = """Analyze this waveform measurement:

Measurements:
{measurements}

Detected Problems:
{problems}

Current Diagnosis:
{diagnosis}

Please provide:
1. A brief summary (1-2 sentences)
2. Technical analysis of what's causing these issues
3. Specific recommended actions with component values

Be practical and actionable."""


OPTIMIZATION_PROMPT = """I optimized circuit components using Bayesian optimization:

Circuit Type: {circuit_type}
Target Waveform: {target_description}

Optimization Result:
- R = {R}
- L = {L}  
- C = {C}
- MSE = {mse}

Original parameters: R=100Ω, L=10µH, C=1nF

Explain:
1. Why these values work better
2. The physics behind the improvement
3. Any trade-offs to consider"""


CLASSIFICATION_PROMPT = """The CNN classifier detected this waveform problem:

Classification: {predicted_class}
Confidence: {confidence}%
Probabilities: {probabilities}

Explain:
1. What this problem means
2. Common causes in real circuits
3. Quick fixes to try first
4. When to investigate deeper"""


# =============================================================================
# Rule-Based Fallback Explanations
# =============================================================================

class RuleBasedExplainer:
    """
    Template-based explanations when LLM is not available.
    Uses pre-written expert responses for common problems.
    """
    
    EXPLANATIONS = {
        "OVERSHOOT": {
            "summary": "The signal overshoots its target voltage before settling.",
            "explanation": """
Overshoot occurs when the circuit is underdamped - there's not enough resistance 
to dissipate the energy stored in inductors and capacitors quickly enough.

In an RLC circuit, overshoot happens when the damping factor ζ < 1:
ζ = R/(2√(L/C))

Common causes:
1. Parasitic inductance in PCB traces (especially for fast edges)
2. Insufficient gate resistor on MOSFET drivers
3. Long wires acting as inductors
4. Capacitor ESL (equivalent series inductance)
""",
            "actions": [
                "Add a snubber capacitor (10-100nF) in parallel with the load",
                "Add a series gate resistor (10-100Ω) to slow switching edges",
                "Reduce trace length and add ground plane",
                "Use lower-ESL capacitors (ceramic instead of electrolytic)"
            ]
        },
        "RINGING": {
            "summary": "The signal oscillates around its final value before settling.",
            "explanation": """
Ringing is sustained oscillation, often caused by reflections on transmission lines 
or resonance between parasitic inductance and capacitance.

Key physics:
- Transmission line reflections occur when impedance isn't matched
- LC tank circuits resonate at f = 1/(2π√LC)
- High-Q circuits ring longer (Q = ωL/R)

Common causes:
1. Unterminated transmission lines
2. Sharp signal edges exciting parasitic LC tanks
3. Long probe ground leads
4. Connector impedance mismatches
""",
            "actions": [
                "Add RC snubber at the source (10Ω + 100pF typical)",
                "Terminate transmission lines with their characteristic impedance",
                "Shorten oscilloscope probe ground lead",
                "Add series resistor to slow edge rate",
                "Use ferrite beads on power supply lines"
            ]
        },
        "NOISE": {
            "summary": "High-frequency noise is corrupting the signal.",
            "explanation": """
Electrical noise comes from many sources - switching power supplies, digital circuits, 
RF interference, and even thermal noise in resistors.

Noise sources:
1. Conducted EMI from switch-mode supplies (100kHz-1MHz typical)
2. Radiated EMI from nearby circuits
3. Ground loops creating 50/60Hz hum
4. Thermal noise (Johnson-Nyquist): V_rms = √(4kTRΔf)

SNR improvement strategies focus on filtering, shielding, and layout.
""",
            "actions": [
                "Add bypass capacitors (100nF ceramic + 10µF electrolytic) at power pins",
                "Improve grounding - use star ground or ground plane",
                "Add LC filter on power input",
                "Shield sensitive circuits with grounded enclosure",
                "Twist and shield signal cables",
                "Use differential signaling for long runs"
            ]
        },
        "CLIPPING": {
            "summary": "The signal is hitting voltage limits and being clipped.",
            "explanation": """
Clipping occurs when the signal tries to exceed the power supply rails or 
an amplifier's output range. The waveform gets "clipped" flat at the limits.

Common causes:
1. Input signal amplitude too high
2. Amplifier gain set too high
3. Power supply voltage too low for signal swing
4. Op-amp hitting rail (especially with single supply)
5. ADC input exceeding reference voltage
""",
            "actions": [
                "Reduce input signal amplitude",
                "Lower amplifier gain",
                "Increase power supply voltage",
                "Use rail-to-rail op-amps",
                "Add input attenuation (voltage divider)",
                "Check and increase ADC reference voltage"
            ]
        },
        "DISTORTION": {
            "summary": "The signal has harmonic distortion, deviating from ideal waveform.",
            "explanation": """
Total Harmonic Distortion (THD) measures how much the signal differs from a pure sine.
High THD means the circuit is adding unwanted harmonics.

THD = √(V2² + V3² + V4² + ...) / V1 × 100%

Common causes:
1. Amplifier operating outside linear region
2. Magnetic core saturation in transformers/inductors
3. Crossover distortion in push-pull amplifiers
4. Nonlinear capacitance (some ceramics, varactors)
5. Clipping (extreme case)
""",
            "actions": [
                "Reduce signal amplitude to stay in linear region",
                "Add negative feedback to linearize amplifier",
                "Check for proper biasing in transistor stages",
                "Use larger core or lower flux in magnetics",
                "Replace nonlinear ceramic capacitors with C0G/NP0"
            ]
        },
        "SLOW_RISE": {
            "summary": "The signal's rise time is too slow for the application.",
            "explanation": """
Slow rise time limits the maximum frequency and can cause timing violations in digital circuits.

Rise time is related to bandwidth: BW ≈ 0.35 / t_rise

For an RC circuit: t_rise = 2.2 × R × C

Common causes:
1. Excessive load capacitance
2. Weak driver (high output impedance)
3. High series resistance
4. Long transmission lines without proper termination
""",
            "actions": [
                "Reduce load capacitance",
                "Use a stronger driver (lower Rout)",
                "Reduce series resistance in signal path",
                "Add a buffer amplifier near the load",
                "Terminate transmission lines properly"
            ]
        },
        "DC_OFFSET": {
            "summary": "There's an unexpected DC offset on the signal.",
            "explanation": """
DC offset is when the signal's average value isn't where expected. 
This can shift the operating point or cause headroom issues.

Common causes:
1. Op-amp input offset voltage
2. Unbalanced voltage dividers
3. Leakage currents charging coupling capacitors
4. Thermocouple effects at dissimilar metal junctions
5. Ground potential differences
""",
            "actions": [
                "Add AC coupling capacitor if DC isn't needed",
                "Use offset null adjustment on op-amp",
                "Check and match resistor values in dividers",
                "Add DC blocking capacitor with proper time constant",
                "Use auto-zero or chopper-stabilized amplifiers"
            ]
        },
        "NORMAL": {
            "summary": "The waveform looks healthy with no significant issues detected.",
            "explanation": """
The signal passes all quality checks:
- Rise/fall times are appropriate
- No significant overshoot or ringing
- Low noise floor
- No clipping
- Minimal distortion

The circuit appears to be functioning as designed.
""",
            "actions": [
                "No immediate action required",
                "Document this as baseline for future comparison",
                "Consider if performance could be further optimized",
                "Verify operation across temperature range if needed"
            ]
        }
    }
    
    def explain(self, problem_type: str, measurements: Optional[Dict] = None) -> LLMExplanation:
        """Generate rule-based explanation"""
        info = self.EXPLANATIONS.get(problem_type, self.EXPLANATIONS["NORMAL"])
        
        # Customize with measurements if available
        explanation = info["explanation"].strip()
        if measurements:
            if "overshoot_pct" in measurements and measurements["overshoot_pct"] > 0:
                explanation += f"\n\nMeasured overshoot: {measurements['overshoot_pct']:.1f}%"
            if "thd_pct" in measurements and measurements["thd_pct"] > 0:
                explanation += f"\nMeasured THD: {measurements['thd_pct']:.1f}%"
        
        return LLMExplanation(
            summary=info["summary"],
            detailed_explanation=explanation,
            recommended_actions=info["actions"],
            confidence=0.95,  # High confidence for known rules
            model_used="rule-based",
            tokens_used=0,
            latency_ms=1.0
        )


# =============================================================================
# Ollama Client
# =============================================================================

class OllamaClient:
    """
    Client for Ollama local LLM server.
    
    Ollama provides a simple API for running LLMs locally.
    Default endpoint: http://localhost:11434
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        timeout: float = 60.0
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._cache: Dict[str, str] = {}
    
    def is_available(self) -> bool:
        """Check if Ollama server is running"""
        if not HAS_HTTPX:
            return False
        
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=2.0)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """List available models"""
        if not HAS_HTTPX:
            return []
        
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except:
            return []
    
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Optional[str]:
        """
        Generate completion from Ollama.
        
        Args:
            prompt: User prompt
            system: System prompt (optional)
            temperature: Creativity (0-1)
            max_tokens: Max response tokens
        
        Returns:
            Generated text or None if failed
        """
        if not HAS_HTTPX:
            return None
        
        # Check cache
        cache_key = hashlib.md5(f"{self.model}:{system}:{prompt}".encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            if system:
                payload["system"] = system
            
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                result = data.get("response", "")
                self._cache[cache_key] = result
                return result
            
            return None
            
        except Exception as e:
            print(f"Ollama error: {e}")
            return None
    
    def chat(
        self,
        messages: List[ConversationMessage],
        temperature: float = 0.7
    ) -> Optional[str]:
        """Chat completion with conversation history"""
        if not HAS_HTTPX:
            return None
        
        try:
            formatted_messages = [
                {"role": m.role, "content": m.content}
                for m in messages
            ]
            
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": formatted_messages,
                    "stream": False,
                    "options": {"temperature": temperature}
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("message", {}).get("content", "")
            
            return None
            
        except Exception as e:
            print(f"Ollama chat error: {e}")
            return None


# =============================================================================
# WaveformGPT LLM Integration
# =============================================================================

class WaveformLLM:
    """
    High-level LLM integration for WaveformGPT.
    
    Uses Ollama when available, falls back to rule-based explanations.
    
    Usage:
        llm = WaveformLLM()
        
        # Check if LLM is available
        if llm.is_available():
            print(f"Using: {llm.model}")
        
        # Get explanation for analysis
        explanation = llm.explain_analysis(measurements, problems)
        print(explanation.summary)
        print(explanation.recommended_actions)
    """
    
    def __init__(
        self,
        model: str = "llama3.2",
        ollama_url: str = "http://localhost:11434",
        use_cache: bool = True
    ):
        self.ollama = OllamaClient(base_url=ollama_url, model=model)
        self.fallback = RuleBasedExplainer()
        self.use_cache = use_cache
        self._llm_available = None
    
    @property
    def is_llm_available(self) -> bool:
        """Check if LLM is available (cached)"""
        if self._llm_available is None:
            self._llm_available = self.ollama.is_available()
        return self._llm_available
    
    @property
    def model(self) -> str:
        """Current model name"""
        if self.is_llm_available:
            return self.ollama.model
        return "rule-based"
    
    def is_available(self) -> bool:
        """Check if LLM (Ollama) is available"""
        return self.is_llm_available
    
    def list_models(self) -> List[str]:
        """List available Ollama models"""
        return self.ollama.list_models()
    
    def explain_analysis(
        self,
        measurements: Dict[str, Any],
        problems: List[str],
        diagnosis: str
    ) -> LLMExplanation:
        """
        Get natural language explanation for waveform analysis.
        
        Args:
            measurements: Dict of measured values
            problems: List of detected problem names
            diagnosis: Rule-based diagnosis text
        
        Returns:
            LLMExplanation with summary, details, and actions
        """
        start = time.time()
        
        # Try LLM first
        if self.is_llm_available:
            # Format measurements
            meas_str = "\n".join([
                f"  - {k}: {v:.3f}" if isinstance(v, float) else f"  - {k}: {v}"
                for k, v in measurements.items()
            ])
            
            prompt = ANALYSIS_PROMPT.format(
                measurements=meas_str,
                problems=", ".join(problems) if problems else "None detected",
                diagnosis=diagnosis
            )
            
            response = self.ollama.generate(prompt, system=SYSTEM_PROMPT)
            
            if response:
                latency = (time.time() - start) * 1000
                
                # Parse response
                parsed = self._parse_llm_response(response)
                
                return LLMExplanation(
                    summary=parsed.get("summary", response[:200]),
                    detailed_explanation=parsed.get("analysis", response),
                    recommended_actions=parsed.get("actions", []),
                    confidence=0.85,
                    model_used=self.ollama.model,
                    tokens_used=len(response.split()),
                    latency_ms=latency
                )
        
        # Fallback to rule-based
        if problems:
            primary_problem = problems[0]
            return self.fallback.explain(primary_problem, measurements)
        
        return self.fallback.explain("NORMAL", measurements)
    
    def explain_classification(
        self,
        predicted_class: str,
        confidence: float,
        probabilities: Dict[str, float]
    ) -> LLMExplanation:
        """Explain CNN classification result"""
        start = time.time()
        
        if self.is_llm_available:
            probs_str = ", ".join([
                f"{k}: {v:.0%}" for k, v in 
                sorted(probabilities.items(), key=lambda x: -x[1])[:3]
            ])
            
            prompt = CLASSIFICATION_PROMPT.format(
                predicted_class=predicted_class,
                confidence=confidence * 100,
                probabilities=probs_str
            )
            
            response = self.ollama.generate(prompt, system=SYSTEM_PROMPT)
            
            if response:
                latency = (time.time() - start) * 1000
                parsed = self._parse_llm_response(response)
                
                return LLMExplanation(
                    summary=parsed.get("summary", f"Detected {predicted_class}"),
                    detailed_explanation=response,
                    recommended_actions=parsed.get("actions", []),
                    confidence=confidence,
                    model_used=self.ollama.model,
                    tokens_used=len(response.split()),
                    latency_ms=latency
                )
        
        # Fallback
        return self.fallback.explain(predicted_class)
    
    def explain_optimization(
        self,
        circuit_type: str,
        optimal_components: Dict[str, float],
        mse: float,
        target_description: str = "critically damped step response"
    ) -> LLMExplanation:
        """Explain optimization result"""
        start = time.time()
        
        if self.is_llm_available:
            prompt = OPTIMIZATION_PROMPT.format(
                circuit_type=circuit_type.upper(),
                target_description=target_description,
                R=f"{optimal_components.get('R', 0):.2f}Ω",
                L=f"{optimal_components.get('L', 0)*1e6:.2f}µH",
                C=f"{optimal_components.get('C', 0)*1e9:.2f}nF",
                mse=f"{mse:.6f}"
            )
            
            response = self.ollama.generate(prompt, system=SYSTEM_PROMPT)
            
            if response:
                latency = (time.time() - start) * 1000
                
                return LLMExplanation(
                    summary=f"Optimized {circuit_type} circuit to MSE={mse:.6f}",
                    detailed_explanation=response,
                    recommended_actions=[
                        f"Use R = {optimal_components.get('R', 0):.1f}Ω",
                        f"Use L = {optimal_components.get('L', 0)*1e6:.2f}µH",
                        f"Use C = {optimal_components.get('C', 0)*1e9:.2f}nF"
                    ],
                    confidence=0.9,
                    model_used=self.ollama.model,
                    tokens_used=len(response.split()),
                    latency_ms=latency
                )
        
        # Fallback
        return LLMExplanation(
            summary=f"Found optimal {circuit_type} values with MSE={mse:.6f}",
            detailed_explanation=f"""
The Bayesian optimization found component values that minimize the 
mean squared error between the simulated and target waveforms.

Optimal values:
- R = {optimal_components.get('R', 0):.2f}Ω
- L = {optimal_components.get('L', 0)*1e6:.2f}µH  
- C = {optimal_components.get('C', 0)*1e9:.2f}nF

The damping factor ζ = R/(2√(L/C)) determines the response character.
""",
            recommended_actions=[
                "Verify with real components (±5% tolerance)",
                "Consider temperature effects on components",
                "Test across input voltage range"
            ],
            confidence=0.85,
            model_used="rule-based",
            tokens_used=0,
            latency_ms=(time.time() - start) * 1000
        )
    
    def chat(self, message: str, history: Optional[List[ConversationMessage]] = None) -> str:
        """
        Chat with the waveform assistant.
        
        Args:
            message: User message
            history: Previous conversation messages
        
        Returns:
            Assistant response
        """
        if history is None:
            history = []
        
        # Add system message if not present
        if not history or history[0].role != "system":
            history.insert(0, ConversationMessage("system", SYSTEM_PROMPT))
        
        # Add user message
        history.append(ConversationMessage("user", message))
        
        if self.is_llm_available:
            response = self.ollama.chat(history)
            if response:
                return response
        
        # Fallback
        return "I'm sorry, the LLM is not available. Please check that Ollama is running with: ollama serve"
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse structured content from LLM response"""
        result = {"summary": "", "analysis": "", "actions": []}
        
        # Try to find sections
        lines = response.strip().split("\n")
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if any(s in line.lower() for s in ["summary", "brief"]):
                current_section = "summary"
                continue
            elif any(s in line.lower() for s in ["analysis", "technical", "explanation"]):
                current_section = "analysis"
                continue
            elif any(s in line.lower() for s in ["action", "recommend", "suggest"]):
                current_section = "actions"
                continue
            
            if current_section == "summary":
                result["summary"] += line + " "
            elif current_section == "analysis":
                result["analysis"] += line + "\n"
            elif current_section == "actions":
                # Extract action items
                if line.startswith(("-", "•", "*", "1", "2", "3", "4", "5")):
                    action = re.sub(r'^[-•*\d.)\s]+', '', line).strip()
                    if action:
                        result["actions"].append(action)
        
        # If no sections found, use whole response
        if not result["summary"]:
            result["summary"] = response[:200]
        if not result["analysis"]:
            result["analysis"] = response
        
        result["summary"] = result["summary"].strip()
        result["analysis"] = result["analysis"].strip()
        
        return result


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("WaveformGPT Local LLM - Demo")
    print("=" * 60)
    
    llm = WaveformLLM()
    
    print(f"\n🤖 LLM Status:")
    if llm.is_available():
        print(f"   ✅ Ollama is running")
        print(f"   Model: {llm.model}")
        print(f"   Available models: {llm.list_models()}")
    else:
        print(f"   ⚠️ Ollama not available - using rule-based fallback")
        print("   To enable LLM: ollama serve && ollama pull llama3.2")
    
    # Test analysis explanation
    print("\n📊 Testing analysis explanation...")
    
    measurements = {
        "vpp": 5.5,
        "rise_time_us": 0.5,
        "overshoot_pct": 35.0,
        "thd_pct": 2.5,
        "ringing_freq_hz": 500000
    }
    
    problems = ["OVERSHOOT", "RINGING"]
    diagnosis = "High overshoot (35%) and ringing detected at 500kHz"
    
    explanation = llm.explain_analysis(measurements, problems, diagnosis)
    
    print(f"\n   Summary: {explanation.summary}")
    print(f"\n   Using: {explanation.model_used}")
    print(f"   Latency: {explanation.latency_ms:.0f}ms")
    print(f"\n   Recommended actions:")
    for i, action in enumerate(explanation.recommended_actions[:3], 1):
        print(f"   {i}. {action}")
    
    # Test classification explanation
    print("\n\n🧠 Testing classification explanation...")
    
    explanation = llm.explain_classification(
        "NOISE",
        0.95,
        {"NOISE": 0.95, "NORMAL": 0.03, "DISTORTION": 0.02}
    )
    
    print(f"\n   Summary: {explanation.summary}")
    print(f"   Using: {explanation.model_used}")
    
    print("\n✅ Demo complete!")
