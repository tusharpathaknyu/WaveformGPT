"""
WaveformGPT Circuit Optimizer

Hybrid approach:
1. DSP feature extraction (fast, accurate measurements)
2. Rule-based diagnosis (expert knowledge)
3. Bayesian optimization (find optimal component values)
4. LLM explanation (optional, for natural language output)

No GPU required - runs on CPU/Mac.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Optional: for Bayesian optimization
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False
    print("Install scikit-optimize for Bayesian optimization: pip install scikit-optimize")


class WaveformProblem(Enum):
    """Detected waveform issues"""
    OVERSHOOT = "overshoot"
    UNDERSHOOT = "undershoot"
    RINGING = "ringing"
    SLOW_RISE = "slow_rise"
    SLOW_FALL = "slow_fall"
    NOISE = "noise"
    DC_OFFSET = "dc_offset"
    CLIPPING = "clipping"
    DISTORTION = "distortion"
    PHASE_SHIFT = "phase_shift"


@dataclass
class WaveformFeatures:
    """Extracted waveform measurements"""
    vpp: float              # Peak-to-peak voltage
    vmax: float             # Maximum voltage
    vmin: float             # Minimum voltage
    vrms: float             # RMS voltage
    dc_offset: float        # DC component
    rise_time: float        # 10% to 90% rise time (seconds)
    fall_time: float        # 90% to 10% fall time (seconds)
    overshoot_pct: float    # Overshoot percentage
    undershoot_pct: float   # Undershoot percentage
    ringing_freq: float     # Ringing frequency (Hz), 0 if none
    settling_time: float    # Time to settle within 2%
    frequency: float        # Dominant frequency
    duty_cycle: float       # For square waves
    noise_rms: float        # Noise RMS
    thd_pct: float          # Total harmonic distortion


@dataclass
class CircuitFix:
    """Recommended circuit modification"""
    problem: WaveformProblem
    severity: str           # "low", "medium", "high"
    component: str          # e.g., "R1", "C_snubber", "L1"
    action: str             # "add", "remove", "increase", "decrease"
    suggested_value: str    # e.g., "100nF", "10kΩ"
    explanation: str        # Why this fix works


class DSPAnalyzer:
    """
    Pure DSP waveform analysis - no ML needed.
    Fast, accurate, deterministic.
    """
    
    def __init__(self, sample_rate: float = 1e6):
        self.sample_rate = sample_rate
    
    def extract_features(self, waveform: np.ndarray) -> WaveformFeatures:
        """Extract all features from waveform data"""
        
        # Basic measurements
        vmax = float(np.max(waveform))
        vmin = float(np.min(waveform))
        vpp = vmax - vmin
        vrms = float(np.sqrt(np.mean(waveform**2)))
        dc_offset = float(np.mean(waveform))
        
        # Rise/fall time
        rise_time = self._measure_rise_time(waveform)
        fall_time = self._measure_fall_time(waveform)
        
        # Overshoot (assumes step response)
        overshoot_pct = self._measure_overshoot(waveform)
        undershoot_pct = self._measure_undershoot(waveform)
        
        # Ringing detection
        ringing_freq = self._detect_ringing(waveform)
        
        # Settling time
        settling_time = self._measure_settling_time(waveform)
        
        # Frequency analysis
        frequency = self._measure_frequency(waveform)
        
        # Duty cycle (for square waves)
        duty_cycle = self._measure_duty_cycle(waveform)
        
        # Noise estimation
        noise_rms = self._estimate_noise(waveform)
        
        # THD
        thd_pct = self._calculate_thd(waveform)
        
        return WaveformFeatures(
            vpp=vpp, vmax=vmax, vmin=vmin, vrms=vrms, dc_offset=dc_offset,
            rise_time=rise_time, fall_time=fall_time,
            overshoot_pct=overshoot_pct, undershoot_pct=undershoot_pct,
            ringing_freq=ringing_freq, settling_time=settling_time,
            frequency=frequency, duty_cycle=duty_cycle,
            noise_rms=noise_rms, thd_pct=thd_pct
        )
    
    def _measure_rise_time(self, waveform: np.ndarray) -> float:
        """Measure 10% to 90% rise time"""
        vmin, vmax = np.min(waveform), np.max(waveform)
        v10 = vmin + 0.1 * (vmax - vmin)
        v90 = vmin + 0.9 * (vmax - vmin)
        
        # Find first rising edge
        above_10 = np.where(waveform > v10)[0]
        above_90 = np.where(waveform > v90)[0]
        
        if len(above_10) > 0 and len(above_90) > 0:
            # Find where 90% crossing comes after 10% crossing
            t10 = above_10[0]
            t90_candidates = above_90[above_90 > t10]
            if len(t90_candidates) > 0:
                t90 = t90_candidates[0]
                return (t90 - t10) / self.sample_rate
        
        return 0.0
    
    def _measure_fall_time(self, waveform: np.ndarray) -> float:
        """Measure 90% to 10% fall time"""
        vmin, vmax = np.min(waveform), np.max(waveform)
        v10 = vmin + 0.1 * (vmax - vmin)
        v90 = vmin + 0.9 * (vmax - vmin)
        
        # Find falling edge
        below_90 = np.where(waveform < v90)[0]
        below_10 = np.where(waveform < v10)[0]
        
        # Look for 90%→10% transition
        for i in range(len(waveform) - 1):
            if waveform[i] > v90 and waveform[i+1] < v90:
                t90 = i
                t10_candidates = below_10[below_10 > t90]
                if len(t10_candidates) > 0:
                    t10 = t10_candidates[0]
                    return (t10 - t90) / self.sample_rate
        
        return 0.0
    
    def _measure_overshoot(self, waveform: np.ndarray) -> float:
        """
        Measure overshoot as percentage of step size.
        Improved algorithm that works for both step responses and oscillating signals.
        """
        n = len(waveform)
        if n < 20:
            return 0.0
        
        # Get initial and final values
        initial = np.mean(waveform[:max(1, n//20)])  # First 5%
        final = np.mean(waveform[-max(1, n//4):])    # Last 25%
        
        step_size = abs(final - initial)
        
        # Handle oscillating signals (no clear step)
        if step_size < 0.01 * (np.max(waveform) - np.min(waveform)):
            # Likely oscillating, not a step response
            return 0.0
        
        # For rising step
        if final > initial:
            peak = np.max(waveform)
            if peak > final:
                overshoot = (peak - final) / step_size * 100
                return min(overshoot, 100.0)  # Cap at 100%
        # For falling step  
        else:
            trough = np.min(waveform)
            if trough < final:
                overshoot = (final - trough) / step_size * 100
                return min(overshoot, 100.0)
        
        return 0.0
    
    def _measure_undershoot(self, waveform: np.ndarray) -> float:
        """Measure undershoot as percentage"""
        steady_state = np.mean(waveform[-len(waveform)//4:])
        initial = np.mean(waveform[:len(waveform)//10])
        trough = np.min(waveform)
        
        step_size = abs(steady_state - initial)
        if step_size > 0 and steady_state > initial:
            undershoot = (initial - trough) / step_size * 100
            return max(0, undershoot)
        return 0.0
    
    def _detect_ringing(self, waveform: np.ndarray) -> float:
        """
        Detect ringing frequency by looking for damped oscillations.
        Returns 0 if no significant ringing detected.
        """
        n = len(waveform)
        if n < 50:
            return 0.0
        
        # Get the "settled" portion - last 25%
        final_value = np.mean(waveform[-n//4:])
        vpp = np.max(waveform) - np.min(waveform)
        
        if vpp == 0:
            return 0.0
        
        # Detrend: remove the step/ramp component
        # Fit a line to first half and subtract
        detrended = waveform - final_value
        
        # Look for zero crossings in detrended signal
        zero_crossings = np.where(np.diff(np.signbit(detrended)))[0]
        
        if len(zero_crossings) < 4:
            return 0.0
        
        # Check if oscillation amplitude is significant (>5% of Vpp)
        oscillation_amplitude = np.std(detrended[n//4:3*n//4])  # Middle 50%
        if oscillation_amplitude < 0.02 * vpp:
            return 0.0  # Too small to be meaningful ringing
        
        # Calculate frequency from zero crossings
        periods = np.diff(zero_crossings) * 2  # Full periods
        
        # Filter out outliers
        median_period = np.median(periods)
        valid_periods = periods[np.abs(periods - median_period) < median_period]
        
        if len(valid_periods) > 0:
            avg_period = np.mean(valid_periods)
            if avg_period > 0:
                freq = self.sample_rate / avg_period
                # Sanity check: freq should be reasonable
                if 100 < freq < self.sample_rate / 2:
                    return freq
        
        return 0.0
    
    def _measure_settling_time(self, waveform: np.ndarray) -> float:
        """Time to settle within 2% of final value"""
        steady_state = np.mean(waveform[-len(waveform)//4:])
        tolerance = 0.02 * abs(steady_state) if steady_state != 0 else 0.02
        
        # Find last point outside tolerance band
        outside = np.where(np.abs(waveform - steady_state) > tolerance)[0]
        if len(outside) > 0:
            return outside[-1] / self.sample_rate
        return 0.0
    
    def _measure_frequency(self, waveform: np.ndarray) -> float:
        """Measure dominant frequency using FFT"""
        spectrum = np.abs(np.fft.rfft(waveform))
        freqs = np.fft.rfftfreq(len(waveform), 1/self.sample_rate)
        
        # Ignore DC component
        spectrum[0] = 0
        
        if len(spectrum) > 1:
            peak_idx = np.argmax(spectrum)
            return float(freqs[peak_idx])
        return 0.0
    
    def _measure_duty_cycle(self, waveform: np.ndarray) -> float:
        """Measure duty cycle (for square waves)"""
        threshold = np.mean(waveform)
        high_samples = np.sum(waveform > threshold)
        return high_samples / len(waveform) * 100
    
    def _estimate_noise(self, waveform: np.ndarray) -> float:
        """Estimate noise using high-pass filter"""
        # Simple high-pass: differentiate and look at high-freq content
        diff = np.diff(waveform)
        # Noise is the RMS of high-frequency components
        return float(np.std(diff) / np.sqrt(2))
    
    def _calculate_thd(self, waveform: np.ndarray) -> float:
        """Calculate Total Harmonic Distortion"""
        spectrum = np.abs(np.fft.rfft(waveform))
        
        if len(spectrum) < 6:
            return 0.0
        
        # Find fundamental
        spectrum[0] = 0  # Ignore DC
        fund_idx = np.argmax(spectrum)
        fundamental = spectrum[fund_idx]
        
        if fundamental == 0:
            return 0.0
        
        # Sum harmonics (2nd through 5th)
        harmonic_power = 0
        for h in range(2, 6):
            harm_idx = fund_idx * h
            if harm_idx < len(spectrum):
                harmonic_power += spectrum[harm_idx]**2
        
        return float(np.sqrt(harmonic_power) / fundamental * 100)


class RuleBasedDiagnostic:
    """
    Expert knowledge encoded as rules.
    Maps waveform problems to circuit fixes.
    """
    
    # Thresholds for problem detection
    OVERSHOOT_THRESHOLD = 10.0      # percent
    RINGING_THRESHOLD = 1000.0      # Hz (any detectable ringing)
    NOISE_THRESHOLD = 0.05          # relative to Vpp
    THD_THRESHOLD = 5.0             # percent
    SLOW_RISE_FACTOR = 5.0          # times expected
    
    def diagnose(self, features: WaveformFeatures) -> List[CircuitFix]:
        """Analyze features and return recommended fixes"""
        fixes = []
        
        # Overshoot detection
        if features.overshoot_pct > self.OVERSHOOT_THRESHOLD:
            severity = "high" if features.overshoot_pct > 30 else "medium" if features.overshoot_pct > 15 else "low"
            fixes.append(CircuitFix(
                problem=WaveformProblem.OVERSHOOT,
                severity=severity,
                component="C_snubber",
                action="add",
                suggested_value="10-100nF",
                explanation=f"Overshoot of {features.overshoot_pct:.1f}% detected. "
                           "Add a snubber capacitor (10-100nF) in parallel with the load, "
                           "or add a series gate resistor (10-100Ω) to slow down switching."
            ))
        
        # Ringing detection
        if features.ringing_freq > 0:
            severity = "high" if features.ringing_freq > 1e6 else "medium"
            # Calculate snubber values for detected frequency
            suggested_c = 1 / (2 * np.pi * features.ringing_freq * 50)  # Assume 50Ω
            fixes.append(CircuitFix(
                problem=WaveformProblem.RINGING,
                severity=severity,
                component="RC_snubber",
                action="add",
                suggested_value=f"R=10-50Ω, C={suggested_c*1e9:.0f}nF",
                explanation=f"Ringing at {features.ringing_freq/1e3:.1f}kHz detected. "
                           f"Add RC snubber tuned to this frequency. "
                           "Also check for parasitic inductance in traces."
            ))
        
        # Noise detection
        noise_ratio = features.noise_rms / features.vpp if features.vpp > 0 else 0
        if noise_ratio > self.NOISE_THRESHOLD:
            fixes.append(CircuitFix(
                problem=WaveformProblem.NOISE,
                severity="medium",
                component="C_bypass",
                action="add",
                suggested_value="100nF + 10µF",
                explanation=f"Noise level is {noise_ratio*100:.1f}% of signal. "
                           "Add bypass capacitors (100nF ceramic + 10µF electrolytic) "
                           "close to power pins. Check probe ground lead length."
            ))
        
        # THD (distortion)
        if features.thd_pct > self.THD_THRESHOLD:
            fixes.append(CircuitFix(
                problem=WaveformProblem.DISTORTION,
                severity="medium" if features.thd_pct < 10 else "high",
                component="input_amplitude",
                action="decrease",
                suggested_value="Reduce by 20-50%",
                explanation=f"THD of {features.thd_pct:.1f}% indicates nonlinearity. "
                           "Reduce input signal amplitude to stay in linear region, "
                           "or add negative feedback to linearize."
            ))
        
        # DC offset
        dc_ratio = abs(features.dc_offset) / features.vpp if features.vpp > 0 else 0
        if dc_ratio > 0.1:
            fixes.append(CircuitFix(
                problem=WaveformProblem.DC_OFFSET,
                severity="low",
                component="C_coupling",
                action="add",
                suggested_value="1-10µF",
                explanation=f"DC offset of {features.dc_offset*1000:.1f}mV detected. "
                           "Add AC coupling capacitor if DC is unwanted, "
                           "or check resistor divider balance."
            ))
        
        return fixes
    
    def summarize(self, fixes: List[CircuitFix]) -> str:
        """Generate human-readable summary"""
        if not fixes:
            return "✅ Waveform looks healthy! No issues detected."
        
        lines = ["⚠️ Issues detected:\n"]
        for i, fix in enumerate(fixes, 1):
            lines.append(f"{i}. **{fix.problem.value.replace('_', ' ').title()}** ({fix.severity})")
            lines.append(f"   - Action: {fix.action} {fix.component}")
            lines.append(f"   - Suggested: {fix.suggested_value}")
            lines.append(f"   - {fix.explanation}\n")
        
        return "\n".join(lines)


class BayesianOptimizer:
    """
    Find optimal component values to match target waveform.
    Uses Bayesian optimization - only needs ~100 simulations.
    """
    
    def __init__(self, simulator_fn):
        """
        Args:
            simulator_fn: Function(R, C, L) -> waveform array
        """
        if not HAS_SKOPT:
            raise ImportError("Install scikit-optimize: pip install scikit-optimize")
        
        self.simulator = simulator_fn
    
    def optimize(
        self,
        target_waveform: np.ndarray,
        component_ranges: Dict[str, Tuple[float, float]],
        n_calls: int = 100,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Find component values that produce target waveform.
        
        Args:
            target_waveform: Target waveform to match
            component_ranges: Dict of component name -> (min, max) values
                e.g., {"R": (100, 100000), "C": (1e-12, 1e-6)}
            n_calls: Number of optimization iterations
            verbose: Print progress
        
        Returns:
            Dict of optimal component values
        """
        
        # Build search space
        component_names = list(component_ranges.keys())
        search_space = [
            Real(low, high, name=name, prior='log-uniform')
            for name, (low, high) in component_ranges.items()
        ]
        
        def objective(params):
            """Objective: minimize MSE between simulated and target"""
            component_values = dict(zip(component_names, params))
            try:
                simulated = self.simulator(**component_values)
                
                # Align lengths if needed
                min_len = min(len(simulated), len(target_waveform))
                mse = np.mean((simulated[:min_len] - target_waveform[:min_len])**2)
                return float(mse)
            except Exception as e:
                if verbose:
                    print(f"Simulation failed: {e}")
                return 1e10  # Large penalty
        
        # Run optimization
        result = gp_minimize(
            objective,
            search_space,
            n_calls=n_calls,
            n_random_starts=min(20, n_calls // 3),
            verbose=verbose,
            random_state=42
        )
        
        # Return optimal values
        optimal = dict(zip(component_names, result.x))
        optimal['_mse'] = result.fun
        optimal['_n_calls'] = n_calls
        
        return optimal


class CircuitOptimizer:
    """
    Main interface combining all approaches.
    """
    
    def __init__(self, sample_rate: float = 1e6):
        self.dsp = DSPAnalyzer(sample_rate)
        self.rules = RuleBasedDiagnostic()
        self.optimizer = None  # Created on demand
    
    def analyze(self, waveform: np.ndarray) -> Dict:
        """
        Full analysis of waveform.
        
        Returns dict with:
            - features: WaveformFeatures
            - problems: List of detected problems
            - fixes: List of CircuitFix recommendations
            - summary: Human-readable summary
        """
        features = self.dsp.extract_features(waveform)
        fixes = self.rules.diagnose(features)
        summary = self.rules.summarize(fixes)
        
        return {
            'features': features,
            'problems': [f.problem for f in fixes],
            'fixes': fixes,
            'summary': summary
        }
    
    def optimize_circuit(
        self,
        target_waveform: np.ndarray,
        simulator_fn,
        component_ranges: Dict[str, Tuple[float, float]],
        n_calls: int = 100
    ) -> Dict[str, float]:
        """
        Find optimal component values to produce target waveform.
        
        Args:
            target_waveform: Desired waveform
            simulator_fn: Circuit simulator function
            component_ranges: Component name -> (min, max) ranges
            n_calls: Optimization iterations
        
        Returns:
            Optimal component values
        """
        self.optimizer = BayesianOptimizer(simulator_fn)
        return self.optimizer.optimize(target_waveform, component_ranges, n_calls)
    
    def quick_diagnosis(self, waveform: np.ndarray) -> str:
        """Quick text summary of waveform issues"""
        result = self.analyze(waveform)
        return result['summary']


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("WaveformGPT Circuit Optimizer - Demo")
    print("=" * 60)
    
    # Create sample waveform with overshoot
    t = np.linspace(0, 100e-6, 10000)
    sample_rate = 10000 / 100e-6  # 100 MS/s
    
    # Simulated step response with overshoot and ringing
    tau = 10e-6
    omega = 2 * np.pi * 500e3  # 500 kHz ringing
    damping = 0.3
    waveform = 3.3 * (1 - np.exp(-t/tau) * (np.cos(omega*t) + damping*np.sin(omega*t)))
    waveform += np.random.normal(0, 0.05, len(waveform))  # Add noise
    
    # Analyze
    optimizer = CircuitOptimizer(sample_rate)
    result = optimizer.analyze(waveform)
    
    print("\n📊 Extracted Features:")
    print(f"   Vpp: {result['features'].vpp*1000:.1f} mV")
    print(f"   Rise time: {result['features'].rise_time*1e6:.2f} µs")
    print(f"   Overshoot: {result['features'].overshoot_pct:.1f}%")
    print(f"   Ringing freq: {result['features'].ringing_freq/1e3:.1f} kHz")
    print(f"   Noise RMS: {result['features'].noise_rms*1000:.1f} mV")
    
    print("\n🔧 Diagnosis:")
    print(result['summary'])
    
    print("\n✅ Demo complete!")
