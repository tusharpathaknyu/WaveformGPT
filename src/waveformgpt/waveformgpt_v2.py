"""
WaveformGPT v2.0 - Complete Pipeline

Full integration of:
1. Image extraction (oscilloscope screenshot → waveform data)
2. DSP analysis (measurements: rise time, overshoot, THD, etc.)
3. Rule-based diagnosis (problem detection → fix recommendations)
4. Bayesian optimization (find optimal component values)

Usage:
    from waveformgpt import WaveformGPT
    
    gpt = WaveformGPT()
    
    # From image
    result = gpt.analyze_image("scope_screenshot.png")
    print(result['diagnosis'])
    
    # Optimize circuit to match target
    optimal = gpt.optimize_to_target(target_waveform, circuit_type='rlc')
    print(f"Use R={optimal['R']}, L={optimal['L']}, C={optimal['C']}")
"""

import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path

# Import our modules (try relative import first, then fallback to direct)
try:
    from waveformgpt.circuit_optimizer import (
        CircuitOptimizer, DSPAnalyzer, RuleBasedDiagnostic,
        WaveformFeatures, CircuitFix
    )
    from waveformgpt.spice_simulator import CircuitSimulator, format_component
except ImportError:
    from circuit_optimizer import (
        CircuitOptimizer, DSPAnalyzer, RuleBasedDiagnostic,
        WaveformFeatures, CircuitFix
    )
    from spice_simulator import CircuitSimulator, format_component

# Try to import image extractor (needs OpenCV)
try:
    from waveformgpt.image_extractor import WaveformImageExtractor, QuickExtractor
    HAS_IMAGE_EXTRACTOR = True
except ImportError:
    try:
        from image_extractor import WaveformImageExtractor, QuickExtractor
        HAS_IMAGE_EXTRACTOR = True
    except ImportError:
        HAS_IMAGE_EXTRACTOR = False

# Try to import CNN classifier (needs numpy, optionally PyTorch)
try:
    from waveformgpt.waveform_cnn import (
        WaveformClassifier, WaveformClass, ClassificationResult,
        SyntheticDataGenerator
    )
    HAS_CNN = True
except ImportError:
    try:
        from waveform_cnn import (
            WaveformClassifier, WaveformClass, ClassificationResult,
            SyntheticDataGenerator
        )
        HAS_CNN = True
    except ImportError:
        HAS_CNN = False


class WaveformGPT:
    """
    Main interface for WaveformGPT v2.0
    
    Combines DSP analysis, rule-based diagnosis, and optimization.
    """
    
    def __init__(self, sample_rate: float = 1e6, duration: float = 100e-6):
        """
        Initialize WaveformGPT.
        
        Args:
            sample_rate: Default sample rate for analysis
            duration: Default simulation duration
        """
        self.sample_rate = sample_rate
        self.duration = duration
        
        # Initialize components
        self.dsp = DSPAnalyzer(sample_rate)
        self.rules = RuleBasedDiagnostic()
        self.simulator = CircuitSimulator(sample_rate=sample_rate, duration=duration)
        
        if HAS_IMAGE_EXTRACTOR:
            self.image_extractor = QuickExtractor()
        else:
            self.image_extractor = None
        
        # CNN classifier (optional)
        self.classifier = None
        if HAS_CNN:
            try:
                self.classifier = WaveformClassifier()
            except:
                pass
    
    def analyze(self, waveform: np.ndarray) -> Dict:
        """
        Analyze a waveform array.
        
        Args:
            waveform: Numpy array of voltage samples
        
        Returns:
            Dict with features, problems, fixes, and summary
        """
        features = self.dsp.extract_features(waveform)
        fixes = self.rules.diagnose(features)
        summary = self.rules.summarize(fixes)
        
        return {
            'features': features,
            'problems': [f.problem for f in fixes],
            'fixes': fixes,
            'summary': summary,
            'waveform': waveform
        }
    
    def analyze_image(
        self, 
        image_path: str,
        channel: Optional[str] = None,
        volts_per_div: Optional[float] = None,
        time_per_div: Optional[float] = None
    ) -> Dict:
        """
        Analyze waveform from oscilloscope screenshot.
        
        Args:
            image_path: Path to oscilloscope screenshot
            channel: 'yellow', 'cyan', 'magenta', 'green', or None for auto
            volts_per_div: Voltage scale setting (optional)
            time_per_div: Time scale setting (optional)
        
        Returns:
            Dict with extraction results and analysis
        """
        if not HAS_IMAGE_EXTRACTOR:
            raise ImportError("Image extraction requires OpenCV: pip install opencv-python")
        
        # Extract waveform from image
        extraction = self.image_extractor.from_image(
            image_path,
            channel_color=channel,
            volts_per_div=volts_per_div,
            time_per_div=time_per_div
        )
        
        # Analyze extracted waveform
        analysis = self.analyze(extraction['waveform'])
        
        # Combine results
        return {
            'extraction': extraction,
            'features': analysis['features'],
            'problems': analysis['problems'],
            'fixes': analysis['fixes'],
            'summary': analysis['summary'],
            'waveform': extraction['waveform']
        }
    
    def quick_diagnosis(self, waveform: np.ndarray) -> str:
        """Get quick text diagnosis"""
        result = self.analyze(waveform)
        return result['summary']
    
    def classify(self, waveform: np.ndarray) -> Dict:
        """
        Classify waveform problem type using CNN or feature-based classifier.
        
        Args:
            waveform: Numpy array of voltage samples
            
        Returns:
            Dict with predicted_class, confidence, and all_probabilities
        """
        if self.classifier is None:
            # Fallback: use rule-based as classifier
            features = self.dsp.extract_features(waveform)
            fixes = self.rules.diagnose(features)
            
            if not fixes:
                return {
                    'predicted_class': 'NORMAL',
                    'confidence': 0.8,
                    'diagnosis': '✅ Waveform looks healthy'
                }
            
            # Return primary problem
            primary = fixes[0]
            return {
                'predicted_class': primary.problem.name,
                'confidence': 0.7,
                'diagnosis': self.classifier.get_diagnosis(waveform) if self.classifier else primary.fix
            }
        
        result = self.classifier.classify(waveform)
        return {
            'predicted_class': result.predicted_class.name,
            'confidence': result.confidence,
            'all_probabilities': result.all_probabilities,
            'diagnosis': self.classifier.get_diagnosis(waveform)
        }

    def get_measurements(self, waveform: np.ndarray) -> Dict:
        """
        Get precise measurements from waveform.
        
        Returns dict with all measured values.
        """
        features = self.dsp.extract_features(waveform)
        
        return {
            'vpp': features.vpp,
            'vmax': features.vmax,
            'vmin': features.vmin,
            'vrms': features.vrms,
            'dc_offset': features.dc_offset,
            'rise_time': features.rise_time,
            'fall_time': features.fall_time,
            'overshoot_pct': features.overshoot_pct,
            'undershoot_pct': features.undershoot_pct,
            'ringing_freq': features.ringing_freq,
            'settling_time': features.settling_time,
            'frequency': features.frequency,
            'duty_cycle': features.duty_cycle,
            'noise_rms': features.noise_rms,
            'thd_pct': features.thd_pct
        }
    
    def simulate_circuit(
        self,
        circuit_type: str = 'rlc',
        **components
    ) -> np.ndarray:
        """
        Simulate a circuit and return output waveform.
        
        Args:
            circuit_type: 'rc', 'rl', 'rlc'
            **components: Component values (R, L, C, etc.)
        
        Returns:
            Output voltage waveform
        """
        if circuit_type == 'rc':
            return self.simulator.simulate_rc(
                R=components.get('R', 1000),
                C=components.get('C', 1e-9)
            )
        elif circuit_type == 'rlc':
            return self.simulator.simulate_rlc(
                R=components.get('R', 100),
                L=components.get('L', 10e-6),
                C=components.get('C', 1e-9)
            )
        else:
            raise ValueError(f"Unknown circuit type: {circuit_type}")
    
    def optimize_to_target(
        self,
        target_waveform: np.ndarray,
        circuit_type: str = 'rlc',
        n_iterations: int = 100
    ) -> Dict:
        """
        Find component values that produce target waveform.
        
        Args:
            target_waveform: Desired output waveform
            circuit_type: 'rc' or 'rlc'
            n_iterations: Optimization iterations (more = better but slower)
        
        Returns:
            Dict with optimal component values and formatted strings
        """
        result = self.simulator.optimize_components(
            target_waveform,
            circuit_type=circuit_type,
            n_calls=n_iterations
        )
        
        # Add formatted values
        formatted = {}
        for key, value in result.items():
            if key == 'R':
                formatted['R_formatted'] = format_component(value, 'Ω')
            elif key == 'L':
                formatted['L_formatted'] = format_component(value, 'H')
            elif key == 'C':
                formatted['C_formatted'] = format_component(value, 'F')
        
        return {**result, **formatted}
    
    def compare_waveforms(
        self,
        waveform1: np.ndarray,
        waveform2: np.ndarray
    ) -> Dict:
        """
        Compare two waveforms and report differences.
        """
        # Align lengths
        min_len = min(len(waveform1), len(waveform2))
        w1 = waveform1[:min_len]
        w2 = waveform2[:min_len]
        
        # Get features for both
        f1 = self.dsp.extract_features(w1)
        f2 = self.dsp.extract_features(w2)
        
        # Calculate differences
        mse = float(np.mean((w1 - w2)**2))
        max_diff = float(np.max(np.abs(w1 - w2)))
        correlation = float(np.corrcoef(w1, w2)[0, 1])
        
        return {
            'mse': mse,
            'max_difference': max_diff,
            'correlation': correlation,
            'rise_time_diff': f1.rise_time - f2.rise_time,
            'overshoot_diff': f1.overshoot_pct - f2.overshoot_pct,
            'features1': f1,
            'features2': f2
        }
    
    def generate_report(self, waveform: np.ndarray, title: str = "Waveform Analysis") -> str:
        """
        Generate a complete markdown report for a waveform.
        """
        analysis = self.analyze(waveform)
        f = analysis['features']
        
        report = f"""# {title}

## Measurements

| Parameter | Value |
|-----------|-------|
| Peak-to-Peak | {f.vpp*1000:.2f} mV |
| V_max | {f.vmax*1000:.2f} mV |
| V_min | {f.vmin*1000:.2f} mV |
| V_rms | {f.vrms*1000:.2f} mV |
| DC Offset | {f.dc_offset*1000:.2f} mV |
| Rise Time | {f.rise_time*1e6:.2f} µs |
| Fall Time | {f.fall_time*1e6:.2f} µs |
| Overshoot | {f.overshoot_pct:.1f}% |
| Ringing Freq | {f.ringing_freq/1e3:.1f} kHz |
| Settling Time | {f.settling_time*1e6:.1f} µs |
| Noise RMS | {f.noise_rms*1000:.2f} mV |
| THD | {f.thd_pct:.2f}% |

## Diagnosis

{analysis['summary']}

## Recommendations

"""
        for i, fix in enumerate(analysis['fixes'], 1):
            report += f"""### {i}. {fix.problem.value.replace('_', ' ').title()}

- **Severity:** {fix.severity}
- **Action:** {fix.action} {fix.component}
- **Suggested Value:** {fix.suggested_value}
- **Explanation:** {fix.explanation}

"""
        
        return report


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("WaveformGPT v2.0 - Full Pipeline Demo")
    print("=" * 60)
    
    # Initialize
    gpt = WaveformGPT(sample_rate=10e6, duration=50e-6)
    
    # Create test waveform: RLC step response with overshoot
    print("\n📊 Creating test waveform (RLC with overshoot)...")
    
    test_waveform = gpt.simulate_circuit(
        circuit_type='rlc',
        R=50,      # 50 ohms - low, causes underdamped response
        L=10e-6,   # 10 µH
        C=1e-9     # 1 nF
    )
    
    # Add some noise
    test_waveform += np.random.normal(0, 0.02, len(test_waveform))
    
    # Analyze
    print("\n🔍 Analyzing waveform...")
    result = gpt.analyze(test_waveform)
    
    print("\n📏 Measurements:")
    f = result['features']
    print(f"   Vpp: {f.vpp:.3f} V")
    print(f"   Rise time: {f.rise_time*1e6:.2f} µs")
    print(f"   Overshoot: {f.overshoot_pct:.1f}%")
    print(f"   Ringing: {f.ringing_freq/1e3:.1f} kHz" if f.ringing_freq > 0 else "   Ringing: None detected")
    
    print("\n🔧 Diagnosis:")
    print(result['summary'])
    
    # Test CNN classifier
    print("\n" + "=" * 60)
    print("🧠 Testing CNN Classifier")
    print("=" * 60)
    
    if HAS_CNN:
        classification = gpt.classify(test_waveform)
        print(f"\n   Predicted: {classification['predicted_class']}")
        print(f"   Confidence: {classification['confidence']:.1%}")
        print(f"   {classification['diagnosis']}")
        
        # Test on different waveform types
        print("\n   Testing on synthetic waveforms:")
        gen = SyntheticDataGenerator()
        test_types = [
            ("Clean step", gen.generate_normal),
            ("Overshoot", gen.generate_overshoot),
            ("Noisy", gen.generate_noise),
            ("Clipped", gen.generate_clipping),
        ]
        for name, gen_func in test_types:
            test_wave = gen_func()
            result_cls = gpt.classify(test_wave)
            print(f"   - {name}: {result_cls['predicted_class']} ({result_cls['confidence']:.0%})")
    else:
        print("\n   CNN not available (install PyTorch for better accuracy)")
        classification = gpt.classify(test_waveform)
        print(f"\n   Using rule-based classifier:")
        print(f"   Predicted: {classification['predicted_class']}")
    
    # Test optimization
    print("\n" + "=" * 60)
    print("🎯 Testing Component Optimization")
    print("=" * 60)
    
    # Create a target waveform
    print("\nCreating target waveform (critically damped response)...")
    target = gpt.simulate_circuit(
        circuit_type='rlc',
        R=200,     # Higher R = more damping
        L=10e-6,
        C=1e-9
    )
    
    print("Running optimization (50 iterations)...")
    try:
        optimal = gpt.optimize_to_target(target, circuit_type='rlc', n_iterations=50)
        
        print("\n✅ Optimization complete!")
        print(f"   R = {optimal.get('R_formatted', format_component(optimal['R'], 'Ω'))}")
        print(f"   L = {optimal.get('L_formatted', format_component(optimal['L'], 'H'))}")
        print(f"   C = {optimal.get('C_formatted', format_component(optimal['C'], 'F'))}")
        print(f"   MSE = {optimal['mse']:.6f}")
        
        print("\n   (Target was R=200Ω, L=10µH, C=1nF)")
        
    except ImportError as e:
        print(f"\n⚠️ Optimization requires: pip install scikit-optimize")
    
    # Generate report
    print("\n" + "=" * 60)
    print("📄 Generating Report")
    print("=" * 60)
    
    report = gpt.generate_report(test_waveform, "Test RLC Circuit Analysis")
    print("\n" + report[:1000] + "...\n")
    
    print("✅ Demo complete!")
    print("\nUsage:")
    print("   gpt = WaveformGPT()")
    print("   result = gpt.analyze(waveform_data)")
    print("   result = gpt.classify(waveform_data)  # CNN classification")
    print("   result = gpt.analyze_image('scope.png')  # If OpenCV installed")
    print("   optimal = gpt.optimize_to_target(target, 'rlc')")
