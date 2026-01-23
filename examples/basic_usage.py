"""
Example: Basic WaveformGPT Usage

This example demonstrates:
1. Natural language VCD queries (original)
2. Analog waveform analysis with v2.0
3. CNN-based problem classification (new!)
"""

import numpy as np

# =============================================================================
# Original: VCD Natural Language Queries
# =============================================================================

print("=" * 60)
print("1. VCD Natural Language Queries")
print("=" * 60)

try:
    from waveformgpt import WaveformChat
    
    # Load a VCD file
    chat = WaveformChat("sample.vcd")
    
    # Ask natural language questions
    response = chat.ask("When does clk rise?")
    print(response.content)
    
    response = chat.ask("How many times does req go high?")
    print(response.content)
    
except FileNotFoundError:
    print("No sample.vcd found - skip VCD demo")
except Exception as e:
    print(f"VCD demo skipped: {e}")


# =============================================================================
# v2.0: Analog Waveform Analysis
# =============================================================================

print("\n" + "=" * 60)
print("2. Analog Waveform Analysis (v2.0)")
print("=" * 60)

try:
    from waveformgpt.waveformgpt_v2 import WaveformGPT
    from waveformgpt.spice_simulator import format_component
    
    # Create analyzer
    gpt = WaveformGPT(sample_rate=1e6, duration=100e-6)
    
    # Simulate an underdamped RLC circuit (will have overshoot)
    waveform = gpt.simulate_circuit(
        circuit_type='rlc',
        R=50,      # Low R = underdamped
        L=10e-6,   # 10 µH
        C=1e-9     # 1 nF
    )
    
    # Analyze
    result = gpt.analyze(waveform)
    
    print("\n📏 Measurements:")
    measurements = gpt.get_measurements(waveform)
    print(f"   Vpp: {measurements['vpp']:.3f} V")
    print(f"   Rise time: {measurements['rise_time']*1e6:.2f} µs")
    print(f"   Overshoot: {measurements['overshoot_pct']:.1f}%")
    
    print("\n🔧 Diagnosis:")
    print(result['summary'])

except Exception as e:
    print(f"v2.0 demo error: {e}")


# =============================================================================
# NEW: CNN Classifier
# =============================================================================

print("\n" + "=" * 60)
print("3. CNN Waveform Classification (NEW!)")
print("=" * 60)

try:
    from waveformgpt.waveform_cnn import (
        WaveformClassifier, SyntheticDataGenerator, WaveformClass
    )
    
    # Create classifier (auto-loads trained model if exists)
    classifier = WaveformClassifier()
    
    # Generate test waveforms
    gen = SyntheticDataGenerator()
    
    test_cases = [
        ("Clean signal", gen.generate_normal),
        ("With overshoot", gen.generate_overshoot),
        ("With ringing", gen.generate_ringing),
        ("Noisy", gen.generate_noise),
        ("Clipped", gen.generate_clipping),
        ("Distorted", gen.generate_distortion),
    ]
    
    print("\n🧠 Classification Results:")
    print("-" * 50)
    for name, gen_func in test_cases:
        waveform = gen_func()
        result = classifier.classify(waveform)
        
        # Get top 2 predictions
        probs = result.all_probabilities
        sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
        
        print(f"\n   {name}:")
        print(f"   ➤ Predicted: {result.predicted_class.name} ({result.confidence:.0%})")
        if len(sorted_probs) > 1 and sorted_probs[1][1] > 0.1:
            print(f"   ➤ Also possible: {sorted_probs[1][0]} ({sorted_probs[1][1]:.0%})")
    
    # Human-readable diagnosis
    print("\n" + "-" * 50)
    print("\n💡 Example Diagnosis:")
    test_waveform = gen.generate_overshoot()
    print(f"   {classifier.get_diagnosis(test_waveform)}")

except Exception as e:
    print(f"CNN demo error: {e}")
    import traceback
    traceback.print_exc()


# =============================================================================
# Full Pipeline: Image → CNN → Diagnosis → Optimization
# =============================================================================

print("\n" + "=" * 60)
print("4. Full Pipeline Example")
print("=" * 60)

print("""
# Full workflow with an oscilloscope screenshot:

from waveformgpt.waveformgpt_v2 import WaveformGPT

gpt = WaveformGPT()

# 1. Extract waveform from image
result = gpt.analyze_image("scope_screenshot.png")

# 2. Get CNN classification
classification = gpt.classify(result['waveform'])
print(f"Problem: {classification['predicted_class']}")
print(f"Diagnosis: {classification['diagnosis']}")

# 3. Get detailed measurements
print(result['features'])

# 4. Get fix recommendations
print(result['summary'])

# 5. Optimize circuit to match a target response
target = gpt.simulate_circuit('rlc', R=200, L=10e-6, C=1e-9)
optimal = gpt.optimize_to_target(target, 'rlc', n_iterations=100)
print(f"Optimal R={optimal['R']}, L={optimal['L']}, C={optimal['C']}")
""")

print("\n✅ Demo complete!")
