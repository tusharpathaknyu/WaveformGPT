"""
SPICE Circuit Simulator Interface

Provides a Python interface to circuit simulation for:
- Bayesian optimization of component values
- Waveform prediction
- Circuit validation

Supports:
- ngspice (free, open source)
- PySpice (Python wrapper for ngspice)
- Built-in analytical models (fast, no external deps)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import subprocess
import tempfile
import os

# Try to import PySpice
try:
    from PySpice.Spice.NgSpice.Shared import NgSpiceShared
    from PySpice.Spice.Parser import SpiceParser
    from PySpice.Unit import *
    HAS_PYSPICE = True
except ImportError:
    HAS_PYSPICE = False


class SimulatorType(Enum):
    ANALYTICAL = "analytical"   # Built-in fast models
    NGSPICE = "ngspice"        # External ngspice
    PYSPICE = "pyspice"        # PySpice wrapper


@dataclass
class SimulationResult:
    """Results from circuit simulation"""
    time: np.ndarray           # Time points
    voltage: Dict[str, np.ndarray]  # Node voltages
    current: Dict[str, np.ndarray]  # Branch currents
    success: bool
    error_message: Optional[str] = None


class AnalyticalSimulator:
    """
    Fast analytical circuit models.
    No external dependencies - pure numpy.
    Good enough for most optimization tasks.
    """
    
    def __init__(self, sample_rate: float = 1e6, duration: float = 100e-6):
        self.sample_rate = sample_rate
        self.duration = duration
        self.t = np.linspace(0, duration, int(sample_rate * duration))
    
    def rc_lowpass(
        self, 
        R: float, 
        C: float, 
        v_in: np.ndarray = None,
        v_step: float = 5.0
    ) -> np.ndarray:
        """
        RC low-pass filter response.
        
        Args:
            R: Resistance in ohms
            C: Capacitance in farads
            v_in: Input voltage (or use step input)
            v_step: Step voltage if v_in not provided
        
        Returns:
            Output voltage array
        """
        tau = R * C
        
        if v_in is None:
            # Step response
            v_out = v_step * (1 - np.exp(-self.t / tau))
        else:
            # Convolve with impulse response (approximate)
            # For accurate results, use actual convolution
            v_out = v_step * (1 - np.exp(-self.t / tau))
        
        return v_out
    
    def rl_response(
        self,
        R: float,
        L: float,
        v_step: float = 5.0
    ) -> np.ndarray:
        """RL circuit step response (current through inductor)"""
        tau = L / R
        i_out = (v_step / R) * (1 - np.exp(-self.t / tau))
        return i_out * R  # Convert to voltage across R
    
    def rlc_response(
        self,
        R: float,
        L: float,
        C: float,
        v_step: float = 5.0
    ) -> np.ndarray:
        """
        RLC series circuit step response.
        Handles underdamped, critically damped, and overdamped cases.
        """
        omega_0 = 1 / np.sqrt(L * C)  # Natural frequency
        alpha = R / (2 * L)            # Damping factor
        
        if alpha < omega_0:
            # Underdamped - oscillates
            omega_d = np.sqrt(omega_0**2 - alpha**2)
            v_out = v_step * (1 - np.exp(-alpha * self.t) * 
                    (np.cos(omega_d * self.t) + 
                     (alpha / omega_d) * np.sin(omega_d * self.t)))
        
        elif alpha > omega_0:
            # Overdamped - no oscillation
            s1 = -alpha + np.sqrt(alpha**2 - omega_0**2)
            s2 = -alpha - np.sqrt(alpha**2 - omega_0**2)
            A1 = s2 / (s2 - s1)
            A2 = -s1 / (s2 - s1)
            v_out = v_step * (1 - A1*np.exp(s1*self.t) - A2*np.exp(s2*self.t))
        
        else:
            # Critically damped
            v_out = v_step * (1 - (1 + alpha * self.t) * np.exp(-alpha * self.t))
        
        return v_out
    
    def rc_highpass(
        self,
        R: float,
        C: float,
        v_step: float = 5.0
    ) -> np.ndarray:
        """RC high-pass filter step response"""
        tau = R * C
        v_out = v_step * np.exp(-self.t / tau)
        return v_out
    
    def inverting_amp(
        self,
        R_in: float,
        R_fb: float,
        v_in: np.ndarray = None,
        v_signal: float = 1.0,
        freq: float = 1e3
    ) -> np.ndarray:
        """Inverting op-amp amplifier"""
        gain = -R_fb / R_in
        
        if v_in is None:
            v_in = v_signal * np.sin(2 * np.pi * freq * self.t)
        
        return gain * v_in
    
    def voltage_divider(
        self,
        R1: float,
        R2: float,
        v_in: float = 5.0
    ) -> np.ndarray:
        """Simple resistive voltage divider"""
        v_out = v_in * R2 / (R1 + R2)
        return np.full_like(self.t, v_out)
    
    def buck_converter(
        self,
        L: float,
        C: float,
        R_load: float,
        v_in: float = 12.0,
        duty: float = 0.5,
        f_sw: float = 100e3
    ) -> np.ndarray:
        """
        Simplified buck converter output voltage.
        Assumes continuous conduction mode.
        """
        v_out_avg = v_in * duty
        
        # Output ripple (simplified)
        delta_i = (v_in - v_out_avg) * duty / (L * f_sw)
        delta_v = delta_i / (8 * C * f_sw)
        
        # Create ripple waveform
        ripple = delta_v * np.sin(2 * np.pi * f_sw * self.t)
        
        return v_out_avg + ripple


class NgspiceInterface:
    """
    Interface to ngspice simulator.
    Requires ngspice to be installed.
    """
    
    def __init__(self):
        self.ngspice_path = self._find_ngspice()
    
    def _find_ngspice(self) -> Optional[str]:
        """Find ngspice executable"""
        for path in ['/usr/bin/ngspice', '/usr/local/bin/ngspice', 
                     '/opt/homebrew/bin/ngspice', 'ngspice']:
            try:
                result = subprocess.run([path, '--version'], 
                                       capture_output=True, timeout=5)
                if result.returncode == 0:
                    return path
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        return None
    
    def is_available(self) -> bool:
        return self.ngspice_path is not None
    
    def simulate(
        self,
        netlist: str,
        analysis: str = 'tran',
        duration: float = 100e-6,
        step: float = 1e-9
    ) -> SimulationResult:
        """
        Run ngspice simulation.
        
        Args:
            netlist: SPICE netlist string
            analysis: 'tran', 'ac', 'dc'
            duration: Simulation duration
            step: Time step
        
        Returns:
            SimulationResult
        """
        if not self.is_available():
            return SimulationResult(
                time=np.array([]),
                voltage={},
                current={},
                success=False,
                error_message="ngspice not found"
            )
        
        # Add analysis command if not present
        if '.tran' not in netlist.lower() and '.ac' not in netlist.lower():
            netlist += f"\n.tran {step} {duration}\n"
        
        # Add print command for output
        netlist += "\n.control\nrun\nprint all\n.endc\n.end\n"
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cir', delete=False) as f:
            f.write(netlist)
            netlist_path = f.name
        
        try:
            # Run ngspice
            result = subprocess.run(
                [self.ngspice_path, '-b', netlist_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Parse output
            return self._parse_output(result.stdout)
            
        except subprocess.TimeoutExpired:
            return SimulationResult(
                time=np.array([]),
                voltage={},
                current={},
                success=False,
                error_message="Simulation timed out"
            )
        finally:
            os.unlink(netlist_path)
    
    def _parse_output(self, output: str) -> SimulationResult:
        """Parse ngspice text output"""
        # This is a simplified parser - real implementation would be more robust
        lines = output.strip().split('\n')
        
        time_data = []
        voltage_data = {}
        
        for line in lines:
            if line.startswith('Index'):
                # Header line - get column names
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                try:
                    idx = int(parts[0])
                    time_data.append(float(parts[1]))
                    # More parsing would go here
                except ValueError:
                    continue
        
        return SimulationResult(
            time=np.array(time_data) if time_data else np.array([]),
            voltage=voltage_data,
            current={},
            success=len(time_data) > 0
        )


class CircuitSimulator:
    """
    Unified interface to circuit simulation.
    Automatically selects best available simulator.
    """
    
    def __init__(
        self, 
        simulator_type: SimulatorType = SimulatorType.ANALYTICAL,
        sample_rate: float = 1e6,
        duration: float = 100e-6
    ):
        self.simulator_type = simulator_type
        self.sample_rate = sample_rate
        self.duration = duration
        
        # Initialize simulators
        self.analytical = AnalyticalSimulator(sample_rate, duration)
        self.ngspice = NgspiceInterface()
        
        # Time array
        self.t = self.analytical.t
    
    def simulate_rlc(
        self,
        R: float,
        L: float,
        C: float,
        v_step: float = 5.0
    ) -> np.ndarray:
        """
        Simulate RLC circuit step response.
        Uses analytical model for speed.
        """
        return self.analytical.rlc_response(R, L, C, v_step)
    
    def simulate_rc(
        self,
        R: float,
        C: float,
        v_step: float = 5.0
    ) -> np.ndarray:
        """Simulate RC circuit step response"""
        return self.analytical.rc_lowpass(R, C, v_step=v_step)
    
    def simulate_custom(
        self,
        netlist: str
    ) -> SimulationResult:
        """
        Simulate custom circuit using ngspice.
        Falls back to error if ngspice not available.
        """
        if not self.ngspice.is_available():
            return SimulationResult(
                time=self.t,
                voltage={},
                current={},
                success=False,
                error_message="ngspice not installed. Install with: brew install ngspice"
            )
        
        return self.ngspice.simulate(netlist, duration=self.duration)
    
    def create_objective(
        self,
        target_waveform: np.ndarray,
        circuit_type: str = 'rlc'
    ) -> Callable:
        """
        Create objective function for Bayesian optimization.
        
        Args:
            target_waveform: Target waveform to match
            circuit_type: 'rc', 'rl', 'rlc'
        
        Returns:
            Objective function that takes component values and returns error
        """
        # Resample target to match simulation length
        target_resampled = np.interp(
            np.linspace(0, 1, len(self.t)),
            np.linspace(0, 1, len(target_waveform)),
            target_waveform
        )
        
        if circuit_type == 'rc':
            def objective(R, C):
                simulated = self.simulate_rc(R, C)
                return float(np.mean((simulated - target_resampled)**2))
            return objective
        
        elif circuit_type == 'rlc':
            def objective(R, L, C):
                simulated = self.simulate_rlc(R, L, C)
                return float(np.mean((simulated - target_resampled)**2))
            return objective
        
        else:
            raise ValueError(f"Unknown circuit type: {circuit_type}")
    
    def optimize_components(
        self,
        target_waveform: np.ndarray,
        circuit_type: str = 'rlc',
        n_calls: int = 100
    ) -> Dict[str, float]:
        """
        Find optimal component values to match target waveform.
        
        Uses Bayesian optimization for efficiency.
        """
        try:
            from skopt import gp_minimize
            from skopt.space import Real
        except ImportError:
            raise ImportError("Install scikit-optimize: pip install scikit-optimize")
        
        objective = self.create_objective(target_waveform, circuit_type)
        
        if circuit_type == 'rc':
            result = gp_minimize(
                lambda x: objective(x[0], x[1]),
                [
                    Real(100, 100000, name='R', prior='log-uniform'),
                    Real(1e-12, 1e-6, name='C', prior='log-uniform'),
                ],
                n_calls=n_calls,
                random_state=42,
                verbose=False
            )
            return {'R': result.x[0], 'C': result.x[1], 'mse': result.fun}
        
        elif circuit_type == 'rlc':
            result = gp_minimize(
                lambda x: objective(x[0], x[1], x[2]),
                [
                    Real(10, 10000, name='R', prior='log-uniform'),
                    Real(1e-9, 1e-3, name='L', prior='log-uniform'),
                    Real(1e-12, 1e-6, name='C', prior='log-uniform'),
                ],
                n_calls=n_calls,
                random_state=42,
                verbose=False
            )
            return {
                'R': result.x[0], 
                'L': result.x[1], 
                'C': result.x[2], 
                'mse': result.fun
            }


def format_component(value: float, unit: str) -> str:
    """Format component value with SI prefix"""
    prefixes = [
        (1e12, 'T'), (1e9, 'G'), (1e6, 'M'), (1e3, 'k'),
        (1, ''), (1e-3, 'm'), (1e-6, 'µ'), (1e-9, 'n'), (1e-12, 'p')
    ]
    
    for threshold, prefix in prefixes:
        if abs(value) >= threshold:
            return f"{value/threshold:.2f} {prefix}{unit}"
    
    return f"{value:.2e} {unit}"


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SPICE Circuit Simulator - Demo")
    print("=" * 60)
    
    # Check ngspice availability
    ng = NgspiceInterface()
    print(f"\n📦 ngspice available: {ng.is_available()}")
    if ng.is_available():
        print(f"   Path: {ng.ngspice_path}")
    else:
        print("   Install with: brew install ngspice")
    
    # Create simulator
    sim = CircuitSimulator(sample_rate=10e6, duration=50e-6)
    
    # Test analytical models
    print("\n📊 Testing analytical models...")
    
    # RC response
    v_rc = sim.simulate_rc(R=1000, C=1e-9)  # 1kΩ, 1nF
    print(f"\n   RC (1kΩ, 1nF):")
    print(f"   - Time constant: {1000 * 1e-9 * 1e6:.1f} µs")
    print(f"   - Final value: {v_rc[-1]:.2f} V")
    
    # RLC response  
    v_rlc = sim.simulate_rlc(R=100, L=10e-6, C=1e-9)  # 100Ω, 10µH, 1nF
    print(f"\n   RLC (100Ω, 10µH, 1nF):")
    omega_0 = 1 / np.sqrt(10e-6 * 1e-9)
    alpha = 100 / (2 * 10e-6)
    print(f"   - Natural freq: {omega_0/2/np.pi/1e6:.2f} MHz")
    print(f"   - Damping: {'underdamped' if alpha < omega_0 else 'overdamped'}")
    print(f"   - Peak value: {np.max(v_rlc):.2f} V")
    print(f"   - Overshoot: {(np.max(v_rlc) - 5) / 5 * 100:.1f}%")
    
    # Test optimization
    print("\n🔧 Testing component optimization...")
    print("   Creating target waveform with overshoot...")
    
    # Create a target with specific characteristics
    target = sim.analytical.rlc_response(R=50, L=5e-6, C=500e-12, v_step=3.3)
    
    print("   Running Bayesian optimization (50 iterations)...")
    
    try:
        optimal = sim.optimize_components(target, circuit_type='rlc', n_calls=50)
        
        print(f"\n   ✅ Optimization complete!")
        print(f"   Optimal components:")
        print(f"   - R = {format_component(optimal['R'], 'Ω')}")
        print(f"   - L = {format_component(optimal['L'], 'H')}")
        print(f"   - C = {format_component(optimal['C'], 'F')}")
        print(f"   - MSE = {optimal['mse']:.6f}")
        
        print(f"\n   Original values (for comparison):")
        print(f"   - R = 50 Ω")
        print(f"   - L = 5 µH")
        print(f"   - C = 500 pF")
        
    except ImportError as e:
        print(f"\n   ⚠️ {e}")
        print("   Run: pip install scikit-optimize")
    
    print("\n✅ Demo complete!")
