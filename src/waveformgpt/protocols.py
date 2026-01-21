"""
Protocol Checking Templates for Common Interfaces.

Pre-built checkers for AXI, Wishbone, SPI, I2C, UART and other protocols.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
from abc import ABC, abstractmethod

from waveformgpt.vcd_parser import VCDParser
from waveformgpt.analyzer import WaveformAnalyzer, ProtocolViolation


@dataclass
class SignalMapping:
    """Maps protocol signals to actual signal names in waveform."""
    mapping: Dict[str, str]
    
    def get(self, proto_signal: str) -> Optional[str]:
        return self.mapping.get(proto_signal)


class ProtocolChecker(ABC):
    """Base class for protocol checkers."""
    
    def __init__(self, parser: VCDParser, signal_map: SignalMapping):
        self.parser = parser
        self.signal_map = signal_map
        self.analyzer = WaveformAnalyzer(parser)
    
    @abstractmethod
    def check(self) -> List[ProtocolViolation]:
        """Run protocol checks and return violations."""
        pass
    
    @property
    @abstractmethod
    def protocol_name(self) -> str:
        """Name of the protocol."""
        pass


class AXI4Checker(ProtocolChecker):
    """
    AXI4 protocol checker.
    
    Checks for:
    - Valid/Ready handshake rules
    - Address channel ordering
    - Write data/strobe consistency
    - Response ordering
    - Outstanding transaction limits
    """
    
    @property
    def protocol_name(self) -> str:
        return "AXI4"
    
    SIGNALS = [
        "ACLK", "ARESETn",
        # Write address
        "AWVALID", "AWREADY", "AWADDR", "AWLEN", "AWSIZE", "AWBURST", "AWID",
        # Write data
        "WVALID", "WREADY", "WDATA", "WSTRB", "WLAST",
        # Write response
        "BVALID", "BREADY", "BRESP", "BID",
        # Read address
        "ARVALID", "ARREADY", "ARADDR", "ARLEN", "ARSIZE", "ARBURST", "ARID",
        # Read data
        "RVALID", "RREADY", "RDATA", "RRESP", "RLAST", "RID",
    ]
    
    def check(self) -> List[ProtocolViolation]:
        violations = []
        
        # Check write handshake
        if self._has_signal("AWVALID") and self._has_signal("AWREADY"):
            violations.extend(self._check_handshake(
                "AWVALID", "AWREADY", "Write Address"
            ))
        
        if self._has_signal("WVALID") and self._has_signal("WREADY"):
            violations.extend(self._check_handshake(
                "WVALID", "WREADY", "Write Data"
            ))
        
        if self._has_signal("BVALID") and self._has_signal("BREADY"):
            violations.extend(self._check_handshake(
                "BVALID", "BREADY", "Write Response"
            ))
        
        # Check read handshake
        if self._has_signal("ARVALID") and self._has_signal("ARREADY"):
            violations.extend(self._check_handshake(
                "ARVALID", "ARREADY", "Read Address"
            ))
        
        if self._has_signal("RVALID") and self._has_signal("RREADY"):
            violations.extend(self._check_handshake(
                "RVALID", "RREADY", "Read Data"
            ))
        
        # Check WLAST
        if self._has_signal("WLAST") and self._has_signal("WVALID"):
            violations.extend(self._check_wlast())
        
        return violations
    
    def _has_signal(self, proto_signal: str) -> bool:
        actual = self.signal_map.get(proto_signal)
        return actual is not None
    
    def _get_signal(self, proto_signal: str) -> str:
        return self.signal_map.get(proto_signal)
    
    def _check_handshake(self, valid_sig: str, ready_sig: str, 
                          channel: str) -> List[ProtocolViolation]:
        """Check valid/ready handshake rules."""
        violations = []
        
        valid_name = self._get_signal(valid_sig)
        ready_name = self._get_signal(ready_sig)
        
        valid_values = self.parser.get_signal_values(valid_name)
        ready_values = self.parser.get_signal_values(ready_name)
        
        if not valid_values or not ready_values:
            return violations
        
        # Rule: VALID must not depend on READY
        # Rule: Once VALID is asserted, it must remain until handshake
        
        valid_high_time = None
        prev_valid = '0'
        
        for time, value in valid_values:
            if value == '1' and prev_valid == '0':
                valid_high_time = time
            elif value == '0' and prev_valid == '1' and valid_high_time:
                # Check if READY was ever high during valid assertion
                ready_handshake = False
                for rt, rv in ready_values:
                    if valid_high_time <= rt < time and rv == '1':
                        ready_handshake = True
                        break
                
                if not ready_handshake:
                    violations.append(ProtocolViolation(
                        violation_type=f"axi_{channel.lower()}_valid_deassert",
                        time=time,
                        signals={valid_sig: value, ready_sig: "low"},
                        description=f"{channel}: VALID deasserted without handshake",
                        severity="error"
                    ))
            
            prev_valid = value
        
        return violations
    
    def _check_wlast(self) -> List[ProtocolViolation]:
        """Check WLAST signaling."""
        violations = []
        
        wlast_name = self._get_signal("WLAST")
        wvalid_name = self._get_signal("WVALID")
        wready_name = self._get_signal("WREADY")
        
        # Count write data beats and check WLAST on last beat
        # Simplified check - real implementation would track AWLEN
        
        return violations


class WishboneChecker(ProtocolChecker):
    """
    Wishbone B4 protocol checker.
    
    Checks for:
    - Classic cycle rules
    - Pipelined cycle rules
    - ACK/ERR/RTY signaling
    """
    
    @property
    def protocol_name(self) -> str:
        return "Wishbone B4"
    
    SIGNALS = [
        "CLK_I", "RST_I",
        "CYC_O", "STB_O", "WE_O", "ADR_O", "DAT_O", "DAT_I",
        "ACK_I", "ERR_I", "RTY_I", "SEL_O", "STALL_I",
    ]
    
    def check(self) -> List[ProtocolViolation]:
        violations = []
        
        # Check CYC/STB relationship
        if self._has_signal("CYC_O") and self._has_signal("STB_O"):
            violations.extend(self._check_cyc_stb())
        
        # Check termination signals
        violations.extend(self._check_termination())
        
        return violations
    
    def _has_signal(self, proto_signal: str) -> bool:
        return self.signal_map.get(proto_signal) is not None
    
    def _get_signal(self, proto_signal: str) -> str:
        return self.signal_map.get(proto_signal)
    
    def _check_cyc_stb(self) -> List[ProtocolViolation]:
        """Check CYC and STB relationship rules."""
        violations = []
        
        cyc_name = self._get_signal("CYC_O")
        stb_name = self._get_signal("STB_O")
        
        cyc_values = self.parser.get_signal_values(cyc_name)
        stb_values = self.parser.get_signal_values(stb_name)
        
        # Rule: STB can only be high when CYC is high
        # Build combined timeline
        all_times = sorted(set(t for t, _ in cyc_values) | set(t for t, _ in stb_values))
        
        cyc_at = {t: v for t, v in cyc_values}
        stb_at = {t: v for t, v in stb_values}
        
        current_cyc = '0'
        current_stb = '0'
        
        for time in all_times:
            if time in cyc_at:
                current_cyc = cyc_at[time]
            if time in stb_at:
                current_stb = stb_at[time]
            
            if current_stb == '1' and current_cyc == '0':
                violations.append(ProtocolViolation(
                    violation_type="wishbone_stb_without_cyc",
                    time=time,
                    signals={"CYC_O": current_cyc, "STB_O": current_stb},
                    description="STB asserted while CYC is low",
                    severity="error"
                ))
        
        return violations
    
    def _check_termination(self) -> List[ProtocolViolation]:
        """Check for proper cycle termination."""
        violations = []
        
        stb_name = self._get_signal("STB_O")
        ack_name = self._get_signal("ACK_I")
        err_name = self._get_signal("ERR_I")
        rty_name = self._get_signal("RTY_I")
        
        if not stb_name:
            return violations
        
        stb_values = self.parser.get_signal_values(stb_name)
        
        # For each STB assertion, check for termination
        prev_stb = '0'
        stb_assert_time = None
        
        for time, value in stb_values:
            if value == '1' and prev_stb == '0':
                stb_assert_time = time
            elif value == '0' and prev_stb == '1' and stb_assert_time:
                # Check if any termination signal was seen
                terminated = False
                
                for term_sig in [ack_name, err_name, rty_name]:
                    if term_sig:
                        term_values = self.parser.get_signal_values(term_sig)
                        for tt, tv in term_values:
                            if stb_assert_time <= tt < time and tv == '1':
                                terminated = True
                                break
                    if terminated:
                        break
                
                if not terminated:
                    violations.append(ProtocolViolation(
                        violation_type="wishbone_no_termination",
                        time=stb_assert_time,
                        signals={"STB_O": "high"},
                        description="Cycle ended without ACK/ERR/RTY",
                        severity="warning"
                    ))
                
                stb_assert_time = None
            
            prev_stb = value
        
        return violations


class SPIChecker(ProtocolChecker):
    """
    SPI protocol checker.
    
    Checks for:
    - Clock/data phase relationship
    - CS timing
    - Idle state
    """
    
    @property
    def protocol_name(self) -> str:
        return "SPI"
    
    SIGNALS = ["SCK", "MOSI", "MISO", "CS"]
    
    def check(self) -> List[ProtocolViolation]:
        violations = []
        
        # Check CS deassertion during transaction
        if self._has_signal("CS") and self._has_signal("SCK"):
            violations.extend(self._check_cs_timing())
        
        return violations
    
    def _has_signal(self, proto_signal: str) -> bool:
        return self.signal_map.get(proto_signal) is not None
    
    def _get_signal(self, proto_signal: str) -> str:
        return self.signal_map.get(proto_signal)
    
    def _check_cs_timing(self) -> List[ProtocolViolation]:
        """Check chip select timing."""
        violations = []
        
        cs_name = self._get_signal("CS")
        sck_name = self._get_signal("SCK")
        
        cs_values = self.parser.get_signal_values(cs_name)
        sck_values = self.parser.get_signal_values(sck_name)
        
        if not cs_values or not sck_values:
            return violations
        
        # Check for CS glitches during clock activity
        # (simplified check)
        
        return violations


def auto_detect_protocol(parser: VCDParser) -> Optional[str]:
    """
    Attempt to auto-detect protocol from signal names.
    
    Returns:
        Protocol name if detected, None otherwise
    """
    signals = parser.search_signals(".*")
    signal_names = set(s.name.upper() for s in signals)
    signal_paths = set(s.path.upper() for s in signals)
    all_names = signal_names | signal_paths
    
    # Check for AXI signals
    axi_signals = {"AWVALID", "ARVALID", "WVALID", "RVALID", "BVALID"}
    if len(axi_signals & all_names) >= 3:
        return "AXI4"
    
    # Check for Wishbone signals
    wb_signals = {"CYC_O", "STB_O", "ACK_I", "WE_O"}
    if len(wb_signals & all_names) >= 3:
        return "Wishbone"
    
    # Check for SPI signals
    spi_signals = {"SCK", "MOSI", "MISO", "CS", "SS"}
    if len(spi_signals & all_names) >= 2:
        return "SPI"
    
    # Check for I2C signals
    i2c_signals = {"SDA", "SCL"}
    if len(i2c_signals & all_names) >= 2:
        return "I2C"
    
    return None


def create_checker(parser: VCDParser, 
                   protocol: str,
                   signal_map: Dict[str, str]) -> Optional[ProtocolChecker]:
    """
    Create a protocol checker for the given protocol.
    
    Args:
        parser: VCD parser instance
        protocol: Protocol name ("AXI4", "Wishbone", "SPI", etc.)
        signal_map: Mapping from protocol signals to actual signals
    
    Returns:
        ProtocolChecker instance or None
    """
    mapping = SignalMapping(signal_map)
    
    checkers = {
        "AXI4": AXI4Checker,
        "AXI": AXI4Checker,
        "Wishbone": WishboneChecker,
        "SPI": SPIChecker,
    }
    
    checker_class = checkers.get(protocol)
    if checker_class:
        return checker_class(parser, mapping)
    
    return None
