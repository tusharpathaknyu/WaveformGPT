"""Tests for protocol checkers."""

import pytest
from waveformgpt.vcd_parser import VCDParser
from waveformgpt.protocols import (
    AXI4Checker,
    WishboneChecker,
    SPIChecker,
    SignalMapping,
    auto_detect_protocol,
    create_checker,
)


class TestAXI4Checker:
    """Test AXI4 protocol checking."""
    
    @pytest.fixture
    def axi_vcd(self, tmp_path):
        """Create AXI4-like VCD file."""
        vcd_content = """$version Test $end
$timescale 1ns $end
$scope module axi $end
$var wire 1 ! ACLK $end
$var wire 1 " ARESETn $end
$var wire 1 # AWVALID $end
$var wire 1 $ AWREADY $end
$var wire 1 % WVALID $end
$var wire 1 & WREADY $end
$var wire 1 ' BVALID $end
$var wire 1 ( BREADY $end
$enddefinitions $end
$dumpvars
0!
1"
0#
1$
0%
1&
0'
1(
$end
#10
1!
#20
0!
1#
#30
1!
#40
0!
1$
#50
1!
0#
1%
#60
0!
1&
#70
1!
0%
1'
#80
0!
1(
#90
1!
0'
"""
        vcd_path = tmp_path / "axi.vcd"
        vcd_path.write_text(vcd_content)
        return str(vcd_path)
    
    def test_axi4_handshake_check(self, axi_vcd):
        """Test AXI4 handshake checking."""
        parser = VCDParser(axi_vcd)
        
        signal_map = SignalMapping({
            "ACLK": "ACLK",
            "ARESETn": "ARESETn",
            "AWVALID": "AWVALID",
            "AWREADY": "AWREADY",
            "WVALID": "WVALID",
            "WREADY": "WREADY",
            "BVALID": "BVALID",
            "BREADY": "BREADY",
        })
        
        checker = AXI4Checker(parser, signal_map)
        violations = checker.check()
        
        assert checker.protocol_name == "AXI4"
        assert isinstance(violations, list)


class TestWishboneChecker:
    """Test Wishbone protocol checking."""
    
    @pytest.fixture
    def wishbone_vcd(self, tmp_path):
        """Create Wishbone VCD file."""
        vcd_content = """$version Test $end
$timescale 1ns $end
$scope module wb $end
$var wire 1 ! CLK_I $end
$var wire 1 " RST_I $end
$var wire 1 # CYC_O $end
$var wire 1 $ STB_O $end
$var wire 1 % ACK_I $end
$var wire 1 & WE_O $end
$enddefinitions $end
$dumpvars
0!
0"
0#
0$
0%
0&
$end
#10
1!
#20
0!
1#
1$
1&
#30
1!
#40
0!
1%
#50
1!
0$
0#
0%
#60
0!
"""
        vcd_path = tmp_path / "wishbone.vcd"
        vcd_path.write_text(vcd_content)
        return str(vcd_path)
    
    def test_wishbone_cyc_stb_check(self, wishbone_vcd):
        """Test Wishbone CYC/STB relationship."""
        parser = VCDParser(wishbone_vcd)
        
        signal_map = SignalMapping({
            "CLK_I": "CLK_I",
            "RST_I": "RST_I",
            "CYC_O": "CYC_O",
            "STB_O": "STB_O",
            "ACK_I": "ACK_I",
            "WE_O": "WE_O",
        })
        
        checker = WishboneChecker(parser, signal_map)
        violations = checker.check()
        
        assert checker.protocol_name == "Wishbone B4"
        assert isinstance(violations, list)


class TestAutoDetect:
    """Test protocol auto-detection."""
    
    @pytest.fixture
    def axi_like_vcd(self, tmp_path):
        """Create VCD with AXI-like signals."""
        vcd_content = """$version Test $end
$timescale 1ns $end
$scope module top $end
$var wire 1 ! AWVALID $end
$var wire 1 " ARVALID $end
$var wire 1 # WVALID $end
$var wire 1 $ RVALID $end
$enddefinitions $end
$dumpvars
0!
0"
0#
0$
$end
"""
        vcd_path = tmp_path / "axi_like.vcd"
        vcd_path.write_text(vcd_content)
        return str(vcd_path)
    
    def test_auto_detect_axi(self, axi_like_vcd):
        """Test AXI protocol auto-detection."""
        parser = VCDParser(axi_like_vcd)
        protocol = auto_detect_protocol(parser)
        
        assert protocol == "AXI4"
    
    def test_create_checker(self, axi_like_vcd):
        """Test checker factory."""
        parser = VCDParser(axi_like_vcd)
        
        checker = create_checker(parser, "AXI4", {
            "AWVALID": "AWVALID",
            "ARVALID": "ARVALID",
        })
        
        assert checker is not None
        assert isinstance(checker, AXI4Checker)


class TestSPIChecker:
    """Test SPI protocol checking."""
    
    @pytest.fixture
    def spi_vcd(self, tmp_path):
        """Create SPI VCD file."""
        vcd_content = """$version Test $end
$timescale 1ns $end
$scope module spi $end
$var wire 1 ! SCK $end
$var wire 1 " MOSI $end
$var wire 1 # MISO $end
$var wire 1 $ CS $end
$enddefinitions $end
$dumpvars
0!
0"
0#
1$
$end
#10
0$
#20
1!
1"
#30
0!
#40
1!
0"
#50
0!
#60
1$
"""
        vcd_path = tmp_path / "spi.vcd"
        vcd_path.write_text(vcd_content)
        return str(vcd_path)
    
    def test_spi_checker(self, spi_vcd):
        """Test SPI protocol checker."""
        parser = VCDParser(spi_vcd)
        
        signal_map = SignalMapping({
            "SCK": "SCK",
            "MOSI": "MOSI",
            "MISO": "MISO",
            "CS": "CS",
        })
        
        checker = SPIChecker(parser, signal_map)
        violations = checker.check()
        
        assert checker.protocol_name == "SPI"
        assert isinstance(violations, list)
