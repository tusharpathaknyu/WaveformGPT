"""Tests for visualization module."""

import pytest
from waveformgpt.vcd_parser import VCDParser
from waveformgpt.visualize import (
    ASCIIWaveform,
    WaveformStyle,
    render_to_html,
)


class TestASCIIWaveform:
    """Test ASCII waveform rendering."""
    
    @pytest.fixture
    def sample_vcd(self, tmp_path):
        """Create sample VCD for testing."""
        vcd_content = """$version Test $end
$timescale 1ns $end
$scope module top $end
$var wire 1 ! clk $end
$var wire 1 " data $end
$var wire 8 # bus [7:0] $end
$enddefinitions $end
$dumpvars
0!
0"
b00000000 #
$end
#10
1!
#20
0!
1"
b00001111 #
#30
1!
#40
0!
0"
b11110000 #
#50
1!
b11111111 #
"""
        vcd_path = tmp_path / "test.vcd"
        vcd_path.write_text(vcd_content)
        return str(vcd_path)
    
    @pytest.fixture
    def ascii_renderer(self, sample_vcd):
        """Create ASCII waveform renderer."""
        parser = VCDParser(sample_vcd)
        return ASCIIWaveform(parser)
    
    def test_render_bit_signal(self, ascii_renderer):
        """Test rendering single-bit signal."""
        output = ascii_renderer.render_signal("clk", width=60)
        
        assert "clk:" in output
        assert len(output) > 0
    
    def test_render_bus_signal(self, ascii_renderer):
        """Test rendering multi-bit bus."""
        output = ascii_renderer.render_signal("bus", width=60)
        
        assert "bus:" in output or "bus" in output
    
    def test_render_with_time_window(self, ascii_renderer):
        """Test rendering with time window."""
        output = ascii_renderer.render_signal(
            "clk",
            start_time=10,
            end_time=40,
            width=40
        )
        
        assert "clk:" in output
    
    def test_render_multiple_signals(self, ascii_renderer):
        """Test rendering multiple signals."""
        output = ascii_renderer.render_multiple(
            ["clk", "data"],
            width=60
        )
        
        assert "clk" in output
        assert "data" in output
    
    def test_render_nonexistent_signal(self, ascii_renderer):
        """Test rendering non-existent signal."""
        output = ascii_renderer.render_signal("nonexistent")
        
        assert "no data" in output.lower()
    
    def test_custom_style(self, sample_vcd):
        """Test custom waveform style."""
        parser = VCDParser(sample_vcd)
        style = WaveformStyle(
            width=40,
            high_char="#",
            low_char=".",
            rise_char="+",
            fall_char="-"
        )
        renderer = ASCIIWaveform(parser, style)
        
        output = renderer.render_signal("clk")
        
        assert "clk:" in output


class TestHTMLRender:
    """Test HTML waveform rendering."""
    
    @pytest.fixture
    def sample_vcd(self, tmp_path):
        """Create sample VCD."""
        vcd_content = """$version Test $end
$timescale 1ns $end
$scope module top $end
$var wire 1 ! clk $end
$var wire 1 " data $end
$enddefinitions $end
$dumpvars
0!
0"
$end
#10
1!
#20
0!
1"
#30
1!
#40
0!
0"
"""
        vcd_path = tmp_path / "test.vcd"
        vcd_path.write_text(vcd_content)
        return str(vcd_path)
    
    def test_render_to_html(self, sample_vcd):
        """Test HTML rendering."""
        parser = VCDParser(sample_vcd)
        
        html = render_to_html(parser, ["clk", "data"])
        
        assert "<!DOCTYPE html>" in html
        assert "WaveformGPT" in html
        assert "clk" in html
        assert "data" in html
    
    def test_render_html_to_file(self, sample_vcd, tmp_path):
        """Test saving HTML to file."""
        parser = VCDParser(sample_vcd)
        output_path = str(tmp_path / "waveform.html")
        
        render_to_html(parser, ["clk"], output_path=output_path)
        
        from pathlib import Path
        assert Path(output_path).exists()
        content = Path(output_path).read_text()
        assert "<!DOCTYPE html>" in content
    
    def test_render_html_with_window(self, sample_vcd):
        """Test HTML rendering with time window."""
        parser = VCDParser(sample_vcd)
        
        html = render_to_html(
            parser,
            ["clk"],
            start_time=10,
            end_time=30
        )
        
        assert "<!DOCTYPE html>" in html


class TestWaveformStyle:
    """Test WaveformStyle configuration."""
    
    def test_default_style(self):
        """Test default style values."""
        style = WaveformStyle()
        
        assert style.width == 80
        assert style.height == 3
        assert style.time_unit == "ns"
    
    def test_custom_style(self):
        """Test custom style values."""
        style = WaveformStyle(
            width=120,
            height=5,
            high_char="*",
            low_char="-",
            time_unit="ps"
        )
        
        assert style.width == 120
        assert style.height == 5
        assert style.high_char == "*"
        assert style.low_char == "-"
        assert style.time_unit == "ps"
