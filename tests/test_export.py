"""Tests for export functionality."""

import pytest
import json
from pathlib import Path
from waveformgpt.vcd_parser import VCDParser
from waveformgpt.export import (
    export_to_csv,
    export_to_json,
    export_to_markdown,
    export_to_wavedrom,
    export_to_systemverilog,
    export_to_cocotb,
    ExportOptions,
)


class TestExportCSV:
    """Test CSV export."""
    
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
    
    def test_export_csv(self, sample_vcd, tmp_path):
        """Test basic CSV export."""
        parser = VCDParser(sample_vcd)
        output_path = str(tmp_path / "output.csv")
        
        result = export_to_csv(parser, output_path)
        
        assert Path(result).exists()
        content = Path(result).read_text()
        assert "time" in content
        assert "clk" in content
        assert "data" in content
    
    def test_export_csv_with_options(self, sample_vcd, tmp_path):
        """Test CSV export with options."""
        parser = VCDParser(sample_vcd)
        output_path = str(tmp_path / "output.csv")
        
        options = ExportOptions(
            signals=["clk"],
            start_time=10,
            end_time=30
        )
        
        result = export_to_csv(parser, output_path, options)
        
        content = Path(result).read_text()
        assert "clk" in content
        # data should not be included
        lines = content.split("\n")
        assert "data" not in lines[0]  # Not in header


class TestExportJSON:
    """Test JSON export."""
    
    @pytest.fixture
    def sample_vcd(self, tmp_path):
        """Create sample VCD."""
        vcd_content = """$version Test $end
$timescale 1ns $end
$scope module top $end
$var wire 1 ! clk $end
$var wire 8 " data [7:0] $end
$enddefinitions $end
$dumpvars
0!
b00000000 "
$end
#10
1!
b00001111 "
#20
0!
"""
        vcd_path = tmp_path / "test.vcd"
        vcd_path.write_text(vcd_content)
        return str(vcd_path)
    
    def test_export_json(self, sample_vcd, tmp_path):
        """Test JSON export."""
        parser = VCDParser(sample_vcd)
        output_path = str(tmp_path / "output.json")
        
        result = export_to_json(parser, output_path)
        
        assert Path(result).exists()
        content = Path(result).read_text()
        data = json.loads(content)
        
        assert "metadata" in data
        assert "signals" in data
        assert "clk" in data["signals"]
    
    def test_json_structure(self, sample_vcd, tmp_path):
        """Test JSON structure."""
        parser = VCDParser(sample_vcd)
        output_path = str(tmp_path / "output.json")
        
        export_to_json(parser, output_path)
        
        with open(output_path) as f:
            data = json.load(f)
        
        # Check signal structure
        assert "changes" in data["signals"]["clk"]
        assert len(data["signals"]["clk"]["changes"]) > 0


class TestExportMarkdown:
    """Test Markdown export."""
    
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
"""
        vcd_path = tmp_path / "test.vcd"
        vcd_path.write_text(vcd_content)
        return str(vcd_path)
    
    def test_export_markdown(self, sample_vcd):
        """Test Markdown table export."""
        parser = VCDParser(sample_vcd)
        
        md = export_to_markdown(parser, signals=["clk", "data"])
        
        assert "| Signal |" in md
        assert "|--------|" in md
        assert "clk" in md
        assert "data" in md


class TestExportWavedrom:
    """Test WaveDrom export."""
    
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
    
    def test_export_wavedrom(self, sample_vcd):
        """Test WaveDrom JSON export."""
        parser = VCDParser(sample_vcd)
        
        wavedrom = export_to_wavedrom(parser, signals=["clk", "data"])
        
        data = json.loads(wavedrom)
        
        assert "signal" in data
        assert len(data["signal"]) == 2
        assert data["signal"][0]["name"] == "clk"


class TestExportSystemVerilog:
    """Test SystemVerilog export."""
    
    @pytest.fixture
    def sample_vcd(self, tmp_path):
        """Create sample VCD."""
        vcd_content = """$version Test $end
$timescale 1ns $end
$scope module top $end
$var wire 1 ! clk $end
$enddefinitions $end
$dumpvars
0!
$end
#10
1!
#20
0!
#30
1!
"""
        vcd_path = tmp_path / "test.vcd"
        vcd_path.write_text(vcd_content)
        return str(vcd_path)
    
    def test_export_systemverilog(self, sample_vcd):
        """Test SystemVerilog stimulus export."""
        parser = VCDParser(sample_vcd)
        
        sv = export_to_systemverilog(parser, "clk")
        
        assert "// Generated by WaveformGPT" in sv
        assert "initial begin" in sv
        assert "clk =" in sv
        assert "end" in sv


class TestExportCocotb:
    """Test cocotb export."""
    
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
"""
        vcd_path = tmp_path / "test.vcd"
        vcd_path.write_text(vcd_content)
        return str(vcd_path)
    
    def test_export_cocotb(self, sample_vcd):
        """Test cocotb test export."""
        parser = VCDParser(sample_vcd)
        
        code = export_to_cocotb(parser, ["clk", "data"])
        
        assert "# Generated by WaveformGPT" in code
        assert "import cocotb" in code
        assert "@cocotb.test()" in code
        assert "async def replay_waveform" in code
        assert "await Timer" in code
