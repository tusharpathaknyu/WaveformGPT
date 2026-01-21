"""
Tests for WaveformGPT Chat Interface.
"""

import pytest
import tempfile

from waveformgpt.chat import WaveformChat, Message


SAMPLE_VCD = """$date Mon Jan 1 00:00:00 2024 $end
$version Test $end
$timescale 1ns $end
$scope module tb $end
$var wire 1 ! clk $end
$var wire 1 " data $end
$upscope $end
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
#50
1!
"""


@pytest.fixture
def vcd_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.vcd', delete=False) as f:
        f.write(SAMPLE_VCD)
        return f.name


@pytest.fixture
def chat(vcd_file):
    return WaveformChat(vcd_file)


class TestWaveformChat:
    
    def test_load_vcd(self, vcd_file):
        """Test loading VCD file."""
        chat = WaveformChat()
        result = chat.load_vcd(vcd_file)
        
        assert "Loaded" in result
        assert chat.parser is not None
        assert chat.engine is not None
    
    def test_ask_simple_query(self, chat):
        """Test asking a simple query."""
        response = chat.ask("when does clk rise")
        
        assert response.role == "assistant"
        assert response.content is not None
        assert response.result is not None
    
    def test_ask_count_query(self, chat):
        """Test asking a count query."""
        response = chat.ask("how many times does clk rise")
        
        assert "Count" in response.content or "3" in response.content
    
    def test_list_signals(self, chat):
        """Test listing signals."""
        response = chat.ask("signals")
        
        assert "clk" in response.content
        assert "data" in response.content
    
    def test_help_command(self, chat):
        """Test help command."""
        response = chat.ask("help")
        
        assert "Examples" in response.content or "example" in response.content.lower()
    
    def test_history_tracking(self, chat):
        """Test that conversation history is tracked."""
        chat.ask("when does clk rise")
        chat.ask("signals")
        
        assert len(chat.history) == 4  # 2 user + 2 assistant
    
    def test_no_vcd_loaded(self):
        """Test querying without VCD loaded."""
        chat = WaveformChat()
        response = chat.ask("when does clk rise")
        
        assert "load" in response.content.lower()
    
    def test_unknown_query(self, chat):
        """Test handling unknown query."""
        response = chat.ask("gibberish query that makes no sense xyzzy")
        
        assert "couldn't understand" in response.content.lower() or "try" in response.content.lower()


class TestMessage:
    
    def test_message_creation(self):
        """Test message dataclass."""
        msg = Message(role="user", content="test query")
        
        assert msg.role == "user"
        assert msg.content == "test query"
        assert msg.result is None
