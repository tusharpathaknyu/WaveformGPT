"""
Tests for WaveformGPT Query Engine and NL Parser.
"""

import pytest
import tempfile

from waveformgpt.vcd_parser import VCDParser
from waveformgpt.query_engine import (
    QueryEngine, Query, QueryType, EventCondition, TemporalConstraint
)
from waveformgpt.nl_parser import NLParser


SAMPLE_VCD = """$date Mon Jan 1 00:00:00 2024 $end
$version Test $end
$timescale 1ns $end
$scope module tb $end
$var wire 1 ! clk $end
$var wire 1 " req $end
$var wire 1 # ack $end
$upscope $end
$enddefinitions $end
$dumpvars
0!
0"
0#
$end
#10
1!
#20
0!
#30
1!
1"
#40
0!
#50
1!
1#
#60
0!
0"
#70
1!
0#
#80
0!
#90
1!
1"
#100
0!
#110
1!
1#
#120
0!
0"
0#
"""


@pytest.fixture
def vcd_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.vcd', delete=False) as f:
        f.write(SAMPLE_VCD)
        return f.name


@pytest.fixture
def engine(vcd_file):
    parser = VCDParser(vcd_file)
    return QueryEngine(parser)


class TestQueryEngine:
    
    def test_find_rising_edge(self, engine):
        """Test finding rising edges."""
        query = Query(
            type=QueryType.FIND_EVENT,
            events=[EventCondition(signal="clk", condition="rise")]
        )
        
        result = engine.execute(query)
        
        assert result.count > 0
        # Clock rises at 10, 30, 50, 70, 90, 110
        assert result.count == 6
    
    def test_find_falling_edge(self, engine):
        """Test finding falling edges."""
        query = Query(
            type=QueryType.FIND_EVENT,
            events=[EventCondition(signal="clk", condition="fall")]
        )
        
        result = engine.execute(query)
        
        # Clock falls at 20, 40, 60, 80, 100, 120
        assert result.count == 6
    
    def test_count_events(self, engine):
        """Test counting events."""
        query = Query(
            type=QueryType.COUNT,
            events=[EventCondition(signal="req", condition="rise")]
        )
        
        result = engine.execute(query)
        
        # req rises at 30 and 90
        assert result.count == 2
    
    def test_find_pattern(self, engine):
        """Test finding temporal pattern."""
        query = Query(
            type=QueryType.FIND_PATTERN,
            events=[
                EventCondition(signal="req", condition="rise"),
                EventCondition(signal="ack", condition="rise")
            ]
        )
        
        result = engine.execute(query)
        
        # req->ack pattern: (30->50) and (90->110)
        assert result.count == 2
    
    def test_find_pattern_with_constraint(self, engine):
        """Test pattern with temporal constraint."""
        query = Query(
            type=QueryType.FIND_PATTERN,
            events=[
                EventCondition(signal="req", condition="rise"),
                EventCondition(signal="ack", condition="rise")
            ],
            temporal=TemporalConstraint(max_cycles=15)
        )
        
        result = engine.execute(query)
        
        # Only patterns within 15 time units
        # req@30 -> ack@50 is 20 units (too long)
        # req@90 -> ack@110 is 20 units (too long)
        assert result.count == 0
    
    def test_measure_timing(self, engine):
        """Test timing measurement."""
        query = Query(
            type=QueryType.MEASURE,
            events=[
                EventCondition(signal="req", condition="rise"),
                EventCondition(signal="ack", condition="rise")
            ]
        )
        
        result = engine.execute(query)
        
        assert result.statistics is not None
        assert "min" in result.statistics
        assert "max" in result.statistics
        assert "avg" in result.statistics
    
    def test_time_range_filter(self, engine):
        """Test filtering by time range."""
        query = Query(
            type=QueryType.FIND_EVENT,
            events=[EventCondition(signal="clk", condition="rise")],
            time_range=(0, 50)
        )
        
        result = engine.execute(query)
        
        # Clock rises at 10, 30, 50 within range
        assert result.count == 3
    
    def test_find_signal_value_at(self, engine):
        """Test getting value at specific time."""
        value = engine.find_signal_value_at("req", 35)
        assert value == "1"  # req went high at 30
        
        value = engine.find_signal_value_at("req", 25)
        assert value == "0"  # req was low before 30


class TestNLParser:
    
    def test_parse_when_query(self):
        """Test parsing 'when' queries."""
        parser = NLParser()
        
        result = parser.parse("when does clk rise")
        
        assert result.query is not None
        assert result.query.type == QueryType.FIND_EVENT
        assert len(result.query.events) == 1
        assert result.query.events[0].signal == "clk"
        assert result.query.events[0].condition == "rise"
    
    def test_parse_find_query(self):
        """Test parsing 'find' queries."""
        parser = NLParser()
        
        result = parser.parse("find all reset falling edges")
        
        assert result.query is not None
        assert result.query.events[0].signal == "reset"
        assert result.query.events[0].condition == "fall"
    
    def test_parse_count_query(self):
        """Test parsing count queries."""
        parser = NLParser()
        
        result = parser.parse("how many times does ack rise")
        
        assert result.query is not None
        assert result.query.type == QueryType.COUNT
        assert result.query.events[0].signal == "ack"
    
    def test_parse_measure_query(self):
        """Test parsing measurement queries."""
        parser = NLParser()
        
        result = parser.parse("measure time between req and ack")
        
        assert result.query is not None
        assert result.query.type == QueryType.MEASURE
        assert len(result.query.events) == 2
    
    def test_parse_pattern_query(self):
        """Test parsing pattern queries."""
        parser = NLParser()
        
        result = parser.parse("find req followed by ack within 10 cycles")
        
        assert result.query is not None
        assert result.query.type == QueryType.FIND_PATTERN
        assert result.query.temporal is not None
        assert result.query.temporal.max_cycles == 10
    
    def test_condition_normalization(self):
        """Test condition normalization."""
        parser = NLParser()
        
        # Various ways to say "rise"
        for phrase in ["rise", "rises", "rising"]:
            result = parser.parse(f"when does clk {phrase}")
            if result.query:
                assert result.query.events[0].condition == "rise"
    
    def test_unknown_query(self):
        """Test handling unknown query format."""
        parser = NLParser()
        
        result = parser.parse("what is the meaning of life")
        
        assert result.query is None
        assert result.confidence == 0.0
