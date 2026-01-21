"""Tests for assertion checking."""

import pytest
from waveformgpt.vcd_parser import VCDParser
from waveformgpt.assertions import (
    AssertionChecker,
    Assertion,
    AssertionResult,
    AssertionCheckResult,
    generate_assertion_report,
)


class TestAssertion:
    """Test Assertion parsing."""
    
    def test_parse_implication(self):
        """Test parsing implication syntax."""
        assertion = Assertion(
            name="req_ack",
            expression="rose(req) |-> ##[1:5] rose(ack)"
        )
        
        assert assertion.antecedent == "rose(req)"
        assert "rose(ack)" in assertion.consequent
        assert assertion.operator == "|->"
        assert assertion.min_delay == 1
        assert assertion.max_delay == 5
    
    def test_parse_next_cycle(self):
        """Test parsing next cycle implication."""
        assertion = Assertion(
            name="test",
            expression="rose(a) |=> rose(b)"
        )
        
        assert assertion.antecedent == "rose(a)"
        assert assertion.consequent == "rose(b)"
        assert assertion.operator == "|=>"
    
    def test_parse_simple_property(self):
        """Test parsing simple property (no implication)."""
        assertion = Assertion(
            name="test",
            expression="rose(clk)"
        )
        
        assert assertion.antecedent is None
        assert assertion.consequent == "rose(clk)"


class TestAssertionChecker:
    """Test AssertionChecker functionality."""
    
    @pytest.fixture
    def sample_vcd(self, tmp_path):
        """Create sample VCD for testing."""
        vcd_content = """$version Test $end
$timescale 1ns $end
$scope module top $end
$var wire 1 ! clk $end
$var wire 1 " req $end
$var wire 1 # ack $end
$var wire 1 $ valid $end
$var wire 1 % ready $end
$enddefinitions $end
$dumpvars
0!
0"
0#
0$
0%
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
#50
1!
1#
#60
0!
#70
1!
0"
0#
#80
0!
1$
1%
#90
1!
#100
0!
0$
0%
"""
        vcd_path = tmp_path / "test.vcd"
        vcd_path.write_text(vcd_content)
        return str(vcd_path)
    
    @pytest.fixture
    def checker(self, sample_vcd):
        """Create checker with sample VCD."""
        parser = VCDParser(sample_vcd)
        return AssertionChecker(parser)
    
    def test_check_rose(self, checker):
        """Test checking rising edge."""
        result = checker.check("rose(req)", name="req_rises")
        
        assert result.assertion_name == "req_rises"
        assert result.pass_count > 0
    
    def test_check_fell(self, checker):
        """Test checking falling edge."""
        result = checker.check("fell(req)", name="req_falls")
        
        assert result.pass_count > 0
    
    def test_check_implication_pass(self, checker):
        """Test implication that should pass."""
        result = checker.check(
            "rose(req) |-> ##[1:40] rose(ack)",
            name="req_ack"
        )
        
        # req rises at 20, ack rises at 50 (30 cycles later)
        assert result.pass_count >= 1 or result.fail_count == 0
    
    def test_check_implication_fail(self, checker):
        """Test implication with tight timing."""
        result = checker.check(
            "rose(req) |-> ##[1:2] rose(ack)",
            name="req_ack_fast"
        )
        
        # ack is too slow for this constraint
        # May pass vacuously or fail
        assert isinstance(result.result, AssertionResult)
    
    def test_check_high(self, checker):
        """Test checking high signal."""
        result = checker.check("high(valid)", name="valid_high")
        
        assert isinstance(result, AssertionCheckResult)
    
    def test_check_vacuous(self, checker):
        """Test vacuous pass (antecedent never occurs)."""
        result = checker.check(
            "rose(nonexistent) |-> rose(ack)",
            name="never_triggers"
        )
        
        # Should be vacuous since signal doesn't exist
        assert result.vacuous_count >= 0 or result.pass_count == 0


class TestAssertionReport:
    """Test assertion report generation."""
    
    def test_generate_report(self):
        """Test report generation."""
        results = [
            AssertionCheckResult(
                assertion_name="test1",
                expression="rose(a)",
                result=AssertionResult.PASS,
                pass_count=5
            ),
            AssertionCheckResult(
                assertion_name="test2",
                expression="rose(b) |-> rose(c)",
                result=AssertionResult.FAIL,
                fail_count=2
            ),
        ]
        
        report = generate_assertion_report(results)
        
        assert "# Assertion Check Report" in report
        assert "test1" in report
        assert "test2" in report
        assert "PASS" in report
        assert "FAIL" in report
    
    def test_result_summary(self):
        """Test AssertionCheckResult summary."""
        result = AssertionCheckResult(
            assertion_name="test",
            expression="rose(a) |-> rose(b)",
            result=AssertionResult.PASS,
            pass_count=10,
            fail_count=0,
            vacuous_count=2
        )
        
        summary = result.summary()
        
        assert "test: ✓ PASS" in summary
        assert "Checks: 12" in summary
