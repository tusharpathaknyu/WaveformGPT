"""
Real-World Integration Demo.

Demonstrates WaveformGPT's integration with actual hardware development workflows:
- CI/CD pipelines
- Simulation tools
- Report generation
- Team notifications
"""

import os
import sys
import tempfile
from datetime import datetime

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def demo_ci_integration():
    """Demo: CI/CD Pipeline Integration."""
    print("\n" + "=" * 60)
    print("📋 CI/CD INTEGRATION DEMO")
    print("=" * 60)
    
    from waveformgpt import WaveformChat
    from waveformgpt.integrations import WaveformCI, ReportGenerator
    
    # Create sample VCD
    vcd_content = """$timescale 1ns $end
$scope module test $end
$var wire 1 ! clk $end
$var wire 1 " valid $end
$var wire 1 # ready $end
$var wire 1 $ error $end
$var wire 8 % data [7:0] $end
$upscope $end
$enddefinitions $end
#0
0!
0"
1#
0$
b00000000 %
#10
1!
#20
0!
1"
b11001010 %
#30
1!
1#
#40
0!
#50
1!
0"
#60
0!
#70
1!
1"
b11110000 %
#80
0!
#90
1!
0"
0#
#100
0!
$end
"""
    
    vcd_path = "/tmp/ci_test.vcd"
    with open(vcd_path, 'w') as f:
        f.write(vcd_content)
    
    print("\n1. Setting up CI checks...")
    
    # Create CI checker
    ci = WaveformCI(vcd_path, use_llm=True)
    
    # Add checks
    ci.add_check(
        "clock_present",
        lambda chat: chat.ask("Is there a clock signal? What is its frequency?"),
        lambda r: "clock" in r.lower() or "clk" in r.lower()
    )
    
    ci.add_check(
        "valid_handshake", 
        lambda chat: chat.ask("Describe the valid/ready handshaking pattern"),
    )
    
    ci.add_check(
        "no_errors",
        lambda chat: chat.ask("Does the error signal ever go high?"),
        lambda r: "no" in r.lower() or "never" in r.lower() or "stays low" in r.lower()
    )
    
    print("   ✓ Added 3 checks: clock_present, valid_handshake, no_errors")
    
    # Run checks
    print("\n2. Running CI checks...")
    results = ci.run_all()
    
    # Print report
    ci.print_report(results)
    
    # Generate JUnit XML
    junit_path = "/tmp/waveform-results.xml"
    ci.generate_junit_xml(results, junit_path)
    print(f"\n3. JUnit XML: {junit_path}")
    
    # Generate HTML report
    print("\n4. Generating HTML report...")
    report = ReportGenerator(vcd_path, title="CI Waveform Analysis")
    
    chat = WaveformChat(vcd_path, use_llm=True)
    report.add_section("Overview", chat.ask("Give a brief overview of this waveform"))
    report.add_section("Handshake Analysis", chat.ask("Describe the valid/ready handshake"))
    report.add_ci_results(results)
    report.add_waveform(["clk", "valid", "ready", "data"], title="Key Signals")
    
    html_path = report.generate_html("/tmp/ci_report.html")
    print(f"   ✓ Report: {html_path}")
    
    # Show pass/fail
    passed = all(r.passed for r in results)
    print(f"\n{'✅ CI PASSED' if passed else '❌ CI FAILED'}")
    
    return passed


def demo_regression_testing():
    """Demo: Regression Testing."""
    print("\n" + "=" * 60)
    print("🔄 REGRESSION TESTING DEMO")
    print("=" * 60)
    
    from waveformgpt.regression import WaveformRegression, CoverageAnalyzer, TestSuite
    
    # Create baseline VCD
    baseline_vcd = """$timescale 1ns $end
$scope module tb $end
$var wire 1 ! clk $end
$var wire 8 " count [7:0] $end
$var wire 1 # full $end
$upscope $end
$enddefinitions $end
#0
0!
b00000000 "
0#
#10
1!
#20
0!
b00000001 "
#30
1!
#40
0!
b00000010 "
#50
1!
#60
0!
b00000011 "
#70
1!
#80
0!
b00000100 "
#90
1!
1#
#100
0!
$end
"""
    
    # Create test VCD (with difference)
    test_vcd = """$timescale 1ns $end
$scope module tb $end
$var wire 1 ! clk $end
$var wire 8 " count [7:0] $end
$var wire 1 # full $end
$upscope $end
$enddefinitions $end
#0
0!
b00000000 "
0#
#10
1!
#20
0!
b00000001 "
#30
1!
#40
0!
b00000010 "
#50
1!
#60
0!
b00000011 "
#70
1!
#80
0!
b00000101 "
#90
1!
1#
#100
0!
$end
"""
    
    baseline_path = "/tmp/baseline.vcd"
    test_path = "/tmp/test.vcd"
    
    with open(baseline_path, 'w') as f:
        f.write(baseline_vcd)
    with open(test_path, 'w') as f:
        f.write(test_vcd)
    
    # Compare
    print("\n1. Comparing waveforms...")
    reg = WaveformRegression()
    result = reg.compare(baseline_path, test_path)
    
    print(result.summary)
    
    if result.signal_diffs:
        print("\n   Detailed differences:")
        for diff in result.signal_diffs[:5]:
            print(f"   - {diff.signal}: {diff.description}")
    
    # Coverage analysis
    print("\n2. Coverage Analysis...")
    cov = CoverageAnalyzer(baseline_path)
    cov.add_point("count_nonzero", "count", "value != 0")
    cov.add_point("full_asserted", "full", "value == 1")
    cov.add_toggle_coverage("clk")
    
    cov_result = cov.analyze()
    print(f"   Coverage: {cov_result.coverage_percent:.1f}%")
    print(f"   Hit: {cov_result.hit_points}/{cov_result.total_points}")
    
    if cov_result.uncovered:
        print(f"   Uncovered: {cov_result.uncovered}")
    
    # Test suite
    print("\n3. Test Suite Demo...")
    suite = TestSuite("FIFO Verification")
    
    suite.add_test("counter_basic", baseline_path, [
        ("Does the counter increment?", lambda r: "increment" in r.lower() or "increase" in r.lower()),
        ("Does full ever assert?", lambda r: "yes" in r.lower() or "goes high" in r.lower()),
    ])
    
    # Note: In real usage, these would be different VCD files
    suite_results = suite.run(use_llm=True)
    suite.print_summary(suite_results)


def demo_debugging():
    """Demo: Advanced Debugging."""
    print("\n" + "=" * 60)
    print("🔍 DEBUGGING DEMO")
    print("=" * 60)
    
    from waveformgpt.debug import WaveformDebugger, StateMachineAnalyzer
    
    # Create VCD with issues
    debug_vcd = """$timescale 1ns $end
$scope module fsm $end
$var wire 1 ! clk $end
$var wire 4 " state [3:0] $end
$var wire 1 # error $end
$var wire 1 $ data $end
$upscope $end
$enddefinitions $end
#0
0!
b0000 "
0#
0$
#5
1$
#10
1!
b0001 "
#15
0$
#20
0!
#25
1$
#30
1!
b0010 "
#40
0!
#45
0$
#50
1!
b0100 "
#55
1$
#58
0$
#60
0!
1#
b1111 "
#70
1!
#80
0!
0#
b0000 "
#90
1!
#100
0!
$end
"""
    
    vcd_path = "/tmp/debug.vcd"
    with open(vcd_path, 'w') as f:
        f.write(debug_vcd)
    
    print("\n1. Creating debug session...")
    dbg = WaveformDebugger(vcd_path, use_llm=True)
    
    # Trace error signal
    print("\n2. Tracing error signal...")
    findings = dbg.trace_signal("error", target_value=1)
    for f in findings:
        print(f"   [{f.severity}] t={f.time}: {f.description}")
    
    # Find glitches
    print("\n3. Checking for glitches in data signal...")
    glitches = dbg.find_glitches("data", min_width=5)
    if glitches:
        for g in glitches:
            print(f"   ⚠️  t={g.time}: {g.description}")
    else:
        print("   No significant glitches found")
    
    # Analyze window around error
    print("\n4. Analyzing around error time (t=60)...")
    window = dbg.analyze_window(60, window=20)
    print(f"   {window['summary']}")
    for t in window["transitions"]:
        print(f"   t={t['time']}: {t['signal']} = {t['value']}")
    
    # State machine analysis
    print("\n5. State Machine Analysis...")
    fsm = StateMachineAnalyzer(vcd_path, "state")
    fsm.learn_states()
    
    print(f"   States found: {fsm.states}")
    print(f"   Transitions: {len(fsm.transitions)}")
    
    issues = fsm.find_issues()
    if issues:
        print("   Issues:")
        for issue in issues:
            print(f"   ⚠️  {issue.description}")
    
    fsm.print_summary()
    
    # AI explanation
    print("\n6. AI Debug Explanation...")
    explanation = dbg.explain_failure(error_signal="error", error_time=60)
    print(f"   {explanation[:300]}...")
    
    print(dbg.get_summary())


def demo_simulation_integration():
    """Demo: Simulation Tool Integration (mock)."""
    print("\n" + "=" * 60)
    print("⚡ SIMULATION INTEGRATION DEMO")
    print("=" * 60)
    
    from waveformgpt.integrations import IcarusRunner, SimulationResult
    
    print("""
This demo shows how WaveformGPT integrates with simulation tools.
In a real environment with Icarus Verilog installed:

    from waveformgpt.integrations import IcarusRunner
    
    runner = IcarusRunner()
    result = runner.run(
        sources=["rtl/counter.v", "tb/tb_counter.v"],
        top="tb_counter"
    )
    
    if result.success:
        chat = WaveformChat(result.vcd_file, use_llm=True)
        print(chat.ask("Did the counter reach its maximum value?"))

Supported simulators:
  - Icarus Verilog (IcarusRunner)
  - Verilator (VerilatorRunner)
  - cocotb (CocotbRunner)
  - Commercial tools via command-line wrappers
""")


def demo_notifications():
    """Demo: Team Notifications."""
    print("\n" + "=" * 60)
    print("📣 NOTIFICATION DEMO")
    print("=" * 60)
    
    print("""
WaveformGPT can send notifications to your team:

    from waveformgpt.integrations import SlackNotifier, GitHubIntegration
    
    # Slack notifications
    slack = SlackNotifier("https://hooks.slack.com/services/...")
    slack.send_violation(
        title="AXI Protocol Violation",
        details="AWVALID held without AWREADY for 100 cycles",
        vcd_file="simulation.vcd",
        time=1523
    )
    slack.send_ci_results(ci_results, "simulation.vcd")
    
    # GitHub issue creation
    gh = GitHubIntegration(token="ghp_...", repo="user/hardware-project")
    gh.create_violation_issue(
        violation_type="Setup Timing Violation",
        details="Data changed 0.5ns before clock edge",
        vcd_file="simulation.vcd",
        time=1523
    )
""")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("🌊 WAVEFORMGPT REAL-WORLD INTEGRATION DEMOS")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        demo_ci_integration()
    except Exception as e:
        print(f"CI Demo error: {e}")
    
    try:
        demo_regression_testing()
    except Exception as e:
        print(f"Regression Demo error: {e}")
    
    try:
        demo_debugging()
    except Exception as e:
        print(f"Debug Demo error: {e}")
    
    demo_simulation_integration()
    demo_notifications()
    
    print("\n" + "=" * 60)
    print("✅ DEMOS COMPLETE")
    print("=" * 60)
    print("""
What you can do with these integrations:

🔧 CI/CD:
   waveformgpt ci simulation.vcd --checks checks.yaml --junit results.xml

🔄 Regression:
   waveformgpt compare golden.vcd new.vcd --report

🔍 Debug:
   waveformgpt debug simulation.vcd --trace error --at 1523

📊 Reports:
   waveformgpt visualize sim.vcd --format html -o report.html

📣 Notifications:
   Configure Slack/GitHub webhooks for automatic alerts
""")


if __name__ == "__main__":
    main()
