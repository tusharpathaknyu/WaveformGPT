"""
Real-World Integrations for WaveformGPT.

Connects WaveformGPT to actual hardware development workflows:
- Simulation tools (Verilator, Icarus, cocotb)
- CI/CD pipelines (GitHub Actions, Jenkins)
- Issue trackers (GitHub, Jira)
- Team notifications (Slack, Teams)
- Waveform viewers (GTKWave)
- Report generation (HTML, PDF)
"""

import os
import json
import subprocess
import tempfile
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import urllib.request
import urllib.parse


# =============================================================================
# SIMULATION TOOL INTEGRATION
# =============================================================================

@dataclass
class SimulationResult:
    """Result from running a simulation."""
    success: bool
    vcd_file: Optional[str] = None
    log_file: Optional[str] = None
    duration_seconds: float = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class VerilatorRunner:
    """
    Run Verilator simulations and capture VCD output.
    
    Usage:
        runner = VerilatorRunner("rtl/")
        result = runner.run("top_module", testbench="tb_top.cpp")
        
        chat = WaveformChat(result.vcd_file, use_llm=True)
        chat.ask("Did the test pass?")
    """
    
    def __init__(self, rtl_dir: str, build_dir: str = "obj_dir"):
        self.rtl_dir = Path(rtl_dir)
        self.build_dir = Path(build_dir)
    
    def compile(self, top_module: str, 
                sources: Optional[List[str]] = None,
                trace: bool = True,
                extra_args: List[str] = None) -> bool:
        """Compile design with Verilator."""
        
        cmd = ["verilator", "--cc"]
        
        if trace:
            cmd.append("--trace")
        
        cmd.extend(["--exe", "--build"])
        cmd.extend(["-Mdir", str(self.build_dir)])
        
        if extra_args:
            cmd.extend(extra_args)
        
        # Add sources
        if sources:
            cmd.extend(sources)
        else:
            cmd.extend([str(f) for f in self.rtl_dir.glob("*.v")])
            cmd.extend([str(f) for f in self.rtl_dir.glob("*.sv")])
        
        cmd.append(top_module)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    
    def run(self, top_module: str, 
            vcd_output: Optional[str] = None,
            timeout: int = 300) -> SimulationResult:
        """Run compiled simulation."""
        
        exe_path = self.build_dir / f"V{top_module}"
        
        if not exe_path.exists():
            return SimulationResult(
                success=False,
                errors=[f"Executable not found: {exe_path}"]
            )
        
        vcd_file = vcd_output or tempfile.mktemp(suffix=".vcd")
        
        start = datetime.now()
        
        try:
            result = subprocess.run(
                [str(exe_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "TRACE_FILE": vcd_file}
            )
            
            duration = (datetime.now() - start).total_seconds()
            
            return SimulationResult(
                success=result.returncode == 0,
                vcd_file=vcd_file if os.path.exists(vcd_file) else None,
                duration_seconds=duration,
                errors=result.stderr.split('\n') if result.returncode != 0 else [],
            )
            
        except subprocess.TimeoutExpired:
            return SimulationResult(
                success=False,
                errors=["Simulation timed out"]
            )


class IcarusRunner:
    """
    Run Icarus Verilog simulations.
    
    Usage:
        runner = IcarusRunner()
        result = runner.run(["design.v", "tb.v"], top="tb")
        
        chat = WaveformChat(result.vcd_file, use_llm=True)
    """
    
    def __init__(self):
        pass
    
    def run(self, sources: List[str], 
            top: Optional[str] = None,
            vcd_output: Optional[str] = None,
            defines: Dict[str, str] = None,
            timeout: int = 300) -> SimulationResult:
        """Compile and run simulation."""
        
        vvp_file = tempfile.mktemp(suffix=".vvp")
        vcd_file = vcd_output or tempfile.mktemp(suffix=".vcd")
        
        # Compile
        cmd = ["iverilog", "-o", vvp_file]
        
        if top:
            cmd.extend(["-s", top])
        
        if defines:
            for key, val in defines.items():
                cmd.append(f"-D{key}={val}")
        
        # Add VCD output define
        cmd.append(f"-DVCD_FILE=\"{vcd_file}\"")
        
        cmd.extend(sources)
        
        compile_result = subprocess.run(cmd, capture_output=True, text=True)
        
        if compile_result.returncode != 0:
            return SimulationResult(
                success=False,
                errors=compile_result.stderr.split('\n')
            )
        
        # Run
        start = datetime.now()
        
        try:
            run_result = subprocess.run(
                ["vvp", vvp_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            duration = (datetime.now() - start).total_seconds()
            
            return SimulationResult(
                success=run_result.returncode == 0,
                vcd_file=vcd_file if os.path.exists(vcd_file) else None,
                duration_seconds=duration,
                errors=run_result.stderr.split('\n') if run_result.returncode != 0 else [],
            )
            
        except subprocess.TimeoutExpired:
            return SimulationResult(
                success=False,
                errors=["Simulation timed out"]
            )
        finally:
            try:
                os.remove(vvp_file)
            except:
                pass


class CocotbRunner:
    """
    Run cocotb testbenches and analyze results.
    
    Usage:
        runner = CocotbRunner("tests/")
        result = runner.run("test_fifo", dut="fifo")
        
        chat = WaveformChat(result.vcd_file, use_llm=True)
        chat.ask("Why did test_overflow fail?")
    """
    
    def __init__(self, test_dir: str):
        self.test_dir = Path(test_dir)
    
    def run(self, test_module: str,
            dut: str,
            simulator: str = "icarus",
            waves: bool = True,
            timeout: int = 600) -> SimulationResult:
        """Run cocotb test."""
        
        env = {
            **os.environ,
            "MODULE": test_module,
            "TOPLEVEL": dut,
            "TOPLEVEL_LANG": "verilog",
            "SIM": simulator,
        }
        
        if waves:
            env["WAVES"] = "1"
        
        vcd_file = str(self.test_dir / f"{dut}.vcd")
        
        start = datetime.now()
        
        result = subprocess.run(
            ["make", "-C", str(self.test_dir)],
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout
        )
        
        duration = (datetime.now() - start).total_seconds()
        
        # Parse cocotb results
        errors = []
        warnings = []
        
        for line in result.stdout.split('\n'):
            if 'FAIL' in line:
                errors.append(line)
            if 'WARN' in line:
                warnings.append(line)
        
        return SimulationResult(
            success=result.returncode == 0 and not errors,
            vcd_file=vcd_file if os.path.exists(vcd_file) else None,
            duration_seconds=duration,
            errors=errors,
            warnings=warnings,
        )


# =============================================================================
# CI/CD INTEGRATION
# =============================================================================

@dataclass
class CICheckResult:
    """Result of a CI waveform check."""
    passed: bool
    check_name: str
    summary: str
    details: List[str] = field(default_factory=list)
    violations: List[Dict] = field(default_factory=list)
    duration_seconds: float = 0


class WaveformCI:
    """
    CI/CD integration for waveform analysis.
    
    Use in GitHub Actions, Jenkins, GitLab CI, etc.
    
    Usage:
        ci = WaveformCI("simulation.vcd")
        
        # Run checks
        ci.add_check("protocol", lambda chat: chat.ask("Any AXI protocol violations?"))
        ci.add_check("timing", lambda chat: chat.ask("Any setup/hold violations?"))
        
        # Run and get exit code
        results = ci.run_all()
        sys.exit(0 if all(r.passed for r in results) else 1)
    """
    
    def __init__(self, vcd_file: str, use_llm: bool = True):
        from waveformgpt import WaveformChat
        
        self.vcd_file = vcd_file
        self.chat = WaveformChat(vcd_file, use_llm=use_llm)
        self.checks: List[tuple] = []
    
    def add_check(self, name: str, 
                  check_fn: Callable,
                  pass_condition: Optional[Callable[[str], bool]] = None):
        """
        Add a check to run.
        
        Args:
            name: Check name
            check_fn: Function that takes WaveformChat and returns result
            pass_condition: Function that takes result and returns bool
        """
        self.checks.append((name, check_fn, pass_condition))
    
    def add_protocol_check(self, protocol: str, signal_prefix: str = ""):
        """Add a protocol compliance check."""
        def check(chat):
            return chat.ask(f"Check {protocol} protocol compliance for signals with prefix '{signal_prefix}'")
        
        def passes(result):
            result_lower = result.lower()
            return "no violation" in result_lower or "compliant" in result_lower
        
        self.checks.append((f"{protocol}_protocol", check, passes))
    
    def add_assertion_check(self, assertion: str, name: Optional[str] = None):
        """Add a temporal assertion check."""
        check_name = name or f"assertion_{len(self.checks)}"
        
        def check(chat):
            from waveformgpt import AssertionChecker
            checker = AssertionChecker(chat.parser)
            result = checker.check(assertion)
            return f"Passed: {result.passed}, Failures: {len(result.failures)}"
        
        def passes(result):
            return "Passed: True" in result
        
        self.checks.append((check_name, check, passes))
    
    def run_all(self) -> List[CICheckResult]:
        """Run all checks and return results."""
        results = []
        
        for name, check_fn, pass_condition in self.checks:
            start = datetime.now()
            
            try:
                response = check_fn(self.chat)
                if hasattr(response, 'content'):
                    result_str = response.content
                else:
                    result_str = str(response)
                
                # Determine if passed
                if pass_condition:
                    passed = pass_condition(result_str)
                else:
                    # Default: check for negative keywords
                    result_lower = result_str.lower()
                    passed = not any(word in result_lower for word in [
                        'violation', 'error', 'fail', 'issue', 'problem'
                    ])
                
                duration = (datetime.now() - start).total_seconds()
                
                results.append(CICheckResult(
                    passed=passed,
                    check_name=name,
                    summary=result_str[:200] + "..." if len(result_str) > 200 else result_str,
                    details=[result_str],
                    duration_seconds=duration,
                ))
                
            except Exception as e:
                results.append(CICheckResult(
                    passed=False,
                    check_name=name,
                    summary=f"Check failed with error: {str(e)}",
                    duration_seconds=(datetime.now() - start).total_seconds(),
                ))
        
        return results
    
    def print_report(self, results: List[CICheckResult]):
        """Print CI report to stdout."""
        print("\n" + "=" * 60)
        print("🔍 WaveformGPT CI Report")
        print("=" * 60)
        print(f"VCD File: {self.vcd_file}")
        print(f"Checks: {len(results)}")
        print()
        
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        
        for r in results:
            status = "✅ PASS" if r.passed else "❌ FAIL"
            print(f"{status} | {r.check_name} ({r.duration_seconds:.2f}s)")
            print(f"       {r.summary}")
            print()
        
        print("-" * 60)
        print(f"Results: {passed}/{len(results)} passed")
        
        if failed > 0:
            print(f"⚠️  {failed} check(s) failed!")
            return 1
        else:
            print("✅ All checks passed!")
            return 0
    
    def generate_junit_xml(self, results: List[CICheckResult], 
                           output_file: str = "waveform-results.xml"):
        """Generate JUnit XML report for CI systems."""
        xml = ['<?xml version="1.0" encoding="UTF-8"?>']
        xml.append(f'<testsuite name="WaveformGPT" tests="{len(results)}">')
        
        for r in results:
            xml.append(f'  <testcase name="{r.check_name}" time="{r.duration_seconds:.3f}">')
            if not r.passed:
                xml.append(f'    <failure message="{r.summary[:100]}">')
                for detail in r.details:
                    xml.append(f'      {detail}')
                xml.append('    </failure>')
            xml.append('  </testcase>')
        
        xml.append('</testsuite>')
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(xml))
        
        return output_file


# =============================================================================
# GTKWAVE INTEGRATION
# =============================================================================

class GTKWaveIntegration:
    """
    Open waveforms in GTKWave with markers and signal selection.
    
    Usage:
        gtk = GTKWaveIntegration("simulation.vcd")
        
        # Open at specific time
        gtk.open_at_time(1523)
        
        # Open with markers from query results
        gtk.open_with_results(chat.ask("When does error go high?"))
    """
    
    def __init__(self, vcd_file: str):
        self.vcd_file = vcd_file
    
    def generate_save_file(self, 
                           signals: List[str],
                           markers: List[int] = None,
                           zoom_start: int = None,
                           zoom_end: int = None,
                           output: str = None) -> str:
        """Generate GTKWave save file."""
        
        output = output or self.vcd_file.replace('.vcd', '.gtkw')
        
        lines = [
            f"[dumpfile] \"{self.vcd_file}\"",
            "[dumpfile_mtime] \"\"",
            "[dumpfile_size] 0",
            "[savefile] \"\"",
        ]
        
        if zoom_start is not None and zoom_end is not None:
            lines.append(f"[timestart] {zoom_start}")
            lines.append(f"[size] 1200 600")
            lines.append(f"[pos] 0 0")
        
        # Add signals
        for sig in signals:
            lines.append(f"@28")  # Binary display
            lines.append(f"{sig}")
        
        # Add markers
        if markers:
            for i, time in enumerate(markers[:26]):  # GTKWave supports A-Z markers
                marker_letter = chr(ord('A') + i)
                lines.append(f"[markername] \"{marker_letter}\" {time}")
        
        with open(output, 'w') as f:
            f.write('\n'.join(lines))
        
        return output
    
    def open(self, signals: List[str] = None, 
             markers: List[int] = None,
             zoom_range: tuple = None):
        """Open GTKWave with specified view."""
        
        if signals or markers or zoom_range:
            save_file = self.generate_save_file(
                signals=signals or [],
                markers=markers,
                zoom_start=zoom_range[0] if zoom_range else None,
                zoom_end=zoom_range[1] if zoom_range else None,
            )
            subprocess.Popen(["gtkwave", save_file])
        else:
            subprocess.Popen(["gtkwave", self.vcd_file])
    
    def open_at_time(self, time: int, window: int = 100, signals: List[str] = None):
        """Open GTKWave centered at a specific time."""
        self.open(
            signals=signals,
            markers=[time],
            zoom_range=(time - window, time + window)
        )
    
    def open_with_results(self, query_result):
        """Open GTKWave with query result markers."""
        if hasattr(query_result, 'result') and query_result.result:
            matches = query_result.result.matches or []
            markers = [m.time for m in matches[:20]]
            signals = []
            if matches:
                signals = list(matches[0].signals.keys())
            self.open(signals=signals, markers=markers)
        else:
            self.open()


# =============================================================================
# NOTIFICATION INTEGRATIONS
# =============================================================================

class SlackNotifier:
    """
    Send waveform analysis results to Slack.
    
    Usage:
        slack = SlackNotifier(webhook_url="https://hooks.slack.com/...")
        
        # Send alert
        slack.send_violation(
            title="AXI Protocol Violation Detected",
            details="AWVALID held high without AWREADY response for 100 cycles",
            vcd_file="simulation.vcd",
            time=1523
        )
    """
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def send(self, message: str, title: str = None, color: str = "#36a64f"):
        """Send a message to Slack."""
        payload = {
            "attachments": [{
                "color": color,
                "title": title or "WaveformGPT",
                "text": message,
                "footer": "WaveformGPT",
                "ts": int(datetime.now().timestamp())
            }]
        }
        
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            self.webhook_url,
            data=data,
            headers={'Content-Type': 'application/json'}
        )
        
        urllib.request.urlopen(req)
    
    def send_violation(self, title: str, details: str, 
                       vcd_file: str = None, time: int = None):
        """Send a violation alert."""
        message = details
        if vcd_file:
            message += f"\n📁 File: `{vcd_file}`"
        if time is not None:
            message += f"\n⏱️ Time: `{time}`"
        
        self.send(message, title=f"🚨 {title}", color="#ff0000")
    
    def send_ci_results(self, results: List[CICheckResult], vcd_file: str):
        """Send CI check results summary."""
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        
        color = "#36a64f" if passed == total else "#ff0000"
        
        message_lines = [f"*{passed}/{total} checks passed*\n"]
        
        for r in results:
            emoji = "✅" if r.passed else "❌"
            message_lines.append(f"{emoji} {r.check_name}")
        
        self.send(
            "\n".join(message_lines),
            title=f"WaveformGPT CI Results - {vcd_file}",
            color=color
        )


class GitHubIntegration:
    """
    Create GitHub issues and comments from waveform analysis.
    
    Usage:
        gh = GitHubIntegration(token="ghp_...", repo="user/repo")
        
        # Create issue for violation
        gh.create_issue(
            title="Protocol violation in FIFO module",
            body="Detected AXI protocol violation...",
            labels=["bug", "verification"]
        )
    """
    
    def __init__(self, token: str, repo: str):
        self.token = token
        self.repo = repo
        self.api_base = f"https://api.github.com/repos/{repo}"
    
    def _request(self, endpoint: str, method: str = "GET", data: Dict = None):
        """Make GitHub API request."""
        url = f"{self.api_base}/{endpoint}"
        
        headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
        }
        
        if data:
            body = json.dumps(data).encode('utf-8')
            headers["Content-Type"] = "application/json"
        else:
            body = None
        
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode('utf-8'))
    
    def create_issue(self, title: str, body: str, 
                     labels: List[str] = None,
                     assignees: List[str] = None) -> Dict:
        """Create a GitHub issue."""
        data = {
            "title": title,
            "body": body,
        }
        
        if labels:
            data["labels"] = labels
        if assignees:
            data["assignees"] = assignees
        
        return self._request("issues", method="POST", data=data)
    
    def comment_on_issue(self, issue_number: int, comment: str) -> Dict:
        """Add a comment to an issue."""
        return self._request(
            f"issues/{issue_number}/comments",
            method="POST",
            data={"body": comment}
        )
    
    def create_violation_issue(self, violation_type: str,
                                details: str,
                                vcd_file: str,
                                time: int = None,
                                signal: str = None):
        """Create an issue for a detected violation."""
        
        body = f"""## Waveform Violation Detected

**Type:** {violation_type}
**File:** `{vcd_file}`
"""
        if time is not None:
            body += f"**Time:** `{time}`\n"
        if signal:
            body += f"**Signal:** `{signal}`\n"
        
        body += f"""
### Details
{details}

---
*Generated by WaveformGPT*
"""
        
        return self.create_issue(
            title=f"[WaveformGPT] {violation_type}",
            body=body,
            labels=["bug", "verification", "waveformgpt"]
        )


# =============================================================================
# REPORT GENERATION
# =============================================================================

class ReportGenerator:
    """
    Generate HTML/PDF reports from waveform analysis.
    
    Usage:
        report = ReportGenerator("simulation.vcd")
        report.add_section("Overview", chat.ask("Summarize the simulation"))
        report.add_section("Protocol Check", chat.ask("Any protocol violations?"))
        report.add_waveform(["clk", "data", "valid"])
        report.generate("report.html")
    """
    
    def __init__(self, vcd_file: str, title: str = None):
        self.vcd_file = vcd_file
        self.title = title or f"Waveform Analysis Report - {Path(vcd_file).name}"
        self.sections: List[Dict] = []
        self.waveforms: List[Dict] = []
        
        from waveformgpt import VCDParser
        self.parser = VCDParser(vcd_file)
    
    def add_section(self, title: str, content: str):
        """Add a text section."""
        if hasattr(content, 'content'):
            content = content.content
        self.sections.append({"title": title, "content": str(content)})
    
    def add_waveform(self, signals: List[str], 
                     time_range: tuple = None,
                     title: str = None):
        """Add a waveform visualization."""
        from waveformgpt import ASCIIWaveform
        
        viz = ASCIIWaveform(self.parser)
        ascii_art = viz.render(signals, time_range=time_range)
        
        self.waveforms.append({
            "title": title or "Waveform",
            "signals": signals,
            "ascii": ascii_art,
            "time_range": time_range
        })
    
    def add_ci_results(self, results: List[CICheckResult]):
        """Add CI check results."""
        content = []
        for r in results:
            status = "✅ PASS" if r.passed else "❌ FAIL"
            content.append(f"### {status} {r.check_name}")
            content.append(f"{r.summary}")
            content.append(f"*Duration: {r.duration_seconds:.2f}s*")
            content.append("")
        
        self.sections.append({
            "title": "CI Check Results",
            "content": "\n".join(content)
        })
    
    def generate_html(self, output: str = None) -> str:
        """Generate HTML report."""
        output = output or self.vcd_file.replace('.vcd', '_report.html')
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{self.title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2563eb; border-bottom: 2px solid #2563eb; padding-bottom: 10px; }}
        h2 {{ color: #1e40af; margin-top: 30px; }}
        .section {{ background: #f8fafc; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .waveform {{ background: #1e293b; color: #e2e8f0; padding: 15px; 
                     font-family: 'Monaco', 'Consolas', monospace; border-radius: 8px;
                     overflow-x: auto; white-space: pre; }}
        .pass {{ color: #22c55e; }}
        .fail {{ color: #ef4444; }}
        .meta {{ color: #64748b; font-size: 0.9em; }}
        pre {{ background: #f1f5f9; padding: 15px; border-radius: 4px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>📊 {self.title}</h1>
    <p class="meta">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
       File: {self.vcd_file}</p>
"""
        
        for section in self.sections:
            # Convert markdown-ish to HTML
            content = section['content']
            content = content.replace('### ', '<h3>').replace('\n### ', '</h3>\n<h3>')
            content = content.replace('✅', '<span class="pass">✅</span>')
            content = content.replace('❌', '<span class="fail">❌</span>')
            content = content.replace('\n\n', '</p><p>')
            
            html += f"""
    <div class="section">
        <h2>{section['title']}</h2>
        <p>{content}</p>
    </div>
"""
        
        for wf in self.waveforms:
            html += f"""
    <div class="section">
        <h2>{wf['title']}</h2>
        <p>Signals: {', '.join(wf['signals'])}</p>
        <div class="waveform">{wf['ascii']}</div>
    </div>
"""
        
        html += """
    <footer class="meta" style="margin-top: 40px; text-align: center;">
        Generated by WaveformGPT 🌊🤖
    </footer>
</body>
</html>
"""
        
        with open(output, 'w') as f:
            f.write(html)
        
        return output
    
    def generate_markdown(self, output: str = None) -> str:
        """Generate Markdown report."""
        output = output or self.vcd_file.replace('.vcd', '_report.md')
        
        md = [f"# {self.title}"]
        md.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        md.append(f"\n**File:** `{self.vcd_file}`\n")
        
        for section in self.sections:
            md.append(f"\n## {section['title']}\n")
            md.append(section['content'])
        
        for wf in self.waveforms:
            md.append(f"\n## {wf['title']}\n")
            md.append(f"Signals: {', '.join(wf['signals'])}\n")
            md.append(f"```\n{wf['ascii']}\n```")
        
        md.append("\n---\n*Generated by WaveformGPT 🌊🤖*")
        
        with open(output, 'w') as f:
            f.write('\n'.join(md))
        
        return output


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_simulation_and_analyze(
    sources: List[str],
    questions: List[str],
    simulator: str = "icarus",
    use_llm: bool = True
) -> Dict[str, Any]:
    """
    One-shot: Run simulation and analyze results.
    
    Args:
        sources: Verilog source files
        questions: Questions to ask about the waveform
        simulator: "icarus" or "verilator"
        use_llm: Use LLM for analysis
    
    Returns:
        Dict with simulation result and answers
    """
    # Run simulation
    if simulator == "icarus":
        runner = IcarusRunner()
        result = runner.run(sources)
    else:
        raise ValueError(f"Unknown simulator: {simulator}")
    
    if not result.success or not result.vcd_file:
        return {
            "success": False,
            "simulation_errors": result.errors,
            "answers": []
        }
    
    # Analyze
    from waveformgpt import WaveformChat
    chat = WaveformChat(result.vcd_file, use_llm=use_llm)
    
    answers = []
    for q in questions:
        response = chat.ask(q)
        answers.append({
            "question": q,
            "answer": response.content
        })
    
    return {
        "success": True,
        "vcd_file": result.vcd_file,
        "duration": result.duration_seconds,
        "answers": answers
    }


def ci_check_waveform(
    vcd_file: str,
    checks: List[Dict[str, str]],
    output_format: str = "junit"
) -> int:
    """
    Run CI checks on a waveform file.
    
    Args:
        vcd_file: Path to VCD file
        checks: List of {"name": "...", "question": "..."} dicts
        output_format: "junit", "json", or "text"
    
    Returns:
        Exit code (0 = all passed, 1 = failures)
    """
    ci = WaveformCI(vcd_file)
    
    for check in checks:
        ci.add_check(
            check["name"],
            lambda chat, q=check["question"]: chat.ask(q)
        )
    
    results = ci.run_all()
    
    if output_format == "junit":
        ci.generate_junit_xml(results)
    elif output_format == "json":
        print(json.dumps([{
            "name": r.check_name,
            "passed": r.passed,
            "summary": r.summary
        } for r in results], indent=2))
    else:
        return ci.print_report(results)
    
    return 0 if all(r.passed for r in results) else 1
