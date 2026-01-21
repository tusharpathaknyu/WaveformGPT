"""
WaveformGPT CLI - Command Line Interface.

Usage:
    waveformgpt <vcd_file>              # Interactive mode
    waveformgpt <vcd_file> -q "query"   # Single query mode
    waveformgpt <vcd_file> --signals    # List signals
"""

import click
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from pathlib import Path

from waveformgpt.chat import WaveformChat


console = Console()


@click.command()
@click.argument("vcd_file", required=False, type=click.Path(exists=True))
@click.option("-q", "--query", help="Single query to execute")
@click.option("--signals", is_flag=True, help="List available signals")
@click.option("--info", is_flag=True, help="Show VCD file information")
@click.option("--llm/--no-llm", default=False, help="Enable LLM-powered parsing")
@click.option("-o", "--output", type=click.Path(), help="Export GTKWave save file")
@click.option("--export-gtkwave", type=click.Path(), help="Export GTKWave save file and exit")
def main(vcd_file, query, signals, info, llm, output, export_gtkwave):
    """
    WaveformGPT - Query simulation waveforms in natural language.
    
    Examples:
    
    \b
        waveformgpt simulation.vcd
        waveformgpt simulation.vcd -q "when does clk rise?"
        waveformgpt simulation.vcd --signals
        waveformgpt simulation.vcd --info
    """
    
    if not vcd_file:
        # No file provided, show help
        console.print(Panel(
            "[bold cyan]WaveformGPT[/] - Query waveforms in natural language\n\n"
            "Usage: waveformgpt <vcd_file> [options]\n\n"
            "Examples:\n"
            "  waveformgpt simulation.vcd\n"
            "  waveformgpt sim.vcd -q 'when does clk rise?'\n"
            "  waveformgpt sim.vcd --signals\n"
            "  waveformgpt sim.vcd --info",
            title="Welcome"
        ))
        return
    
    # Show info mode
    if info:
        from waveformgpt.vcd_parser import VCDParser
        parser = VCDParser(vcd_file)
        header = parser.header
        time_range = parser.get_time_range()
        
        console.print(Panel(
            f"[bold]File:[/] {vcd_file}\n"
            f"[bold]Date:[/] {header.date or 'N/A'}\n"
            f"[bold]Version:[/] {header.version or 'N/A'}\n"
            f"[bold]Timescale:[/] {header.timescale}\n"
            f"[bold]Signals:[/] {len(header.signals)}\n"
            f"[bold]Time Range:[/] {time_range[0]} to {time_range[1]} {header.timescale_unit}",
            title="VCD Info"
        ))
        return
    
    # Export GTKWave mode
    if export_gtkwave:
        from waveformgpt.gtkwave import generate_savefile
        from waveformgpt.vcd_parser import VCDParser
        parser = VCDParser(vcd_file)
        sigs = [s.full_name for s in list(parser.header.signals.values())[:20]]
        result = generate_savefile(vcd_file, export_gtkwave, sigs)
        console.print(f"[green]{result}[/]")
        return
    
    # Initialize chat
    chat = WaveformChat(vcd_file, use_llm=llm)
    
    # List signals mode
    if signals:
        result = chat.ask("signals")
        console.print(result.content)
        return
    
    # Single query mode
    if query:
        result = chat.ask(query)
        console.print(Panel(result.content, title="Result"))
        
        # Export if requested
        if output and result.result:
            chat.export_gtkwave_savefile(output)
            console.print(f"[green]Saved GTKWave file:[/] {output}")
        return
    
    # Interactive mode
    interactive_mode(chat, output)


def interactive_mode(chat: WaveformChat, output_path: str = None):
    """Run interactive chat mode."""
    console.print(Panel(
        "[bold cyan]WaveformGPT Interactive Mode[/]\n\n"
        f"Loaded: [green]{chat.vcd_file}[/]\n"
        f"Signals: {len(chat.parser.header.signals)}\n\n"
        "Commands:\n"
        "  [yellow]signals[/]  - List available signals\n"
        "  [yellow]help[/]     - Show query examples\n"
        "  [yellow]export[/]   - Export GTKWave save file\n"
        "  [yellow]quit[/]     - Exit\n",
        title="Welcome"
    ))
    
    while True:
        try:
            query = Prompt.ask("\n[bold cyan]>[/]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Goodbye![/]")
            break
        
        query = query.strip()
        if not query:
            continue
        
        if query.lower() in ("quit", "exit", "q"):
            console.print("[yellow]Goodbye![/]")
            break
        
        if query.lower() == "export":
            # Export GTKWave file
            if output_path:
                out = output_path
            else:
                vcd_path = Path(chat.vcd_file)
                out = str(vcd_path.with_suffix('.gtkw'))
            
            result = chat.export_gtkwave_savefile(out)
            console.print(f"[green]{result}[/]")
            continue
        
        if query.lower() == "open" or query.lower() == "gtkwave":
            # Open in GTKWave
            from waveformgpt.gtkwave import launch_gtkwave
            vcd_path = Path(chat.vcd_file)
            gtkw_path = vcd_path.with_suffix('.gtkw')
            
            if gtkw_path.exists():
                launch_gtkwave(str(chat.vcd_file), str(gtkw_path))
            else:
                launch_gtkwave(str(chat.vcd_file))
            console.print("[green]Launched GTKWave[/]")
            continue
        
        # Execute query
        result = chat.ask(query)
        
        # Format output
        if result.result and result.result.matches:
            console.print(Panel(
                result.content,
                title=f"[green]Found {len(result.result.matches)} matches[/]"
            ))
        else:
            console.print(Panel(result.content))



if __name__ == "__main__":
    main()
