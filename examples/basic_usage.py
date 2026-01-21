"""
Example: Basic WaveformGPT Usage

This example demonstrates how to use WaveformGPT to query
a VCD waveform file using natural language.
"""

from waveformgpt import WaveformChat

# Load a VCD file
chat = WaveformChat("sample.vcd")

# Ask natural language questions
response = chat.ask("When does clk rise?")
print(response.content)
print()

response = chat.ask("How many times does req go high?")
print(response.content)
print()

response = chat.ask("Find req followed by ack")
print(response.content)
print()

# Access the raw query results
if response.result:
    print(f"Found {len(response.result.matches)} matches")
    for match in response.result.matches[:5]:
        print(f"  Time: {match.time}, Signals: {match.signals}")

# Export results to GTKWave
chat.export_gtkwave_savefile("results.gtkw")
print("\nExported GTKWave save file: results.gtkw")
