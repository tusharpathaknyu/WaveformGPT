"""
Example: Timing Analysis with WaveformGPT

This example shows how to analyze timing relationships
between signals in your design.
"""

from waveformgpt import VCDParser, QueryEngine
from waveformgpt.query_engine import Query, QueryType, EventCondition, TemporalConstraint

# Load VCD
parser = VCDParser("sample.vcd")
engine = QueryEngine(parser)

# Query 1: Find all request-to-acknowledge latencies
query = Query(
    type=QueryType.MEASURE,
    events=[
        EventCondition(signal="req", condition="rise"),
        EventCondition(signal="ack", condition="rise")
    ]
)

result = engine.execute(query)
print("Request-to-Acknowledge Timing Analysis")
print("=" * 40)
print(result.answer)
print()

if result.statistics:
    print(f"  Min latency: {result.statistics['min']} ns")
    print(f"  Max latency: {result.statistics['max']} ns")
    print(f"  Avg latency: {result.statistics['avg']:.2f} ns")
print()

# Query 2: Find handshakes completing within spec (30ns)
query = Query(
    type=QueryType.FIND_PATTERN,
    events=[
        EventCondition(signal="req", condition="rise"),
        EventCondition(signal="ack", condition="rise")
    ],
    temporal=TemporalConstraint(max_cycles=30)
)

result = engine.execute(query)
print("Handshakes within 30ns spec")
print("=" * 40)
print(f"Found {result.count} compliant handshakes")
for match in result.matches:
    delay = match.metadata.get('delay', 0)
    print(f"  req@{match.time} -> ack@{match.end_time} (delay: {delay}ns)")
print()

# Query 3: Count clock cycles
query = Query(
    type=QueryType.COUNT,
    events=[EventCondition(signal="clk", condition="rise")]
)

result = engine.execute(query)
print(f"Total clock cycles: {result.count}")
