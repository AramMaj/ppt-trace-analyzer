"""

"""
import sys
from trace_parser import TraceParser
from fsdp_detector import StandardFSDPDetector

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Analyze a single trace:    python main.py <trace.json> [--output report.txt]")
        print("  Compare multiple traces:   python main.py --compare trace1.json trace2.json --output comparison.csv [--op OP]")
        sys.exit(1)

    if sys.argv[1] == "--compare":
        """ Vergrößern zu Rank Compare  """
        pass

    trace_file = sys.argv[1]
    output_file = None
    if len(sys.argv) >= 3 and sys.argv[2] == "--output":
        output_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    print(f"Loading trace from {trace_file}...")
    parser = TraceParser(trace_file)
    if not parser.load():
        sys.exit(1)
    print(f"Loaded {len(parser.cpu_events)} CPU, {len(parser.gpu_events)} GPU, {len(parser.memory_events)} memory events.")
    roots = parser.build_tree()
    parser.attribute_gpu_kernel_with_logical_operation(roots)
    parser.attribute_memory(roots)

    detector = StandardFSDPDetector()
    fsdp = detector.extract_fsdp_phases(roots)

if __name__ == "__main__":
    main()