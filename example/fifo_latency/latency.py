import numpy as np
from wavekit import VcdReader

def analyze_write_latency():
    # Analyze how long write requests are blocked (Backpressure Latency)
    with VcdReader('fifo_tb.vcd') as f:
        clock = 'fifo_tb.s_fifo.clk'

        # Load signals sampled on clock edges
        w_en = f.load_waveform('fifo_tb.s_fifo.w_en', clock=clock)
        full = f.load_waveform('fifo_tb.s_fifo.full', clock=clock)

        # Logic: Request (w_en) is blocked if FIFO is full
        # blocked mask: when both w_en and full are high
        blocked = w_en & full

        if not np.any(blocked.value):
            print("No write backpressure detected.")
            return

        blocked_start = blocked.rising_edge().filter(lambda v: v!=0)
        blocked_end = blocked.falling_edge().filter(lambda v: v!=0)
        blocked_num = len(blocked_end.value)
        duration = (blocked_end.clock[:blocked_num] - blocked_start.clock[:blocked_num]) + 1

        print(f"Write Backpressure Analysis:")
        print(f"  Total blocked cycles: {np.sum(blocked.value)}")
        print(f"  Max consecutive blocked cycles: {np.max(duration)}")
        print(f"  Average blocking duration: {np.mean(duration):.2f} cycles")
        print(f"  Number of blocking events: {blocked_num}")

if __name__ == "__main__":
    analyze_write_latency()
