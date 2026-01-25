import numpy as np
from wavekit import VcdReader, Waveform

def verify_fifo_data_integrity():
    """
    Verifies that data written to the FIFO matches data read from the FIFO.
    This demonstrates:
    1. Loading multi-bit bus signals.
    2. Using boolean masking to extract valid transactions.
    3. Comparing expected vs actual data streams (Scoreboarding).
    """
    with VcdReader('fifo_tb.vcd') as f:
        clock = 'fifo_tb.clk'

        # Load signals sampled on clock edges
        w_en = f.load_waveform('fifo_tb.w_en', clock=clock)
        full = f.load_waveform('fifo_tb.full', clock=clock)
        data_in = f.load_waveform('fifo_tb.data_in', clock=clock)

        r_en = f.load_waveform('fifo_tb.r_en', clock=clock)
        empty = f.load_waveform('fifo_tb.empty', clock=clock)
        data_out = f.load_waveform('fifo_tb.data_out', clock=clock)

        valid_w_idx = (w_en & (~ full)).filter(lambda x:x != 0).clock
        valid_w_data = data_in.take(valid_w_idx)

        valid_r_idx = (r_en & (~ empty)).filter(lambda x:x != 0).clock
        valid_r_data = data_out.take(valid_r_idx + 1)

        print(f"Total Valid Writes: {len(valid_w_data.value)}")
        print(f"Total Valid Reads:  {len(valid_r_data.value)}")

        num_checked = min(len(valid_w_data.value), len(valid_r_data.value))

        expected = valid_w_data.value[:num_checked]
        actual = valid_r_data.value[:num_checked]

        if np.array_equal(expected, actual):
            print("\n[PASS] Data Integrity Check Passed!")
            print(f"Verified {num_checked} transactions.")
        else:
            print("\n[FAIL] Data Integrity Check Failed!")
            mismatch_indices = np.where(expected != actual)[0]
            print(f"First mismatch at index {mismatch_indices[0]}:")
            print(f"  Expected: {expected[mismatch_indices[0]]}")
            print(f"  Actual:   {actual[mismatch_indices[0]]}")

if __name__ == "__main__":
    verify_fifo_data_integrity()
