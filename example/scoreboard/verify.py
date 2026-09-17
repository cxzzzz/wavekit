import numpy as np

from wavekit import VcdReader


def verify_fifo_data_integrity():
    """
    Verifies that data written to the FIFO matches data read from the FIFO.
    This demonstrates:
    1. Dict-style signal access from a scope.
    2. Using boolean masking to extract valid transactions.
    3. Comparing expected vs actual data streams (Scoreboarding).
    """
    with VcdReader('fifo_tb.vcd') as f:
        tb = f['fifo_tb']

        with f.clock_domain(tb['clk']):
            valid_w_idx = (tb['w_en'].w & (~tb['full'].w)).filter(lambda x: x != 0).cycle
            valid_w_data = tb['data_in'].w.take(valid_w_idx)

            valid_r_idx = (tb['r_en'].w & (~tb['empty'].w)).filter(lambda x: x != 0).cycle
            valid_r_data = tb['data_out'].w.take(valid_r_idx + 1)

        print(f'Total Valid Writes: {len(valid_w_data.value)}')
        print(f'Total Valid Reads:  {len(valid_r_data.value)}')

        num_checked = min(len(valid_w_data.value), len(valid_r_data.value))

        expected = valid_w_data.value[:num_checked]
        actual = valid_r_data.value[:num_checked]

        if np.array_equal(expected, actual):
            print('\n[PASS] Data Integrity Check Passed!')
            print(f'Verified {num_checked} transactions.')
        else:
            print('\n[FAIL] Data Integrity Check Failed!')
            mismatch_indices = np.where(expected != actual)[0]
            print(f'First mismatch at index {mismatch_indices[0]}:')
            print(f'  Expected: {expected[mismatch_indices[0]]}')
            print(f'  Actual:   {actual[mismatch_indices[0]]}')


if __name__ == '__main__':
    verify_fifo_data_integrity()
