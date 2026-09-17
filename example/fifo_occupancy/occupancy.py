import numpy as np

from wavekit import VcdReader

with VcdReader('fifo_tb.vcd') as f:
    fifo = f['fifo_tb.s_fifo']
    depth = 8

    with f.clock_domain(fifo['clk']):
        w_ptr = fifo['w_ptr'].w
        r_ptr = fifo['r_ptr'].w

    fifo_water_level = (w_ptr + depth - r_ptr) % depth
    average_fifo_water_level = np.mean(fifo_water_level.value)
    print('FIFO Occupancy Analysis:')
    print(f'  Average occupancy level: {average_fifo_water_level:.2f}')
    print(f'  Max occupancy level: {np.max(fifo_water_level.value)}')
    print(f'  Min occupancy level: {np.min(fifo_water_level.value)}')
