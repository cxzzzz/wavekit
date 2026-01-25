import numpy as np

from wavekit import VcdReader

with VcdReader('fifo_tb.vcd') as f:
    clock = 'fifo_tb.s_fifo.clk'
    depth = 8

    w_ptr = f.load_waveform('fifo_tb.s_fifo.w_ptr[2:0]', clock=clock)
    r_ptr = f.load_waveform('fifo_tb.s_fifo.r_ptr[2:0]', clock=clock)
    fifo_water_level = (w_ptr + depth - r_ptr) % depth
    average_fifo_water_level = np.mean(fifo_water_level.value)
    print('FIFO Occupancy Analysis:')
    print(f'  Average occupancy level: {average_fifo_water_level:.2f}')
    print(f'  Max occupancy level: {np.max(fifo_water_level.value)}')
    print(f'  Min occupancy level: {np.min(fifo_water_level.value)}')
