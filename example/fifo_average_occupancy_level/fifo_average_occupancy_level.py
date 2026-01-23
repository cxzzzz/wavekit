import numpy as np

from wavekit import VcdReader

with VcdReader('fifo_tb.vcd') as f:
    clock = 'fifo_tb.s_fifo.clk'
    depth = 8

    w_ptr = f.load_waveform('fifo_tb.s_fifo.w_ptr[2:0]', clock=clock)
    r_ptr = f.load_waveform('fifo_tb.s_fifo.r_ptr[2:0]', clock=clock)
    fifo_water_level = (w_ptr + depth - r_ptr) % depth
    average_fifo_water_level = np.mean(fifo_water_level.value)
    print(f'average fifo occupancy level: {average_fifo_water_level}')

    w_en = f.load_waveform('fifo_tb.s_fifo.w_en', clock=clock)
    full = f.load_waveform('fifo_tb.s_fifo.full', clock=clock)
    wr_backpressure = w_en & full
    wr_backpressure_cnt = np.sum(wr_backpressure.value)
    print(f'wr backpressure count: {wr_backpressure_cnt}')
