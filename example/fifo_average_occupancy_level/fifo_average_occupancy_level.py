import numpy as np
from wavekit import VcdReader


with VcdReader("fifo_tb.vcd") as f: # open the VCD file
    clock = "fifo_tb.s_fifo.clk"
    depth = 8

    #calculate the average FIFO occupancy level.
    w_ptr = f.load_wave("fifo_tb.s_fifo.w_ptr[2:0]", clock=clock)   # load fifo write pointer signal
    r_ptr = f.load_wave("fifo_tb.s_fifo.r_ptr[2:0]", clock=clock)   # load fifo read pointer signal
    fifo_water_level = (w_ptr + depth - r_ptr) % depth              # calculate the occupancy level
    average_fifo_water_level = np.mean(fifo_water_level.value)      # calculate the average occupancy level using numpy
    print( f"average fifo occupancy level: {average_fifo_water_level}" )

    #count the number of times FIFO write backpressure
    w_en = f.load_wave("fifo_tb.s_fifo.w_en", clock=clock)      # load fifo write enable signal
    full = f.load_wave("fifo_tb.s_fifo.full", clock=clock)      # load fifo full signal
    wr_backpressure = w_en & full                               # calculate the fifo write backpressure
    wr_backpressure_cnt = np.sum(wr_backpressure.value)         # count the number of times FIFO write backpressure
    print( f"wr backpressure count: {wr_backpressure_cnt}" )