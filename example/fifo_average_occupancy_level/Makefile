sim:
	iverilog -o fifo_tb.vvp fifo_tb.sv fifo.sv -g2005-sv
	vvp fifo_tb.vvp

py:
	python3 fifo_average_occupancy_level.py

all: sim py