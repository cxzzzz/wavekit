sim:
	iverilog -o fifo_tb.vvp fifo_tb.sv fifo.sv -g2005-sv
	vvp fifo_tb.vvp

analyze:
	python3 occupancy.py

all: sim analyze
.PHONY: sim analyze all
