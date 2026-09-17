sim:
	iverilog -g2005-sv -o fifo_tb.vvp fifo_tb.sv fifo.sv
	vvp fifo_tb.vvp

analyze:
	python3 latency.py

all: sim analyze
.PHONY: sim analyze all
