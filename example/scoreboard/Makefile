sim:
	iverilog -o fifo_tb.vvp fifo_tb.sv fifo.sv -g2005-sv
	vvp fifo_tb.vvp

verify:
	python3 verify.py

all: sim verify
.PHONY: sim verify all
