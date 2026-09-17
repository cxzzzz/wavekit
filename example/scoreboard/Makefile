sim:
	iverilog -g2005-sv -o fifo_tb.vvp fifo_tb.sv fifo.sv
	vvp fifo_tb.vvp

verify:
	python3 verify.py

all: sim verify
.PHONY: sim verify all
