sim:
	iverilog -g2012 -o system_tb.vvp system_tb.sv router.sv fifo_bank.sv fifo.sv
	vvp system_tb.vvp

analyze:
	python3 occupancy_report.py

all: sim analyze
.PHONY: sim analyze all
