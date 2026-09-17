sim:
	iverilog -o system_tb.vvp system_tb.sv router.sv fifo_bank.sv fifo.sv -g2012
	vvp system_tb.vvp

analyze:
	python3 occupancy_report.py

all: sim analyze
.PHONY: sim analyze all
