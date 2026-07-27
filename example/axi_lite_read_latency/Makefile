sim:
	iverilog -o axi_lite_tb.vvp axi_lite_tb.sv axi_lite_master.sv axi_lite_reg_slave.sv -g2005-sv
	vvp axi_lite_tb.vvp

analyze:
	python3 latency.py

all: sim analyze
.PHONY: sim analyze all
