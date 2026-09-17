sim:
	iverilog -g2005-sv -o axi_lite_tb.vvp axi_lite_tb.sv axi_lite_master.sv axi_lite_reg_slave.sv
	vvp axi_lite_tb.vvp

analyze:
	python3 latency.py

all: sim analyze
.PHONY: sim analyze all
