sim:
	iverilog -o axi_read_reorder_tb.vvp axi_read_reorder_tb.sv axi_read_master.sv axi_read_reorder_slave.sv -g2005-sv
	vvp axi_read_reorder_tb.vvp

analyze:
	python3 match_by_id.py

all: sim analyze
.PHONY: sim analyze all
