sim:
	iverilog -g2005-sv -Pfifo_tb.BUGGY_DATA=0 -o fifo_tb.vvp fifo_tb.sv fifo.sv
	vvp fifo_tb.vvp
	mv fifo_tb.vcd fifo_tb_v1.vcd
	iverilog -g2005-sv -Pfifo_tb.BUGGY_DATA=1 -o fifo_tb.vvp fifo_tb.sv fifo.sv
	vvp fifo_tb.vvp
	mv fifo_tb.vcd fifo_tb_v2.vcd

analyze:
	python3 regression_diff.py

all: sim analyze
.PHONY: sim analyze all
