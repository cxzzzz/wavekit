sim:
	iverilog -o dma_tb.vvp dma_tb.sv dma_engine.sv -g2005-sv
	vvp dma_tb.vvp

analyze:
	python3 extract_commands.py

all: sim analyze
.PHONY: sim analyze all
