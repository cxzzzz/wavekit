sim:
	iverilog -g2005-sv -o dma_tb.vvp dma_tb.sv dma_engine.sv
	vvp dma_tb.vvp

analyze:
	python3 extract_commands.py

all: sim analyze
.PHONY: sim analyze all
