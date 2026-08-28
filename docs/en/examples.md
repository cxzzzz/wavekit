# Examples

The complete examples are hand-maintained projects under [`example/`](../example/).
Each directory contains its HDL fixture, analysis script, and Makefile. Run the
commands from the repository root.

| Example | Purpose | WaveKit focus | Run command |
| --- | --- | --- | --- |
| [FIFO occupancy](../example/fifo_occupancy/) | Compute FIFO occupancy from sampled pointers | Reader loading; waveform arithmetic; NumPy interoperability | `make -C example/fifo_occupancy all` |
| [FIFO backpressure](../example/fifo_latency/) | Measure how long write requests are blocked by a full FIFO | Waveform operations; edge detection | `make -C example/fifo_latency all` |
| [AXI-Lite read latency](../example/axi_lite_read_latency/) | Measure AXI-Lite read response latency | Declarative pattern matching; event consumption | `make -C example/axi_lite_read_latency all` |
| [AXI ID matching](../example/axi_id_matching/) | Match read responses with requests by transaction ID | Declarative pattern matching; capture-dependent conditions | `make -C example/axi_id_matching all` |
| [DMA command stream](../example/dma_command_stream/) | Extract variable-length read and write commands | Programmable patterns; Python control flow | `make -C example/dma_command_stream all` |
| [Scoreboard](../example/scoreboard/) | Verify FIFO data integrity and ordering between writes and reads | Waveform filtering; indexed extraction; NumPy interoperability | `make -C example/scoreboard all` |
