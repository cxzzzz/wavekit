from wavekit import VcdReader
from wavekit.pattern import Pattern, match

with VcdReader('axi_lite_tb.vcd') as reader:
    tb = reader['axi_lite_tb']

    with reader.clock_domain(tb['clk']):
        result = match(
            Pattern()
            .wait(tb['arvalid'].w & tb['arready'].w)
            .consume(tb['rvalid'].w & tb['rready'].w)
            .capture('rdata', tb['rdata'].w)
        )

    ok = result.filter_ok()
    print('AXI-Lite read latency (cycles):', list(ok.end.cycle - ok.start.cycle))
    print('AXI-Lite read data:', [int(value) for value in ok.captures['rdata'].value])
