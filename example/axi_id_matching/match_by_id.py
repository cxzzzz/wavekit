from wavekit import VcdReader
from wavekit.pattern import Pattern, match

with VcdReader('axi_read_reorder_tb.vcd') as reader:
    tb = reader['axi_read_reorder_tb']

    with reader.clock_domain(tb['clk']):

        def resp_fire(index, captures):
            return bool(tb['rvalid'].w.value[index] & tb['rready'].w.value[index]) and int(
                tb['rid'].w.value[index]
            ) == int(captures['arid'])

        result = match(
            Pattern()
            .wait(tb['arvalid'].w & tb['arready'].w)
            .capture('arid', tb['arid'].w)
            .consume(resp_fire)
            .capture('rdata', tb['rdata'].w)
        )

    ok = result.filter_ok()
    pairs = sorted(zip(ok.captures['arid'].value, ok.captures['rdata'].value))
    print('AXI read responses by ID:', [(int(arid_), int(data_)) for arid_, data_ in pairs])
