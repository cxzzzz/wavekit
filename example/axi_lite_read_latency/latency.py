from wavekit import VcdReader
from wavekit.pattern import Pattern, match

with VcdReader('axi_lite_tb.vcd') as reader:
    ar_sigs = {
        key[0].groups[0]: wave
        for key, wave in reader.load_matched_waveforms(
            r'axi_lite_tb./ar([a-z]+)/', 'axi_lite_tb.clk'
        ).items()
    }
    r_sigs = {
        key[0].groups[0]: wave
        for key, wave in reader.load_matched_waveforms(
            r'axi_lite_tb./r([a-z]+)/', 'axi_lite_tb.clk'
        ).items()
    }

    result = match(
        Pattern()
        .wait(ar_sigs['valid'] & ar_sigs['ready'])
        .consume(r_sigs['valid'] & r_sigs['ready'], channel='read-response')
        .capture('rdata', r_sigs['data'])
    )

    ok = result.filter_ok()
    print('AXI-Lite read latency (cycles):', list(ok.end.clock - ok.start.clock))
    print('AXI-Lite read data:', [int(value) for value in ok.captures['rdata'].value])
