from wavekit import VcdReader
from wavekit.pattern import Pattern, match

with VcdReader('axi_read_reorder_tb.vcd') as reader:
    ar_sigs = {
        key[0].groups[0]: wave
        for key, wave in reader.load_matched_waveforms(
            r'axi_read_reorder_tb./ar([a-z]+)/', 'axi_read_reorder_tb.clk'
        ).items()
    }
    r_sigs = {
        key[0].groups[0]: wave
        for key, wave in reader.load_matched_waveforms(
            r'axi_read_reorder_tb./r([a-z]+)/', 'axi_read_reorder_tb.clk'
        ).items()
    }

    def resp_fire(index, captures):
        return bool(r_sigs['valid'].value[index] & r_sigs['ready'].value[index]) and int(
            r_sigs['id'].value[index]
        ) == int(captures['arid'])

    result = match(
        Pattern()
        .wait(ar_sigs['valid'] & ar_sigs['ready'])
        .capture('arid', ar_sigs['id'])
        .consume(resp_fire, channel=lambda index, captures: captures['arid'])
        .capture('rdata', r_sigs['data'])
    )

    ok = result.filter_ok()
    pairs = sorted(zip(ok.captures['arid'].value, ok.captures['rdata'].value))
    print('AXI read responses by ID:', [(int(arid_), int(data_)) for arid_, data_ in pairs])
