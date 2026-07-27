from wavekit import VcdReader
from wavekit.pattern import collect

with VcdReader('dma_tb.vcd') as reader:
    cmd_sigs = {
        key[0].groups[0]: wave
        for key, wave in reader.load_matched_waveforms(
            r'dma_tb./cmd_([a-z]+)/', 'dma_tb.clk'
        ).items()
    }
    w_sigs = {
        key[0].groups[0]: wave
        for key, wave in reader.load_matched_waveforms(r'dma_tb./w([a-z]+)/', 'dma_tb.clk').items()
    }
    r_sigs = {
        key[0].groups[0]: wave
        for key, wave in reader.load_matched_waveforms(r'dma_tb./r([a-z]+)/', 'dma_tb.clk').items()
    }
    rsp_sigs = {
        key[0].groups[0]: wave
        for key, wave in reader.load_matched_waveforms(
            r'dma_tb./rsp_([a-z]+)/', 'dma_tb.clk'
        ).items()
    }

    cmd_fire = cmd_sigs['valid'] & cmd_sigs['ready']
    w_fire = w_sigs['valid'] & w_sigs['ready']
    r_fire = r_sigs['valid'] & r_sigs['ready']
    rsp_fire = rsp_sigs['valid'] & rsp_sigs['ready']

    def read_dma_cmd(ctx):
        if not ctx.value(cmd_fire):
            return None

        op = int(ctx.value(cmd_sigs['op']))
        length = int(ctx.value(cmd_sigs['len']))

        if op == 1:
            data = []
            for _ in range(length):
                ctx.consume(w_fire, channel='wdata')
                data.append(int(ctx.value(w_sigs['data'])))

            ctx.consume(rsp_fire, channel='rsp')
            return {
                'op': 'write',
                'data': data,
                'status': int(ctx.value(rsp_sigs['status'])),
            }

        if op == 0:
            ctx.consume(rsp_fire, channel='rsp')
            data = []
            for _ in range(length):
                ctx.consume(r_fire, channel='rdata')
                data.append(int(ctx.value(r_sigs['data'])))

            return {'op': 'read', 'data': data, 'status': int(ctx.value(rsp_sigs['status']))}

        ctx.require(False, message=f'unknown DMA op {op}')
        return None

    commands = collect(read_dma_cmd)
    print('DMA commands:', commands)
