from wavekit import VcdReader
from wavekit.pattern import collect

with VcdReader('dma_tb.vcd') as reader:
    tb = reader['dma_tb']

    with reader.clock_domain(tb['clk']):
        cmd_fire = tb['cmd_valid'].w & tb['cmd_ready'].w
        cmd_op = tb['cmd_op'].w
        cmd_len = tb['cmd_len'].w
        w_fire = tb['wvalid'].w & tb['wready'].w
        w_data = tb['wdata'].w
        r_fire = tb['rvalid'].w & tb['rready'].w
        r_data = tb['rdata'].w
        rsp_fire = tb['rsp_valid'].w & tb['rsp_ready'].w
        rsp_status = tb['rsp_status'].w

    def read_dma_cmd(ctx):
        if not ctx.value(cmd_fire):
            return None

        op = int(ctx.value(cmd_op))
        length = int(ctx.value(cmd_len))

        if op == 1:
            data = []
            for _ in range(length):
                ctx.consume(w_fire, channel='wdata')
                data.append(int(ctx.value(w_data)))

            ctx.consume(rsp_fire, channel='rsp')
            return {
                'op': 'write',
                'data': data,
                'status': int(ctx.value(rsp_status)),
            }

        if op == 0:
            ctx.consume(rsp_fire, channel='rsp')
            data = []
            for _ in range(length):
                ctx.consume(r_fire, channel='rdata')
                data.append(int(ctx.value(r_data)))

            return {'op': 'read', 'data': data, 'status': int(ctx.value(rsp_status))}

        ctx.require(False, message=f'unknown DMA op {op}')
        return None

    commands = collect(read_dma_cmd)
    print('DMA commands:', commands)
