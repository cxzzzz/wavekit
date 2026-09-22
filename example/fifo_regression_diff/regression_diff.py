import numpy as np

from wavekit import VcdReader
from wavekit.pattern import Pattern, match


def read_out_data(path):
    with VcdReader(path) as f:
        tb = f['fifo_tb']
        with f.clock_domain(tb['clk']):
            result = match(
                Pattern().consume(tb['r_en'].w & ~tb['empty'].w).capture('data', tb['data_out'].w)
            )
        return result.filter_ok().captures['data']


data_v1 = read_out_data('fifo_tb_v1.vcd')
data_v2 = read_out_data('fifo_tb_v2.vcd')

print(f'v1: {len(data_v1.value)} reads, v2: {len(data_v2.value)} reads')

n = min(len(data_v1.value), len(data_v2.value))
mismatches = np.where(data_v1.value[:n] != data_v2.value[:n])[0]

if len(mismatches) == 0:
    print('No differences found.')
else:
    print(f'{len(mismatches)} mismatches:')
    for i in mismatches:
        print(
            f'  read #{i} (cycle {data_v1.cycle[i]}): '
            f'v1={data_v1.value[i]}, v2={data_v2.value[i]}'
        )
