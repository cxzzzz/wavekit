import numpy as np

from wavekit import VcdReader

with VcdReader('system_tb.vcd') as reader:
    tb = reader['system_tb']
    depth = 8

    fifo_scopes = reader.get_matched_scopes(
        r'system_tb.cluster_gen[{0..3}].cluster_inst.**./fifo_(\d+)/'
    )

    with reader.clock_domain(tb['clk']):
        occupancies = {}
        for key, scope in fifo_scopes.items():
            cluster_idx = key[0].groups[0]
            fifo_idx = key[2].groups[0]
            w_ptr = scope['w_ptr'].w
            r_ptr = scope['r_ptr'].w
            occupancies[(cluster_idx, fifo_idx)] = (w_ptr + depth - r_ptr) % depth

    print('Per-FIFO average occupancy:')
    for (cluster_idx, fifo_idx), occ in sorted(occupancies.items()):
        print(f'  cluster_{cluster_idx} / fifo_{fifo_idx}: {np.mean(occ.value):.2f}')

    cluster_averages: dict[str, list[float]] = {}
    for (cluster_idx, _fifo_idx), occ in occupancies.items():
        cluster_averages.setdefault(cluster_idx, []).append(np.mean(occ.value))

    print('\nPer-cluster average occupancy:')
    busiest_cluster, busiest_avg = None, -1.0
    for cluster_idx, avgs in sorted(cluster_averages.items()):
        avg = np.mean(avgs)
        print(f'  cluster_{cluster_idx}: {avg:.2f}')
        if avg > busiest_avg:
            busiest_cluster, busiest_avg = cluster_idx, avg

    print(f'\nBusiest cluster: cluster_{busiest_cluster} (avg occupancy {busiest_avg:.2f})')
