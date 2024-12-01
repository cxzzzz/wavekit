import pytest
from wavekit import VcdReader


def test_vcdreader():

    vcd_reader = VcdReader("tests/testdata/jtag.vcd")

    j_state = vcd_reader.load_wave(
        "tb.u0.J_state[3:0]", clock="tb.tck", signed=True, sample_on_posedge=False
    )

    print(j_state)
    assert (j_state.signal == "tb.u0.J_state[3:0]")
    assert (j_state.width == 4)
    assert (j_state.signed == True)

    j_regex = vcd_reader.load_waves(
        r"tb.u0.@J_([a-z]+\[3:0\])", "tb.tck", signed=True, sample_on_posedge=False
    )

    print(j_regex)
    assert len(j_regex) == 2
    for k,v in j_regex.items():
        assert(len(k) == 1 and len(k[0]) == 1)
        assert(k[0][0] in ['next[3:0]','state[3:0]'])
        assert(v.width == 4)
