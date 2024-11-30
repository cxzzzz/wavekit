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
        "tb.u0.@J_([a-z]+)\[3:0\]", "tb.tck", signed=True, sample_on_posedge=False
    )

    print(j_regex)
    assert len(j_regex) == 2
