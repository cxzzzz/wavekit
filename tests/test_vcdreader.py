import pytest
from wavekit.reader import VcdReader


def test_vcdreader():
    """
    This defines the expected usage, which can then be used in various test cases.
    Pytest will not execute this code directly, since the function does not contain the suffex "test"
    """

    vcd_reader = VcdReader("tests/testdata/jtag.vcd")

    j_state = vcd_reader.load_wave(
        "tb.u0.J_state[3:0]", clock="tb.tck", signed=True, sample_on_posedge=False
    )

    print(j_state)

    j_state = vcd_reader.load_waves(
        "tb.u0.J_<[a-z]+>", "tb.tck", signed=True, sample_on_posedge=False
    )

    print(j_state)
