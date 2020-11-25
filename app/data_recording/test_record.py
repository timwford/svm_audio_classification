from .record_sample import unique_file_name_for_
from utilities.enums import WaterState


def test_test():
    assert 2 == 2


def test_file_name():
    assert unique_file_name_for_(WaterState.FULL, 1) == "data/1_Full.wav"
