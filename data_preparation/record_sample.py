import sounddevice as sd
from scipy.io.wavfile import write
import time

from enum import Enum, auto
from typing import Annotated

Seconds = Annotated[int, 'seconds']


class AutoName(Enum):
    def _generate_next_value_(self, start, count, last_values):
        return self


class WaterState(AutoName):
    OFF = auto()
    DRIP = auto()
    FULL = auto()


fs = 44100  # Sample rate


def unique_file_name_for_(sample: WaterState, uuid: int) -> str:
    return f"{uuid}_{sample.name.title()}.wav"


def record_sample_for_(filename: str, length: Seconds):
    recording = sd.rec(int(length * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, recording)


def record_sample_pack():
    uuid: int = int(time.time())
    sample_length: Seconds = int(input("Enter sample length (seconds): "))

    input(f"Make sure it's quiet, then press enter to record for {sample_length} seconds...")
    record_sample_for_(unique_file_name_for_(WaterState.OFF, uuid), sample_length)

    input(f"Set the faucet to drip, then press enter to record for {sample_length} seconds...")
    record_sample_for_(unique_file_name_for_(WaterState.DRIP, uuid), sample_length)

    input(f"Turn the faucet on, then press enter to record for {sample_length} seconds...")
    record_sample_for_(unique_file_name_for_(WaterState.FULL, uuid), sample_length)


if __name__ == '__main__':
    record_sample_pack()
