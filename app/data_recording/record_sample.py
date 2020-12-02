import sounddevice as sd
from scipy.io.wavfile import write
import pandas as pd
import numpy as np
import time

from feature_extraction.features import generate_features_for_

from utilities.annotations import Seconds, Recording
from utilities.enums import WaterState
from utilities.constants import fs, sample_length, prompt


def unique_file_name_for_(sample: WaterState, uuid: int) -> str:
    return f"data/{uuid}_{sample.name.title()}.wav"


def scale_recording(audio: Recording) -> Recording:
    louder_audio = np.multiply(audio, 1.2)
    return louder_audio


def record_sample_for_(length: Seconds) -> Recording:
    print("\nRecording...\n")
    audio: Recording = sd.rec(int(length * fs), samplerate=fs, channels=1)
    sd.wait()
    return audio


def save_sample_to_file_(filename: str, audio: Recording):
    write(filename, fs, audio)


def record_off_sample_for_(sample_length: Seconds) -> Recording:
    input(f"Make sure it's quiet, then press enter to record for {sample_length} seconds...")
    return record_sample_for_(sample_length)


def record_drip_sample_for_(sample_length: Seconds) -> Recording:
    input(f"Set the faucet to drip, then press enter to record for {sample_length} seconds...")
    return record_sample_for_(sample_length)


def record_on_sample_for_(sample_length: Seconds) -> Recording:
    input(f"Turn the faucet on, then press enter to record for {sample_length} seconds...")
    return record_sample_for_(sample_length)


def record_sample_triple():
    uuid: int = int(time.time())
    length: Seconds = int(input("Enter sample length (seconds): "))

    off_array = record_off_sample_for_(length)
    drip_array = record_drip_sample_for_(length)
    on_array = record_on_sample_for_(length)

    sample_df = pd.DataFrame(data=np.hstack((off_array, drip_array, on_array)),
                             columns=[WaterState.OFF, WaterState.DRIP, WaterState.FULL])

    try:
        save_data = int(input("Do you want to save this (1 = yes, other = no): "))
        if save_data == 1:
            save_sample_to_file_(unique_file_name_for_(WaterState.OFF, uuid), off_array)
            save_sample_to_file_(unique_file_name_for_(WaterState.DRIP, uuid), drip_array)
            save_sample_to_file_(unique_file_name_for_(WaterState.FULL, uuid), on_array)

            sample_df.to_csv(f"data/{uuid}_data.csv")

            print("\nData has been saved!\n")
    except ValueError:
        print("Ok, see you next time!")


class Sample:
    def __init__(self, recording: Recording, classification: WaterState):
        self.recording = recording
        self.classification = classification


def get_input() -> int:
    try:
        return int(input(prompt))
    except ValueError:
        print("huh?")
        return 3


if __name__ == '__main__':
    uuid: int = int(time.time())
    samples: list[Sample] = []

    response = get_input()
    while response >= 0:
        if response == 0:
            samples.append(Sample(record_off_sample_for_(sample_length), WaterState.OFF))
        elif response == 1:
            samples.append(Sample(record_drip_sample_for_(sample_length), WaterState.DRIP))
        elif response == 2:
            samples.append(Sample(record_on_sample_for_(sample_length), WaterState.ON))

        response = get_input()

    amplitudes: list[float] = []
    peak_counts: list[int] = []
    classifications: list[str] = []

    for s in samples:
        amplitude, peak_count = generate_features_for_(s.recording)
        amplitudes.append(amplitude)
        peak_counts.append(peak_count)
        classifications.append(s.classification.name)

    row_dict = {'amplitude': amplitudes, 'peak_count': peak_counts, 'classification': classifications}

    sample_df = pd.DataFrame(data=row_dict, columns=["amplitude", "peak_count", "classification"])
    sample_df.to_csv(f"data/{uuid}_features.csv")
