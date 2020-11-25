import sounddevice as sd
from scipy.io.wavfile import write
import pandas as pd
import numpy as np
import time

from utilities.annotations import Seconds, Recording
from utilities.enums import WaterState
from utilities.constants import fs


def unique_file_name_for_(sample: WaterState, uuid: int) -> str:
    return f"{uuid}_{sample.name.title()}.wav"


def record_sample_for_(length: Seconds) -> Recording:
    print("\nRecording...\n")
    recording: Recording = sd.rec(int(length * fs), samplerate=fs, channels=1)
    sd.wait()
    return recording


def save_sample_to_file_(filename: str, audio: Recording):
    write(filename, fs, audio)


def record_sample_pack():
    uuid: int = int(time.time())
    sample_length: Seconds = int(input("Enter sample length (seconds): "))

    input(f"Make sure it's quiet, then press enter to record for {sample_length} seconds...")
    off_array = record_sample_for_(sample_length)

    input(f"Set the faucet to drip, then press enter to record for {sample_length} seconds...")
    drip_array = record_sample_for_(sample_length)

    input(f"Turn the faucet on, then press enter to record for {sample_length} seconds...")
    full_array = record_sample_for_(sample_length)

    sample_df = pd.DataFrame(data=np.hstack((off_array, drip_array, full_array)),
                             columns=[WaterState.OFF, WaterState.DRIP, WaterState.FULL])

    try:
        save_data = int(input("Do you want to save this (1 = yes, other = no): "))
        if save_data == 1:
            save_sample_to_file_(unique_file_name_for_(WaterState.OFF, uuid), off_array)
            save_sample_to_file_(unique_file_name_for_(WaterState.DRIP, uuid), drip_array)
            save_sample_to_file_(unique_file_name_for_(WaterState.FULL, uuid), full_array)

            sample_df.to_csv(f"{uuid}_data.csv")

            print("\nData has been saved!\n")
    except ValueError:
        print("See you next time!")


if __name__ == '__main__':
    record_sample_pack()
