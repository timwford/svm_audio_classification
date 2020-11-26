import pandas as pd

from utilities.annotations import Recording
from utilities.enums import WaterState

filename = "data/data.csv"

def read_sample_recordings() -> (list[Recording], pd.DataFrame):
    df = pd.read_csv(filename)
    off, drip, full = df[str(WaterState.OFF)], df[str(WaterState.DRIP)], df[str(WaterState.FULL)]
    samples: list[Recording] = [off.to_numpy(), drip.to_numpy(), full.to_numpy()]
    return samples, df


if __name__ == '__main__':
    recordings, audio_df = read_sample_recordings()

    for r in recordings:
        print(r)
