import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
import librosa.display
import librosa
from scipy.signal import find_peaks
from scipy.stats import stats

from utilities.annotations import Recording
from utilities.enums import WaterState

filename = "water_data/data.csv"
filename_hard = "water_data_harder/data.csv"


def read_sample_recordings(f: str) -> (list[Recording], pd.DataFrame):
    df = pd.read_csv(f)
    off, drip, full = df[str(WaterState.OFF)], df[str(WaterState.DRIP)], df[str(WaterState.FULL)]
    samples: list[Recording] = [off.to_numpy(), drip.to_numpy(), full.to_numpy()]
    return samples, df


def show_spectogram_for_(audio: Recording):
    D = librosa.stft(audio)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    plt.figure()
    librosa.display.specshow(S_db)
    plt.colorbar()
    plt.show()


def show_amplitude_for_(audio: Recording):
    S, phase = librosa.magphase(librosa.stft(audio))
    plt.plot(S)
    plt.show()


def get_amplitude_for_(audio: Recording):
    S, phase = librosa.magphase(librosa.stft(audio))
    return mean([np.max(a) for a in S])


def line(slope, intercept):
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')


def show_tempogram_for_(audio: Recording):
    t = librosa.feature.tempogram(audio)

    avgs = []
    for a in t:
        avgs.append(np.mean(a))

    avgs = [exp for exp in avgs]
    inds = [i for i in range(0, len(avgs))]

    slope, intercept, r_value, p_value, std_err = stats.linregress(inds, avgs)

    peaks = find_peaks(avgs, prominence=0.05)
    print(f"Peak count: {len(peaks[0])}")

    plt.plot(avgs)
    x = peaks[0]
    y = [avgs[peak] for peak in peaks[0]]
    plt.plot(x, y, "x")

    line(slope, intercept)
    plt.show()


def get_tempo_peak_count_for_(audio: Recording):
    t = librosa.feature.tempogram(audio)
    avgs = []
    for a in t:
        avgs.append(np.mean(a))

    avgs = [exp for exp in avgs]
    peaks = find_peaks(avgs, prominence=0.05)
    return len(peaks[0])


def generate_features_for_(audio: Recording) -> (float, int):
    amplitude = get_amplitude_for_(audio)
    peak_count = get_tempo_peak_count_for_(audio)

    return amplitude, peak_count

def generate_data_set_for_(recordings: [Recording]) -> pd.DataFrame:
    results = []

    for i in range(0, len(recordings)):
        amplitudes, peak_count = generate_features_for_(recordings[i])
        results.append([amplitudes, peak_count, i])

    return pd.DataFrame(results, columns=["amp", "peaks", "class"])


if __name__ == '__main__':
    recordings_ez, audio_df_ez = read_sample_recordings(filename)
    recordings_hard, audio_hard_df = read_sample_recordings(filename_hard)

    ez_df = generate_data_set_for_(recordings_ez)
    hard_df = generate_data_set_for_(recordings_hard)

    df = ez_df.append(hard_df)

    off = df[df['class'] == 0]
    drip = df[df['class'] == 1]
    on = df[df['class'] == 2]

    plt.scatter(x=off['amp'], y=off['peaks'], c='red')
    plt.scatter(x=drip['amp'], y=drip['peaks'], c='blue')
    plt.scatter(x=on['amp'], y=on['peaks'], c='green')
    plt.show()

    df.to_csv("dataset.csv")
