import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100  # Sample rate
seconds = 2  # Duration of recording

if __name__ == '__main__':
    print("Sweet")

    print(sd.query_devices())

    my_recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()

    print(my_recording.type)
    print(my_recording.reshape((-1,1)))

    write('output.wav', fs, my_recording)  # Save as WAV file
