# SVM Audio Classification

AOA Final Project, using an SVM (Support Vector Machine) to classify audio from a sink

# Project structure

There are configurations for everything you need to run in Pycharm.
There is a `requirements.txt` that has all of the pip packages you need.
Lastly, this depends on `Python 3.9` for type annotations!

## Part 1: Getting the data

The goal is to classify audio data from a kitchen sink.
We will try to classify data into three categories:

1. Off - the sink is completely off, only ambient noise
2. Drip - the sink is turned such that there is a slight but consistent drip
3. Full - the sink is turned on fully

First, we'll write a little script that helps us get this data/

### `record_sample.py`

TODO

### `plot_audio_real_time.py`

This was a file that I borrowed from the `sounddevice` Python libraries docs.
It helped me debug any audio issues I was having and got me familiar with the library.
It should run just fine with it's configuration, fun to try out honestly!