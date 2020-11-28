# SVM Audio Classification

AOA Final Project, using an SVM (Support Vector Machine) to classify audio from a kitchen sink.

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

This is a script that walks you through recording a sample of `wav` files and a `csv` file for the data you need.
It uses your computer's built in microphone default to record, my Mac's was good enough to provide good results.
The prompt is as follows:


```
Enter sample length (seconds): 1
Make sure it's quiet, then press enter to record for 1 seconds...

Recording...

Set the faucet to drip, then press enter to record for 1 seconds...

Recording...

Turn the faucet on, then press enter to record for 1 seconds...

Recording...

Do you want to save this (1 = yes, other = no): 1

Data has been saved!
```

Warnings:
- You may need to change the number of channels you have for recording

### `plot_audio_real_time.py`

This was a file that I borrowed from the `sounddevice` Python libraries docs.
It helped me debug any audio issues I was having and got me familiar with the library.
It should run just fine with it's configuration, fun to try out honestly!

## Part 2: Cleaning/Preparing the data

Now that we have our samples and our file of raw input, we need to generate some useful information from it.
I hope to do this using the [Librosa Package](https://librosa.org/doc/latest/index.html).

### In Search of Features

This is a Jupyter notebook that documented my (very) creative process of feature selection for the audio samples I made.
It can be run the following way if you already have a virtual environment with the requirements installed:
- `jupyter-notebook`
- navigate to where the `.ipynb` file is and open it
- press `Run All`

### `features.py`

This uses the findings from `In Search of Features` to build out a csv that will be useful for training our data.
It finds the amplitudes and peak count for each of our samples and then stores them in a dataframe.
It also plots the results, colored by state type (off, drip, on) on a scatter plot.
The first thing I noticed is that I need more data!

[Scatter plot!](app/feature_extraction/water%20scatter.png)

 
