# Project: Comparative Analysis of Single-Task and Multi-Task Learning for Air Pollutant Time Series Prediction

## Members: Lavanya Middha, Snigdha Challapalli, Harsh More (Group 3)

## Set Up:

Create a python environment using the file `environment.yaml`. This should install all the required libraries. Make sure to use python 3.11 and install nvidia-cuda drivers

## Running LSTM:

- Running LSTM notebooks is very straightforard.
- These can also be run using collab.
- You might have to change the directory for train, test and validation files. These files have been provided in the data folder.
- All the notebooks containing the output results have been uploaded to GitHub for Single-task LSTM
- For Multi-Task, to verify the results for 7 day forcast, one would need to update `SEQ_LEN = 7*24` before running the code

## Running SOTA:

- Running LSTM notebooks is very straightforard.
- These can also be run using collab.
- You might have to change the directory for train, test and validation files. These files have been provided in the data folder.
- You will have to use GPU for faster training.
- All the notebooks containing the output results have been uploaded to GitHub

## Running TFT:

- If running TFT on HPC, you will have to create the environment using the environment.yaml file.
- You would need an interactive GPU session to run the notebooks
- Most of the training was done using GPU XTX and GPU A10.
- All the notebooks containing the output results have been uploaded to GitHub
- There is also a file called Evaluate-Results to run the checkpoints provided.

## Python Libraries Required:

- torch
- torchvision
- torchaudio
- lightning
- pytorch-forecasting
- pandas
- numpy
- tensorboard
- optuna
- tensorboardx
- nvidia-cuda driver

If running on collab, at this at the top: `!pip install torch torchvision torchaudio lightning pytorch-forecasting pandas numpy tensorboard optuna tensorboardx`
