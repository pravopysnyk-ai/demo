# Pravopysnyk Model Demo

This repository provides code for running the state-of-the-art model for grammatical error correction for the Ukrainian language.

It is mainly based on `PyTorch` and `transformers`.

## Requirements

describe requirements for cuda here

## Installation

First, you need to initialize the helper submodule directory:

`git submodule update --init`

Then, run the following command to install all necessary packages:

`pip install -r requirements.txt`

The project was tested using Python 3.7.

## Testing

To use the demo, simply run this command:

`python demo.py`

After quick initialization, the program will print the welcome message and ask for input. 
Expected input is supposed to be a full sentence in the Ukrainian language, so the program might handle invalid input incorrectly.
To exit the program, simply press Enter with no input.
