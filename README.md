# Pravopysnyk Model Demo

This repository provides code for running the state-of-the-art model for grammatical error correction for the Ukrainian language.

It is mainly based on `PyTorch` and `transformers`. 
The schema behind the model can be found in the following paper:

> [Comparative Study of Models Trained on Synthetic Data for Ukrainian Grammatical Error Correction.](https://aclanthology.org/2023.unlp-1.13/)
>
> [Maksym Bondarenko](https://github.com/Lenguist), [Artem Yushko](https://github.com/artemyushko), [Andrii Shportko](https://github.com/Antebe), and [Andrii Fedorych](https://github.com/StopFuture). 2023.
>
> In Proceedings of the Second Ukrainian Natural Language Processing Workshop (UNLP), pages 103–113, Dubrovnik, Croatia. Association for Computational Linguistics.


## Requirements

Our language models work only in CUDA-enabled environment. To set everything up, follow the [instructions](https://docs.nvidia.com/cuda/) from the official NVIDIA website.

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
