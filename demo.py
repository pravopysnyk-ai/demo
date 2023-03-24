import os
import sys
import torch
import spacy
import spacy_udpipe
import pymorphy2
import contextlib
from symspellpy import SymSpell
from pathlib import Path
from transformers import pipeline
from helpers.classes.SymSpellUkr import SymSpellUkr
from helpers.classes.SpaceHandler import SpaceHandler

def main():
    # transformers pipeline won't work without CUDA
    # checks if the environment is set up properly
    if not torch.cuda.is_available():
        print("The local Transformers pipeline will not work without CUDA.\nPlease make sure you have everything set up properly!\n")
        sys.exit()

    # setting up the model
    device = torch.device('cuda') # setting device on GPU if available, else CPU
    model_checkpoint = "Pravopysnyk/best-unlp" # importing the correct model from Huggingface
    translator = pipeline("translation", src_lang="uk_UA", tgt_lang="uk_UA", model=model_checkpoint, device=0) # setting up the translation pipeline

    # setting up the helper modules
    dicts_dir = "./helpers/dicts"
    models_dir = "./helpers/modules"
    ukrainian_dict_path = dicts_dir + "/new_wordlist.json"
    freq_path = dicts_dir + "/trim-frequency-vocab.txt"
    bigrams_folder_path = dicts_dir + "/bigrams_parts"
    spacy_model_path = models_dir + "/ukrainian-iu-ud-2.5-191206.udpipe"
    SPACY_UDPIPE_MODEL = spacy_udpipe.load_from_path(
        lang="uk",
        path=spacy_model_path,
    ) # initializing the spacy model
    morph = pymorphy2.MorphAnalyzer(lang='uk') # initializing pymorphy
    s = SymSpellUkr(SymSpell(), bigrams_folder_path, freq_path, ukrainian_dict_path, SpaceHandler(), SPACY_UDPIPE_MODEL) # initializing the spelling module

    # printing the use information on the screen
    print("Welcome to Pravopysnyk!\nPlease enter your sentence below.\nTo exit the program, press Enter with no input.\n\n\n")

    # while loop for running the program
    while True:
        errorful_sentence = input("Your sentence: ") # user input
        if len(errorful_sentence) == 0: # if empty, exit
            break
        else:
            spellified = s.spellify(errorful_sentence) # else use spelling
            corrected = translator(spellified, batch_size=16) # and our model to correct it
            print("Corrected sentence: " + corrected[0]['translation_text'] + "\n") # and print out
            continue
    return

if __name__=="__main__":
    main()
