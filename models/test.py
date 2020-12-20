from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import models
from tensorflow.keras import utils

import numpy as np
import string
from pathlib import Path


# Some constants required for is_text_positive()
word_to_index = imdb.get_word_index()
oov_char = 2  # Should be equal to 'oov_char' passed to mdb.load_data() during training
maxlen = 400  # Should be equal to 'maxlen' used for training
num_words = 5000  # Should be equal to 'num_words' passed to mdb.load_data() during training
start_char = 1  # Should be equal to 'start_char' passed to mdb.load_data() during training


def is_text_positive(model, text):
    # Change all letters to lower case, remove punctuation, split the text into separate words
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()

    # Obtain indices of the words, replace those that are not from the dictionary, prepend start_char.
    # Note, that according to tf.keras.datasets.imdb docs, words that are not present in the full vocabulary (in
    # contrast to those that were excluded due to num_words limitation, for example) are marked with 0 instead of
    # oov_char, so we replicate the same behavior here).
    indices = [start_char] + [word_to_index[word] if word in word_to_index else 0 for word in words]

    # Remove indices of words that were not included into training due to max_words limitation
    if num_words is not None:
        indices = [index if index <= num_words else oov_char for index in indices]

    indices = np.asarray(indices)[np.newaxis, :]

    # Some models might not be invariant to the input shape, so the input is either padded or truncated to the maxlen
    # (in the same way that was used during training)
    if indices.shape[1] > maxlen:
        indices = indices[:, :maxlen]
    elif indices.shape[1] < maxlen:
        indices = sequence.pad_sequences(indices, maxlen=maxlen)

    return model(indices).numpy()[0][0]
