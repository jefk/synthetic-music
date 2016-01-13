'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.datasets.data_utils import get_file
from keras.optimizers import SGD
import numpy as np
import random
import sys

def with_training(training_data):
    text = training_data
    print('corpus length:', len(text))

    chars = set(text)
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of maxlen characters
    maxlen = 20
    step = 1
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))

    print('Vectorization...')
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1


    # build the model
    print('Build model...')
    model = Sequential()
    model.add(Flatten(input_shape=(maxlen, len(chars))))
    model.add(Dense(64, init="glorot_uniform"))
    model.add(Activation('tanh'))
    model.add(Dense(output_dim=len(chars), init="glorot_uniform"))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)

    def sample(a, temperature=1.0):
        # helper function to sample an index from a probability array
        a = np.log(a) / temperature
        a = np.exp(a) / np.sum(np.exp(a))
        return np.argmax(np.random.multinomial(1, a, 1))

    print()
    print('-' * 50)
    model.fit(X, y, batch_size=16, nb_epoch=100)

    start_index = random.randint(0, len(text) - maxlen - 1)

    diversity = 0.75
    print()
    print('----- diversity:', diversity)

    sentence = text[start_index: start_index + maxlen - 1]
    generated = sentence
    print('----- Generating with seed: {sentence!s}'.format(**locals()))
    print()

    for i in range(100):
        x = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]

        generated.append(next_char)
        sentence = sentence[1:]
        sentence.append(next_char)

    return generated
