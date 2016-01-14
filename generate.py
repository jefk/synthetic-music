from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.optimizers import SGD
import numpy as np
import random
import sys

def with_training(training_data):
    print('corpus length:', len(training_data))

    tokens = set(training_data) # set of all notes
    print('total tokens:', len(tokens))
    token_indexes = dict((c, i) for i, c in enumerate(tokens))
    indexes_token = dict((i, c) for i, c in enumerate(tokens))

    # cut the training_data in semi-redundant sequences of maxlen characters
    maxlen = 20
    step = 1
    sentences = []
    next_tokens = []
    for i in range(0, len(training_data) - maxlen, step):
        sentences.append(training_data[i: i + maxlen])
        next_tokens.append(training_data[i + maxlen])
    print('nb sequences:', len(sentences))

    print('Vectorization...')
    X = np.zeros((len(sentences), maxlen, len(tokens)), dtype=np.bool)
    y = np.zeros((len(sentences), len(tokens)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, token_indexes[char]] = 1
        y[i, token_indexes[next_tokens[i]]] = 1


    # build the model
    print('Build model...')
    model = Sequential()
    model.add(Flatten(input_shape=(maxlen, len(tokens))))
    model.add(Dense(64, init="glorot_uniform"))
    model.add(Activation('tanh'))
    model.add(Dense(output_dim=len(tokens), init="glorot_uniform"))
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

    start_index = random.randint(0, len(training_data) - maxlen - 1)

    diversity = 0.75
    print()
    print('----- diversity:', diversity)

    sentence = training_data[start_index: start_index + maxlen - 1]
    generated = sentence
    print('----- Generating with seed: {sentence!s}'.format(**locals()))
    print()

    for i in range(100):
        x = np.zeros((1, maxlen, len(tokens)))
        for t, char in enumerate(sentence):
            x[0, t, token_indexes[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indexes_token[next_index]

        generated.append(next_char)
        sentence = sentence[1:]
        sentence.append(next_char)

    return generated
