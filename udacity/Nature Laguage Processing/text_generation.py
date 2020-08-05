import tensorflow as tf
from numpy.core._multiarray_umath import dtype

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Other imports for processing data
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def tokenize_corpus(corpus, num_word=-1):
    # Fit a tokenizer on the corpus
    if num_word > -1:
        tokenizer = Tokenizer(num_words=num_word)
    else:
        tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    return tokenizer


def create_lyrics_corpus(dataset, field):
    # Remove all other punctuation
    dataset[field] = dataset[field].str.replace('[{}]'.format(string.punctuation), '')
    # Make it lower case
    dataset[field] = dataset[field].str.lower()
    # Make it one long string to split by line
    lyrics = dataset[field].str.cat()
    corpus = lyrics.split('\n')
    # remove any trailing whitespace
    for l in range(len(corpus)):
        corpus[l] = corpus[l].rstrip()
    # remove any empty line
    corpus = [ l for l in corpus if l != '']
    return corpus


if __name__ == '__main__':
    path = tf.keras.utils.get_file('reviews.csv',
                                   'https://drive.google.com/uc?id=1LiJFZd41ofrWoBtW-pMYsfz1w8Ny0Bj8')

    dataset = pd.read_csv(path, dtype=str)[:250]
    corpus = create_lyrics_corpus(dataset, 'text')

    tokenizer = tokenize_corpus(corpus)

    total_words = len(tokenizer.word_index) + 1

    sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequences = token_list[:i+1]
            sequences.append(n_gram_sequences)

    max_sequences_len = max([len(seq) for seq in sequences])
    sequences = np.array(pad_sequences(sequences, maxlen=max_sequences_len, padding='pre'))

    input_sequences, labels = sequences[:, :-1], sequences[:, -1]

    one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=total_words)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(total_words, 64, input_length=max_sequences_len-1),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
        tf.keras.layers.Dense(total_words, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    history = model.fit(input_sequences, one_hot_labels, epochs=100, verbose=1)
    model.save('saved_models\\text_generation')

    plt.plot(history.history['accuracy'])
    plt.xlabel("Epochs")
    plt.ylabel('accuracy')
    plt.show()