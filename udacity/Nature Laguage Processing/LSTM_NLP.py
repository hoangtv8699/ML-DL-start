# Import Tokenizer and pad_sequences
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Import numpy and pandas
import numpy as np
import pandas as pd
import io
import matplotlib.pylab as plt


def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Loss')
    plt.show()


if __name__ == '__main__':
    path = tf.keras.utils.get_file('reviews.csv',
                                   'https://drive.google.com/uc?id=13ySLC_ue6Umt9RJYSeM2t-V0kCv-4C-P')

    dataset = pd.read_csv(path)
    sentences = dataset['text'].tolist()
    labels = dataset['sentiment'].tolist()

    vocab_size = 1000
    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(sentences, vocab_size, max_subword_length=5)
    for i, sentence in enumerate(sentences):
        sentences[i] = tokenizer.encode(sentence)

    embedding_dim = 16
    max_length = 50
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"

    sequences_padded = pad_sequences(sentences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    training_size = int(len(sentences) * 0.8)

    training_sentences = sequences_padded[:training_size]
    testing_sentences = sequences_padded[training_size:]
    training_labels = labels[:training_size]
    testing_labels = labels[training_size:]

    training_labels_final = np.array(training_labels)
    testing_labels_final = np.array(testing_labels)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    EPOCHS = 30
    history = model.fit(
        training_sentences, training_labels_final,
        epochs=EPOCHS, validation_data=(testing_sentences, testing_labels_final)
    )

    model.save('saved_models\\multilayer_LSTM_NLP')

    plot_history(history)

