# Import Tokenizer and pad_sequences
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Import numpy and pandas
import numpy as np
import pandas as pd
import io


if __name__ == '__main__':
    path = tf.keras.utils.get_file('reviews.csv',
                                   'https://drive.google.com/uc?id=13ySLC_ue6Umt9RJYSeM2t-V0kCv-4C-P')

    dataset = pd.read_csv(path)
    sentences = dataset['text'].tolist()
    labels = dataset['sentiment'].tolist()

    training_size = int(len(sentences) * 0.8)

    training_sentences = sentences[:training_size]
    testing_sentences = sentences[training_size:]
    training_labels = labels[:training_size]
    testing_labels = labels[training_size:]

    training_labels_final = np.array(training_labels)
    testing_labels_final = np.array(testing_labels)

    vocab_size = 1000
    embedding_dim = 16
    max_length = 100
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)

    word_index = tokenizer.word_index

    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded = pad_sequences(sequences, padding=padding_type, truncating=trunc_type, maxlen=max_length)

    test_sequences = tokenizer.texts_to_sequences(testing_sentences)
    test_padded = pad_sequences(test_sequences, padding=padding_type, truncating=trunc_type, maxlen=max_length)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    EPOCHS = 10
    model.fit(padded, training_labels_final, epochs=EPOCHS, validation_data=(test_padded, testing_labels_final))
    model.save('saved_models\\embedding')

    model = tf.keras.models.load_model('saved_models\\embedding')

    e = model.layers[0]
    weights = e.get_weights()[0]

    out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
    out_m = io.open('meta.tsv', 'w', encoding='utf-8')
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    for word_num in range(1, vocab_size):
        word = reverse_word_index[word_num]
        embeddings = weights[word_num]
        out_m.write(word + "\n")
        out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
    out_v.close()
    out_m.close()