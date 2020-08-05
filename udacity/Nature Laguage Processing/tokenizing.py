from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences

if __name__ == '__main__':
    sentences = [
        'my favourite food is ice cream',
        'do you like ice cream too?',
        'My dog like ice cream!',
        'your favourite flavor of ice cream is chocolate',
        "chocolate isn't good for dogs",
        "your dog, your cat, and your parrot prefer broccoli"
    ]

    print(sentences)

    tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)
    print(tokenizer.word_index)
    sequences = tokenizer.texts_to_sequences(sentences)
    print(sequences)

    padded = pad_sequences(sequences, maxlen=15, padding="post")
    print("Padded Sequences")
    print(padded)