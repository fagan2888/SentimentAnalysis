import re
import numpy as np

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class DataPreprocessing:
    def __init__(self):
        pass
    
    def clean_data(self, dataset):
        # -- Convert to lower case
        dataset["Tweet"] = dataset["Tweet"].apply(lambda x: x.lower())

        # -- Remove mentions
        dataset['Tweet'] = dataset['Tweet'].apply(lambda x : ' '.join([y for y in x.split() if not y.startswith('@')])) 

        # -- Remove links
        dataset['Tweet'] = dataset['Tweet'].apply(lambda x : ' '.join([y for y in x.split() if not y.startswith('http')])) 

        # -- Remove hashtags
        dataset['Tweet'] = dataset['Tweet'].apply(lambda x : ' '.join([y for y in x.split() if not y.startswith('#')])) 

        # -- Remove weird characters
        dataset["Tweet"] = dataset["Tweet"].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

        # -- Remove AlphaNumeric Words
        dataset["Tweet"] = dataset["Tweet"].apply(lambda x: re.sub('\S*\d\S*', '', x).strip())

        # -- Remove Stopwords
        stop_words = set(stopwords.words('english'))
        dataset["Tweet"] = dataset["Tweet"].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))

        return dataset

    def tokenize_and_pad_data(self, sequences, max_len, alt_sequences=None):
        """
        Description:
            Transforms word sequences to padded integer
            sequences

        Attributes:
            - sequences(numpy array): Dataset to transform.
            e.g: sequences = my_df["my_column"]

            - max_len(int): max length for padding.

            - alt_sequences(numpy array): Feed the tokenizer with
            the data of another dataset.
        """
        if alt_sequences is None:
            return self.tok_and_pad(sequences, max_len)
        else:
            self.tokenizer = Tokenizer(lower = True, filters='')
            self.tokenizer.fit_on_texts(alt_sequences.values)

            # Text to integer
            tokenized_seq = self.tokenizer.texts_to_sequences(sequences)

            # Return padded data
            return pad_sequences(tokenized_seq, maxlen = max_len)

    def tok_and_pad(self, sequences, max_len):
        """
        Description:
            - Auxiliar method of tokenize_and_pad_data.
        """
        self.tokenizer = Tokenizer(lower = True, filters='')
        self.tokenizer.fit_on_texts(sequences.values)

        # Text to integer
        tokenized_seq = self.tokenizer.texts_to_sequences(sequences)

        # Return padded data
        return pad_sequences(tokenized_seq, maxlen = max_len)

    def get_coefs(self, word,*arr):
        return word, np.asarray(arr, dtype='float32')

    def get_embedding_index(self,embedding_file):
        return dict(self.get_coefs(*o.strip().split(" ")) for o in open(embedding_file))

    def create_embedding_matrix(self, vocab_size, embedding_size, word_index, embedding_index):
        embedding_matrix = np.zeros((vocab_size + 1, embedding_size))

        for word, i in word_index.items():
            if i >= vocab_size:
                continue
            embedding_vector = embedding_index.get(word)

            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        return embedding_matrix






