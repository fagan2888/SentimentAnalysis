import pandas as pd

from sklearn.manifold import TSNE
from gensim.models import Word2Vec

from TextPreprocessing import DataPreprocessing

class DataLogic:
    """
    Description:
        Handles all necessary operations for app.py to work. It does that
        by calling all necessary classes.
        
    Attributes:
        - train_ds(String): path to train dataset.
        - test_ds(String): path to test dataset
    """
    def __init__(self, train_ds, test_ds):
        self.train_ds = train_ds
        self.test_ds = test_ds

    def text_preproc(self):
        """
        Description:
            Runs all necessary preprocessing for cleaning and
            tokenizing the tweets.
            
        """
        test_df = pd.read_csv(self.test_ds,
                               names = ["Polarity", "Id", "Date", "Query", "User", "Tweet"],
                               encoding = "ISO-8859-1")

        train_df = pd.read_csv(self.train_ds,
                               names = ["Polarity", "Id", "Date", "Query", "User", "Tweet"],
                               encoding = "ISO-8859-1")

        test_df = test_df[["Polarity", "Tweet"]]
        train_df = train_df[["Polarity", "Tweet"]]

        # Select Columns
        test_df = test_df[["Polarity", "Tweet"]]
        train_df = train_df[["Polarity", "Tweet"]]

        # Drop Neutral Values (0: negative; 4: positive)
        test_df = test_df[test_df.Polarity != 2]

        # Create Dropdown list
        self.dropdown_options = test_df["Tweet"].values

        # Clean Data
        data_preproc = DataPreprocessing()
        test_df = data_preproc.clean_data(test_df)
        train_df = data_preproc.clean_data(train_df)

        # Tokenize and Pad data
        max_len = 140
        X_test = data_preproc.tokenize_and_pad_data(test_df["Tweet"], max_len, alt_sequences = train_df["Tweet"])

        # Get Polarity. 0: negative; 4: positive
        Y_test = test_df["Polarity"].values

        return X_test, Y_test, test_df

    def get_dropdown(self):
        """
        Description:
            Returns a list of all tweets in test dataset.
        """
        return self.dropdown_options

    def count_pos_neg(self, Y):
        """
        Description:
            Returns the number of positives and negatives
            of Y.
            
        Args:
            - Y(numpy array or list): polarity values
        """
        pos, neg = 0, 0

        for i in Y:
            if i == 0:
                neg += 1
            else:
                pos += 1

        return pos, neg

class GraphLogic:
    """
    Description:
        Handles all neccesary operations for the Graph in app.py to work.

    Attributes:
        - path(String): path to word2vec model. 
    """
    def __init__(self, path):
        self.path = path

    def load_w2v(self):
        """
        Description:
            Load word2vec model from path
        """
        self.w2v = Word2Vec.load(self.path)

    def create_word_vectors(self, number_words):
        """
        Description:
            Returns a numpy array of word_vectors of length
            number_words.

        Args:
            - number_words(int): number of words to extract from the
            vocabulary of word2vec model.
        """
        self.word_vectors = [self.w2v[word] for word in self.w2v.wv.vocab]
        self.word_vectors = self.word_vectors[:number_words]

        return self.word_vectors

    def create_word_vocab(self, number_words):
        """
        Description:
            Returns word vocabulary of word2vec model of lenght
            number_words.

        Args:
            - number_words(int): number of words to extract from the
            vocabulary of word2vec model.        
        """
        self.word_vocab = list(self.w2v.wv.vocab)[:number_words]

        return self.word_vocab

    def create_tsne(self):
        """
        Description:
            Creates and trains a tsne model based on self.word_vectors.
            Returns the model.
        """
        tsne = TSNE(perplexity = 35, n_components = 3, verbose = 1, random_state = 0)
        self.tsne_w2v = tsne.fit_transform(self.word_vectors)

        return self.tsne_w2v

    def get_dimensions(self):
        """
        Description:
            Returns the x, y and z dimensions of the vectors in the
            word2vec model
        """
        x_w2v = [point[0] for point in self.tsne_w2v]
        y_w2v = [point[1] for point in self.tsne_w2v]
        z_w2v = [point[2] for point in self.tsne_w2v]

        return x_w2v, y_w2v, z_w2v

