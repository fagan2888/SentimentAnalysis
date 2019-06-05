from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, LSTM, Dropout, Embedding, SpatialDropout1D


class Model:
    """
    Description:
        Creates a basic LSTM model.
    
    Attributes:
        - vocab_size: vocabulary size for the embedding layer.
        - input_len: input length for the embedding layer.
        - embed_matrix: Embedded matrix for preloaded weights.
        - output_dimension: Output dimension for the embedding layer.
        - lstm_cells: number of cells for the LSTM layer.

    Args:
        - load (boolean): True for load a trained model,
        False to create one using Keras.
        - model_name: Path/Name of the model to be loaded.
    """
    def __init__(self,
                 vocab_size = None,
                 input_len = None,
                 embed_matrix = None,
                 output_dimension = None,
                 lstm_cells = None,
                 load = False,
                 model_name=""):

        if load:
            self.model = load_model(model_name)
        else:
            self.model = Sequential([
                Embedding(input_dim = vocab_size + 1,
                                     input_length = input_len,
                                     weights = [embed_matrix],
                                     output_dim = output_dimension,
                                     trainable = False),

                SpatialDropout1D(0.2),
                
                LSTM(lstm_cells, dropout = 0.2,
                                recurrent_dropout = 0.2),

                Dense(2, activation = "softmax")
            ])
    
    def summary(self):
        return self.model.summary()

    def compile_model(self, opt, loss_function):
        """
        Description:
            Configures the model for training.

        Args:
            - opt (String): optimizer to use.
            - loss_function (String): loss function to use.
        """
        self.model.compile(optimizer = opt,
                           loss = loss_function,
                           metrics = ["accuracy"])

    def train_model(self, X_train, Y_train, X_test, Y_test, epochs, batch_size):
        """
        Description:
            Trains the model for a given number of epochs.

        Args:
            - X_train(numpy array): training data.
            - Y_train(numpy array): target data.
            - X_test(numpy array): validation data.
            - Y_test(numpy array): validation target
            - batch_size(int): samples per gradient update.
        """
        self.model.fit(X_train, Y_train,
                       epochs = epochs,
                       batch_size = batch_size,
                       validation_data = (X_test, Y_test))

    def save_model(self, name):
        """
        Description:
            Saves the model.

        Args:
            - name: path to save.
        """
        self.model.save(name)

    def make_prediction(self, X):
        """
        Description:
            Makes a prediction for a given input.

        Args:
            - X (numpy array): input data.
        """
        return self.model.predict(X)


