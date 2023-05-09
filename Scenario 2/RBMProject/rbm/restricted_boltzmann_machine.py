import numpy as np
import progressbar

from helper_functions import training_status
from helper_functions import batch_iterator
from rbm.activation_functions import Sigmoid


class RBM():
    """
    Class for general Restricted Boltzmann Machine
    """
    def __init__(self, n_hidden: int = 128, eta: float = 0.1, batch_size: int = 10, epochs: int = 100, act_fn=Sigmoid()):
        """
        Initialize the RBM with relevant hyperparameters.
        :param n_hidden: int - the number of hidden units
        :param eta: float - learning rate parameter
        :param batch_size: int - batch size of how many samples to train with
        :param epochs: int - the number of iterations to train over
        :param act_fn: func - the activation function to use
        """
        self.epochs: int = epochs
        self.batch_size: int = batch_size
        self.eta: float = eta
        self.n_hidden: int = n_hidden
        self.status = progressbar.ProgressBar(widgets=training_status)
        self.act_fn = act_fn

    def _initialize_weights_and_biases(self, input_data: np.array, weights: np.array = None, biases: np.array = None, debug:bool=True):
        """
        Initialize the weights and biases of the RBM with the option to specify those weights and biases.
        :param input_data: np.array - list of input examplars
        :param weights: np.array - list of node weights
        :param biases: np.array - list of node biases
        :param debug: bool - flag indicating whether or not to print debug info
        :return: null
        """
        self.n_visible: int = input_data.shape[1]
        if weights is None:
            self.W: np.array = np.random.normal(scale=0.1, size=(self.n_visible, self.n_hidden))
        else:
            self.W: np.array = weights

        if biases is None:
            self.v0: np.array = np.zeros(self.n_visible)
            self.h0: np.array = np.zeros(self.n_hidden)
        else:
            self.v0: np.array = biases
            self.h0: np.array = biases

        if debug:
            print(f"weights: {self.W}")

    def train_rbm(self, input_data: np.array, weights: np.array = None, biases: np.array = None, debug: bool = True):
        """
        Fit RBM using contrastive divergence procedure.
        :param input_data: np.array - list of input examplars.
        :param weights: np.array - list of node weights
        :param biases: np.array - list of node biases
        :param debug: bool - flag indicating whether or not to print debug info
        :return: null
        """
        self._initialize_weights_and_biases(input_data=input_data, weights=weights, biases=biases, debug=debug)
        self.mean_squared_errors: list = []
        self.reconstructions: list = []

        for _ in self.status(range(self.epochs)):
            batch_errors: list = []
            for batch in batch_iterator(input_data, batch_size=self.batch_size):

                # Begin positive phase
                pos_hidden_probs: np.array = self.act_fn(batch.dot(self.W) + self.h0)
                hidden_states: np.array = self._sample(pos_hidden_probs)
                pos_associations: np.array = batch.T.dot(pos_hidden_probs)
                if debug:
                    print(f"positive hidden probs: {pos_hidden_probs}")
                    print(f"positive hidden states: {hidden_states}")
                    print(f"positive associations: {pos_associations}")

                # Begin negative phase
                neg_visible_probs: np.array = self.act_fn(hidden_states.dot(self.W.T) + self.v0)
                neg_visible_states: np.array = self._sample(neg_visible_probs)
                neg_hidden_probs: np.array = self.act_fn(neg_visible_probs.dot(self.W) + self.h0)
                neg_hidden_states: np.array = self._sample(neg_hidden_probs)
                neg_associations: np.array = neg_visible_probs.T.dot(neg_hidden_probs)
                if debug:
                    print(f"negative hidden states: {neg_hidden_states}")
                    print(f"negative hidden probs: {neg_hidden_probs}")
                    print(f"negative visible states: {neg_visible_states}")
                    print(f"negative visible probs: {neg_visible_probs}")
                    print(f"negative associations: {neg_associations}")

                self.W += self.eta * (pos_associations - neg_associations)
                self.h0 += self.eta * (pos_hidden_probs.sum(axis=0) - neg_hidden_probs.sum(axis=0))
                self.v0 += self.eta * (batch.sum(axis=0) - neg_visible_probs.sum(axis=0))

                # Save mean squared error for batch iteration
                mse: float = np.mean((batch - neg_visible_probs) ** 2)
                batch_errors.append(mse)

            self.mean_squared_errors.append(np.mean(batch_errors))
            # Reconstruct a batch of images from the training set
            idx = np.random.choice(range(input_data.shape[0]), self.batch_size)
            # test_data = input_data[idx]
            # print(test_data.shape)
            # self.reconstructions.append(self.reconstruct(test_data[:, :64]))
            self.reconstructions.append(self.reconstruct(input_data[idx]))

    def _sample(self, sample_list: np.array):
        """
        Take a random sample from the specified input
        :param sample_list: np.array - the list to sample from
        :return: np.array - the list of sampled examplars
        """
        return sample_list > np.random.random_sample(size=sample_list.shape)

    def reconstruct(self, inputs: np.array):
        """
        Reconstruct the input from the latent space
        :param inputs: np.array - the list of inputs to reconstruct from
        :return: np.array - the list of reconstructed inputs
        """
        positive_hidden: np.array = self.act_fn(inputs.dot(self.W) + self.h0)
        hidden_states: np.array = self._sample(positive_hidden)
        negative_visible: np.array = self.act_fn(hidden_states.dot(self.W.T) + self.v0)
        return negative_visible

    def get_hidden_units(self, data: np.array):
        """
        Show hidden states for a set of data.
        :param data: np.ndarray - the input data from which to show hidden states of
        :return: np.ndarray - the hidden states of the RBM
        """
        num_examples = data.shape[0]
        hidden_states = np.ones((num_examples, self.n_hidden))
        hidden_activations = np.dot(data, self.W)
        hidden_probs = self.act_fn(hidden_activations)
        hidden_states[:, :] = hidden_probs > np.random.rand(num_examples, self.n_hidden)
        # Set hidden state biases to 1 - should we do this??
        # hidden_states[:,0] = 1
        # Ignore bias
        # hidden_states = hidden_states[:, 1:]
        return hidden_states

    def get_visible_units(self, data: np.ndarray):
        """
        Show visible states for a set of data.
        :param data: np.ndarray - the input data from which to show visible states of
        :return: np.ndarray - the visible states of the RBM
        """
        num_examples = data.shape[0]
        visible_states = np.ones((num_examples, self.n_visible))
        visible_activations = np.dot(data, self.W.T)
        visible_probs = self.act_fn(visible_activations)
        visible_states[:, :] = visible_probs > np.random.rand(num_examples, self.n_visible)
        # Ignore bias
        # visible_states = visible_states[:, 1:]
        return visible_states
