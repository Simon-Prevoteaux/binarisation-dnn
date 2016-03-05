from datasets import DataSet
from training_data import TrainingData
import json
import pickle
import numpy
import theano.tensor as T
import crino

NEURALNET_REQUIRED_PARAMS = ['learning_params', 'hidden_geometry', 'pretraining_geometry']

class NeuralNetwork(object):
    def __init(self):
        super(NeuralNetwork, self).__init__()
        self.initialised = False
        self.learning_params = None
        self.features_count = None
        self.hidden_geometry = None
        self.pretraining_geometry = None
        self.network = None

    # TODO: Documentation for config format.
    def initialise(self, features_count, config_values):
        """
        Initialises the neural network with given configuration values.

        Args:
            features_count (int): Size of the input vectors.
            config_values (dict): Configuration values for the neural network.

        Raises:
            ValueError: If the configuration values were not valid.
        """
        self.features_count = features_count
        for k, v in config_values.items():
            if not k in NEURALNET_REQUIRED_PARAMS:
                raise ValueError('Unknown configuration parameter: ' + str(k))
            setattr(self, k, v)
        for param in NEURALNET_REQUIRED_PARAMS:
            if param == None:
                raise ValueError('Configuration must contain parameter: ' + str(param))
        self.priv_create_network()
        self.initialised = True

    def initialise_from_file(self, features_count, config_filepath):
        """
        Initialises the neural network using a json configuration file.

        Args:
            features_count (int): Size of the input vectors.
            config_filepath (str): Path of the configuration file.

        Raises:
            IOError: If the file could not be opened/read.
            ValueError: If the configuration values were not valid.
        """
        config_values = json.load(open(config_filepath, 'r'))
        self.initialise(features_count, config_values)

    def save_config_to_file(self, config_filepath, weights_filepath=None):
        """
        Saves the neural network configuration to a file.

        Args:
            filepath (str): Path of the file in which the neural network will be saved.

        Raises:
            IOError: If the file could not be opened/read.
        """
        assert self.initialised, "The neural network must be initialised first."
        config_values = dict()
        for k in NEURALNET_REQUIRED_PARAMS:
            config_values[k] = getattr(self, k)
        json.dump(config_values, open(config_filepath, 'w+'))

    def load_weights(self, weights_filepath):
        """
        Loads all the weights of the neural network from a file.

        Args:
            weights_filepath (str): Path of the file from which the weights will be loaded.

        Raises:
            IOError: If the file could not be opened/read.
            ValueError: If the loaded weights were not compatible with the network.
        """
        assert self.initialised, "The neural network must be initialised first."
        self.network.setParameters(pickle.load(open(weights_filepath, 'r')))

    def save_weights(self, weights_filepath):
        """
        Saves all the weights of the neural network to a file.

        Args:
            weights_filepath (str): Path of the file in which the weights will be saved.

        Raises:
            IOError: If the file could not be opened/written.
        """
        assert self.initialised, "The neural network must be initialised first."
        pickle.dump(self.network.getParameters(), open(weights_filepath, 'w+'))

    def train(self, input_patches, output_patches):
        """
        Trains the neural network on the given data.

        Warning: The neural network must have been initialised first.

        Args:
            input_patches (numpy.ndarray): Input data that will be used to train the network.
            output_patches (numpy.ndarray): Output data that will be used to train the network.
        """
        assert self.initialised, "The neural network must be initialised first."
        delta_time = self.network.train(input_patches, output_patches, **self.learning_params)
        print "Training complete (" + str(delta_time) + " seconds)."

    def apply(self, input_patches):
        """
        Computes the output of the neural network for a given set of patches.

        Warning: The neural network must have been initialised first.

        Args:
            input_patches (numpy.ndarray): Input data. Can be a vector (one patch) or a matrix (a set of patches).

        Returns:
            numpy.ndarray: Output patches computed by the neural network.
        """
        assert self.initialised, "The neural network must be initialised first."
        # Allow forwarding of a single patch
        one_patch = len(input_patches.shape) == 1
        if one_patch:
            input_patches = numpy.array([input_patches])
        output = self.network.forward(input_patches)
        # Return a vector instead of a matrix if a single patch was sent
        if one_patch:
            return output[0]
        return output

    def priv_create_network(self):
        self.network = crino.network.PretrainedMLP([self.features_count] + self.hidden_geometry + [self.features_count], outputActivation=crino.module.Sigmoid, **self.pretraining_geometry)
        self.network.setInputs(T.matrix('x'), self.features_count)
        self.network.prepare()
        self.network.setCriterion(crino.criterion.CrossEntropy(self.network.getOutputs(), T.matrix('nn_output')))

