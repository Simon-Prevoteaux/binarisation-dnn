from binarisation.datasets import DataSet
from binarisation.training_data import TrainingData
import json
import pickle
import numpy
import theano.tensor as T
import crino

NEURALNET_REQUIRED_PARAMS = ['learning_params', 'hidden_geometry', 'pretraining_geometry']

class LoggingNeuralNet(crino.network.PretrainedMLP):
    def __init__(self, controlling_net, logged_epochs, nUnits, outputActivation=crino.module.Sigmoid, nInputLayers=0, nOutputLayers=0, InputAutoEncoderClass=crino.network.AutoEncoder, OutputAutoEncoderClass=crino.network.OutputAutoEncoder):
        super(LoggingNeuralNet, self).__init__(nUnits, outputActivation=outputActivation, nInputLayers=nInputLayers, nOutputLayers=nOutputLayers, InputAutoEncoderClass=InputAutoEncoderClass, OutputAutoEncoderClass=OutputAutoEncoderClass)
        self.controlling_net = controlling_net
        self.logged_epochs = logged_epochs
        self.output_file = None

    def setLoggingFile(self, log_file):
        self.output_file = log_file

    def checkEpochHook(self, finetune_vars):
        # Dump the network waits if necessary
        epoch_id = finetune_vars['epoch']
        if self.output_file != None and epoch_id in self.logged_epochs:
            print 'Logging epoch ' + str(epoch_id) + '...'
            self.controlling_net.save_weights(self.output_file, epoch_id)
        return super(LoggingNeuralNet, self).checkEpochHook(finetune_vars)

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
    def initialise(self, input_features_count,output_features_count, logged_epochs, config_values):
        """
        Initialises the neural network with given configuration values.

        Args:
            intput_features_count(int) : Size of the input vectors.
            output_features_count (int): Size of the output vectors.
            config_values (dict): Configuration values for the neural network.

        Raises:
            ValueError: If the configuration values were not valid.
        """
        self.input_features_count = input_features_count
        self.output_features_count = output_features_count
        for k, v in config_values.items():
            if not k in NEURALNET_REQUIRED_PARAMS:
                raise ValueError('Unknown configuration parameter: ' + str(k))
            setattr(self, k, v)
        for param in NEURALNET_REQUIRED_PARAMS:
            if param == None:
                raise ValueError('Configuration must contain parameter: ' + str(param))
        self.priv_create_network(logged_epochs)
        self.initialised = True

    def initialise_from_file(self, input_features_count,output_features_count , logged_epochs, config_filepath):
        """
        Initialises the neural network using a json configuration file.

        Args:
            intput_features_count(int) : Size of the input vectors.
            output_features_count (int): Size of the output vectors.
            config_filepath (str): Path of the configuration file.

        Raises:
            IOError: If the file could not be opened/read.
            ValueError: If the configuration values were not valid.
        """
        config_values = json.load(open(config_filepath, 'r'))
        self.initialise(input_size,output_size, logged_epochs, config_values)

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

    def load_weights(self, weights_file):
        """
        Loads all the weights of the neural network from a file.

        Args:
            weights_file (file): File from which the weights will be loaded.

        Returns:
            int: Id of the epoch at which the weights were recorded. (-1 if unknown epoch)

        Raises:
            IOError: If the file could not be opened/read.
            ValueError: If the loaded weights were not compatible with the network.
        """
        assert self.initialised, "The neural network must be initialised first."
        epoch_id = pickle.load(weights_file)
        self.network.setParameters(pickle.load(weights_file))
        return epoch_id

    def save_weights(self, weights_file, epoch_id=-1):
        """
        Saves all the weights of the neural network to a file.

        Args:
            weights_file (file): File in which the weights will be saved.
            epoch_id (int): Id of the epoch at which the weights were recorded. (-1 if unknown epoch)

        Raises:
            IOError: If the file weights could not be written to the file.
        """
        assert self.initialised, "The neural network must be initialised first."
        pickle.dump(epoch_id, weights_file)
        pickle.dump(self.network.getParameters(), weights_file)

    def train(self, input_patches, output_patches, log_file=None):
        """
        Trains the neural network on the given data.

        Warning: The neural network must have been initialised first.

        Args:
            input_patches (numpy.ndarray): Input data that will be used to train the network.
            output_patches (numpy.ndarray): Output data that will be used to train the network.
            log_file (file): File in which the weights will be logged during training.
        """
        assert self.initialised, "The neural network must be initialised first."
        self.network.setLoggingFile(log_file)
        delta_time = self.network.train(input_patches, output_patches, **self.learning_params)
        # Reord the last set of weights if it has not been
        if log_file != None:
            epoch_id = self.learning_params['epochs'] - 1
            if epoch_id not in self.network.logged_epochs:
                print 'Logging epoch ' + str(epoch_id) + '...'
                self.save_weights(log_file, epoch_id)
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

    def priv_create_network(self, logged_epochs):
        self.network = LoggingNeuralNet(self, logged_epochs, [self.input_features_count] + self.hidden_geometry + [self.output_features_count], outputActivation=crino.module.Sigmoid, **self.pretraining_geometry)
        self.network.setInputs(T.matrix('x'), self.input_features_count)
        self.network.prepare()
        self.network.setCriterion(crino.criterion.CrossEntropy(self.network.getOutputs(), T.matrix('nn_output')))
