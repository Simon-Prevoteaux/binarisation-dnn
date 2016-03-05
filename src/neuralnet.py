import json

NEURALNET_REQUIRED_PARAMS = ['learning_params', 'patches_size', 'hidden_geometry', 'pretraining_geometry']

class NeuralNetwork(object):
    def __init(self):
        super(NeuralNetwork, self).__init__()
        self.initialised = False
        self.learning_params = None
        self.patches_size = None
        self.hidden_geometry = None
        self.pretraining_geometry = None
        self.network = None

    # TODO: Documentation for config format.
    def initialise(self, config_values):
        """
        Initialises the neural network with given configuration values.

        Args:
            config_values (dict): Configuration values for the neural network.

        Raises:
            ValueError: If the configuration values were not valid.
        """
        for k, v in config_values:
            if not k in NEURALNET_REQUIRED_PARAMS:
                raise ValueError('Unknown configuration parameter: ' + str(k))
            setattr(self, k, v)
        for param in NEURALNET_REQUIRED_PARAMS:
            if param == None:
                raise ValueError('Configuration must contain parameter: ' + str(param))
        self.priv_create_network()
        self.initialised = True

    def initialise_from_file(self, config_filepath):
        """
        Initialises the neural network using a json configuration file.

        Args:
            config_filepath (str): Path of the configuration file.

        Raises:
            IOError: If the file could not be opened/read.
            ValueError: If the configuration values were not valid.
        """
        config_values = json.load(open(config_filepath, 'r'))
        self.initialise(config_values)

    def save_to_file(self, config_filepath, weights_filepath=None):
        """
        Saves the neural network to a file.

        Args:
            filepath (str): Path of the file in which the neural network will be saved.

        Raises:
            IOError: If the file could not be opened/read.
        """
        assert self.initialised, "The neural network must be initialised first."
        for k in NEURALNET_REQUIRED_PARAMS:
            config_values[k] = getattr(k)
        json.dump(open(config_filepath, 'w+'))

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
        pickle.dump(self.network.getParameters(open(weights_filepath, 'w+')))

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
            input_patches = array([input_patches])
        output = self.network.forward(input_patches)
        # Return a vector instead of a matrix if a single patch was sent
        if one_patch:
            return output[0]
        return output

    def priv_create_network(self):
        self.network = crino.network.PretrainedMLP([self.patch_size] + self.hidden_geometry + [self.patch_size], outputActivation=crino.module.Sigmoid, **pretraining_geometry)
        self.network.setInputs(T.matrix('x'), self.patch_size)
        self.network.prepare()
        self.network.setCriterion(CrossEntropy(self.network.getOutputs()))

