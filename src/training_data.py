from datasets import DataSet
import json
import pickle
import numpy as np

class TrainingData(object):
    """
    Enables data generation for training.

    Attributes:
        config (dict): Configuration of the training data. Must contain 'patch_size'.
        samples (DataSet): Input data.
        ground_truth (Optional[DataSet]): Ground truth corresponding to the input data.
    """

    def __init__(self, samples, ground_truth=None):
        """
        Args:
            samples (DataSet): Input data. (Will be preloaded if not loaded already.)
            ground_truth (Optional[DataSet]): Ground truth corresponding to the input data. Can be None if the ground truth is not known. (Will be preloaded if not loaded already.)
        """
        if len(samples.images_names()) == 0:
            samples.preload()
        if len(ground_truth.images_names()) == 0:
            ground_truth.preload()
        self.samples = samples
        self.ground_truth = ground_truth
        self.config = {'patch_size': 64}

    def load_config_file(self, config_file_name):
        """
        Loads the training data set configuration from a Json configuration file.

        Args:
            config_file_name (str): Name of the configuration file.

        Raises:
            IOError: If the file could not be read/opened.
            ValueError: If the file was not a valid configuration file.
        """
        temp = json.load(open(config_file_name, 'r'))
        if not ('patch_size' in temp):
            raise ValueError('Invalid configuration in file ' + config_file_name)
        self.config = temp

    def save_config_file(self, config_file_name):
        """
        Saves the training data set configuration to a Json configuration file.

        Args:
            config_file_name (str): Name of the configuration file.

        Raises:
            IOError: If the file could not be opened for writing.
        """
        json.dump(self.config, open(config_file_name, 'w+'))

    def generate_input_data(self, gen_config):
        """
        Creates an array containing the input patches.
        
        Args:
            gen_config: Generation configuration. It is recommended to generate it through one of the methods of this class.

        Raises:
            IOError: If an image could not be read.
            ValueError: If the generation configuration was not valid.

        Returns:
            numpy.array: Array containing the input data.
        """
        return priv_gen_data(self.config, gen_config, self.samples)

    def generate_ground_truth_data(self, gen_config):
        """
        Creates an array containing the ground truth patches.
        
        Args:
            gen_config: Generation configuration. It is recommended to generate it through one of the methods of this class.

        Raises:
            IOError: If an image could not be read.
            ValueError: If the generation configuration was not valid.

        Returns:
            numpy.array: Array containing the input data.
        """
        return priv_gen_data(self.config, gen_config, self.ground_truth)

    def random_gen_config(self, patches_per_image):
        """
        Creates a generation configuration containing random patches of the images.

        Args:
            patches_per_image (int): Amount of patches to generate per image.

        Returns:
            Generation configuration.
        """
        if self.ground_truth == None:
            files = self.samples.images_names()
        else:
            files = self.samples.common_images(self.ground_truth)
        gen_config = {}
        psize = self.config['patch_size']
        for imgname in files:
            (width, height) = self.samples.open_image(imgname).size
            gen_config[imgname] = np.concatenate((
                np.random.randint(0, width - psize, (patches_per_image, 1)),
                np.random.randint(0, height - psize, (patches_per_image, 1))
            ), axis=1)
        return gen_config

    def exhaustive_gen_config(self, patches_padding):
        """
        Creates a generation configuration containing the whole images.

        Args:
            patches_padding (numpy.array): (x, y) padding between each patch of the image.

        Returns:
            Generation configuration.
        """
        if self.ground_truth == None:
            files = self.samples.images_names()
        else:
            files = self.samples.common_images(self.ground_truth)
        gen_config = {}
        psize = self.config['patch_size']
        for imgname in files:
            (width, height) = self.samples.open_image(imgname).size
            curlist = []
            for y in range(0, height - psize, patches_padding[1]):
                for x in range(0, width - psize, patches_padding[0]):
                    curlist.append([x, y])
            gen_config[imgname] = np.array(curlist, np.int16)
        return gen_config

    def load_gen_config(self, filename):
        """
        Loads a generation configuration from a file.

        Args:
            filename (str): Path of the file.

        Raises:
            IOError: If the file could not be read/opened.
            ValueError: If the file was not a valid generation configuration file.

        Returns:
            Generation configuration.
        """
        return pickle.load(open(filename, 'r'))

    def save_gen_config(self, filename, gen_config):
        """
        Saves a generation configuration to a file.

        Args:
            filename (str): Path of the file.
            gen_config: Generation configuration.

        Raises:
            IOError: If the file could not be opened for writing.
        """
        pickle.dump(gen_config, open(filename, 'w+'))

def priv_gen_data(config, gen_config, dataset):
    pcount = sum([coords.shape[0] for coords in gen_config.itervalues()])
    psize = config['patch_size']
    data = np.empty([pcount, psize * psize], np.uint8)
    i = 0
    for (imgname, coords) in gen_config.iteritems():
        img = dataset.open_image(imgname).convert('L')
        (w, h) = img.size
        # Load image as 2D array
        print (w, h)
        for (x, y) in coords:
            # Get the patch as 1D array
            data[i, :] = img.crop((x, y, x + psize, y + psize)).getdata()
            i += 1
    return data

