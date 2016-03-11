from binarisation.datasets import DataSet;
from binarisation.training_data import TrainingData;
from binarisation.neuralnet import NeuralNetwork;
import json

class Project(object):
    def __init__(self, config_filepath):
        self.config = json.load(open(config_filepath, 'r'))
        self.training = None
        self.training_genconfig = None
        self.validation = None
        self.validation_genconfig = None
        self.models = [None] * len(self.config['models'])
        self.load_gen_config = {
            'random': lambda dataset, conf: dataset.random_gen_config(conf['patches_per_image']),
            'exhaustive': lambda dataset, conf: dataset.exhaustive_gen_config(conf['patches_padding']),
            'load': lambda dataset, conf: dataset.load_gen_config(['conf.generation_file'])
        }

    def get_training_dataset(self):
        if self.training == None:
            print 'Preloading training dataset...'
            self.training = TrainingData(DataSet(self.config['training']['samples']), DataSet(self.config['training']['ground_truth']))
            self.training.set_config({'patch_size': self.config['patch_size']})
        return self.training

    def get_training_genconfig(self):
        if self.training_genconfig == None:
            print 'Generating training dataset configuration...'
            self.training_genconfig = self.load_gen_config.get(self.config['training']['generation_type'])(self.get_training_dataset(), self.config['training'])
        return self.training_genconfig

    def get_validation_dataset(self):
        if self.validation == None:
            print 'Generating validation dataset configuration...'
            dataset = TrainingData(DataSet(self.config['validation']['samples']), DataSet(self.config['validation'].ground_truth))
            dataset.set_config({'patch_size': self.config['patch_size']})
            self.validation = self.load_gen_config.get(self.config['validation']['generation_type'])(dataset, self.config['validation'])
        return self.validation

    def get_validation_genconfig(self):
        if self.validation_genconfig == None:
            print 'Generating validation dataset configuration...'
            self.validation_genconfig = self.load_gen_config.get(self.config['validation']['generation_type'])(self.get_validation_dataset(), self.config['validation'])
        return self.validation_genconfig

    def get_models_count(self):
        return len(self.models)

    def get_model(self, i):
        if self.models[i] == None:
            print 'Configuring model ' + self.config['models'][i]['name'] + '...'
            self.models[i] = NeuralNetwork()
            patch_size = self.config['patch_size']
            print 'Featules count: ' + str(patch_size ** 2)
            self.models[i].initialise(self.config['patch_size'] ** 2, self.config['models'][i]['network'])
        return self.models[i]

    def get_model_name(self, i):
        return self.config['models'][i]['name']

