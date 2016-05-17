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
            'random': lambda dataset, conf, psize: dataset.random_gen_config(conf['patches_per_image'],psize),
            'exhaustive': lambda dataset, conf: dataset.exhaustive_gen_config(conf['patches_padding']),
            'load': lambda dataset, conf: dataset.load_gen_config(['conf.generation_file'])
        }

    def get_training_dataset(self):
        if self.training == None:
            print 'Preloading training dataset...'
            self.training = TrainingData(DataSet(self.config['training']['samples']), DataSet(self.config['training']['ground_truth']))
            self.training.set_config({'input_patch_size': self.config['input_patch_size'],'output_patch_size': self.config['output_patch_size']})
        return self.training

    def get_input_training_genconfig(self):
        if self.training_genconfig == None:
            print 'Generating input training dataset configuration...'
            self.training_genconfig = self.load_gen_config.get(self.config['training']['generation_type'])(self.get_training_dataset(), self.config['training'],self.config['input_patch_size'])
        return self.training_genconfig

    def get_output_training_genconfig(self):
        if self.training_genconfig == None:
            print 'Generating output training dataset configuration...'
            self.training_genconfig = self.load_gen_config.get(self.config['training']['generation_type'])(self.get_training_dataset(), self.config['training'],self.config['output_patch_size'])
        return self.training_genconfig

    def get_validation_dataset(self):
        if self.validation == None:
            print 'Generating validation dataset configuration...'
            self.validation = TrainingData(DataSet(self.config['validation']['samples']), DataSet(self.config['validation']['ground_truth']))
            self.validation.set_config({'input_patch_size': self.config['input_patch_size'],'output_patch_size': self.config['output_patch_size']})
        return self.validation

    def get_input_validation_genconfig(self):
        if self.validation_genconfig == None:
            print 'Generating input validation dataset configuration...'
            self.validation_genconfig = self.load_gen_config.get(self.config['validation']['generation_type'])(self.get_validation_dataset(), self.config['validation'],self.config['input_patch_size'])
        return self.validation_genconfig

    def get_output_validation_genconfig(self):
        if self.validation_genconfig == None:
            print 'Generating output validation dataset configuration...'
            self.validation_genconfig = self.load_gen_config.get(self.config['validation']['generation_type'])(self.get_validation_dataset(), self.config['validation'],self.config['output_patch_size'])
        return self.validation_genconfig

    def get_models_count(self):
        return len(self.models)

    def get_model(self, i):
        if self.models[i] == None:
            model_config = self.config['models'][i]
            print 'Configuring model ' + model_config['name'] + '...'
            self.models[i] = NeuralNetwork()
            in_patch_size = self.config['input_patch_size']
            out_patch_size = self.config['output_patch_size']

            self.models[i].initialise(self.config['input_patch_size'] ** 2, self.config['output_patch_size'] ** 2, range(0, model_config['network']['learning_params']['epochs'], model_config['logging_period']), model_config['network'])
        return self.models[i]

    def get_model_name(self, i):
        return self.config['models'][i]['name']
