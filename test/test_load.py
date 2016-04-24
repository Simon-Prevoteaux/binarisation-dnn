#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from training_data import TrainingData
from datasets import DataSet

# Loading constants
TRAINING_SAMPLES_PATH = 'data/train/samples'
GROUND_TRUTH_PATH = 'data/train/groundTruth'
DATA_FORMAT_CONFIG_PATH = 'config.json'
DATA_GENERATION_CONFIG_PATH = 'generation_params.pck'
DEFAULT_PATCH_SIZE = 64
DEFAULT_PATCHES_PER_IMG = 4

def main():
    # Analyze the dataset paths
    print 'Loading datasets...'
    ds_train_samples = DataSet(TRAINING_SAMPLES_PATH)
    ds_train_gt = DataSet(GROUND_TRUTH_PATH)
    print 'Loading data generation parameters...'
    data = TrainingData(ds_train_samples, ds_train_gt)
    # Load the data format parameters or use the default ones
    try:
        data.load_config_file(DATA_FORMAT_CONFIG_PATH)
    except IOError:
        data.config['patch_size'] = DEFAULT_PATCH_SIZE
        try:
            data.save_config_file(DATA_FORMAT_CONFIG_PATH)
        except IOError:
            print 'Cannot load data format parameters.'
    # Load the data generation parameters or use the default ones
    try:
        genconf = data.load_gen_config(DATA_GENERATION_CONFIG_PATH)
    except IOError:
        genconf = data.random_gen_config(DEFAULT_PATCHES_PER_IMG)
        try:
            data.save_gen_config(DATA_GENERATION_CONFIG_PATH, genconf)
        except IOError:
            print 'Cannot save generation parameters.'
    # Generate the data itself
    print 'Generating data...'
    samples_data = data.generate_input_data(genconf)
    gt_data = data.generate_ground_truth_data(genconf)
    print samples_data.shape[0], 'training samples were generated.'

if __name__ == '__main__':
    main()

