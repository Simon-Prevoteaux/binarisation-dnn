#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from training_data import TrainingData
from datasets import DataSet

def main():
    print 'Loading datasets...'
    ds_train_samples = DataSet('data/train/samples')
    ds_train_gt = DataSet('data/train/groundTruth')
    print 'Loading data generation parameters...'
    data = TrainingData(ds_train_samples, ds_train_gt)
    try:
        data.load_config_file('config.json')
    except IOError:
        data.config['patch_size'] = 64
        try:
            data.save_config_file('config.json')
        except IOError:
            print 'Cannot save configuration file.'
    try:
        genconf = data.load_gen_config('generation_params.pck')
    except IOError:
        genconf = data.exhaustive_gen_config((16, 16))
        try:
            data.save_gen_config('generation_params.pck', genconf)
        except IOError:
            print 'Cannot save generation parameters.'
    print 'Generating data...'
    samples_data = data.generate_input_data(genconf)
    gt_data = data.generate_ground_truth_data(genconf)
    print samples_data.shape[0], 'training samples were generated.'

if __name__ == '__main__':
    main()
    
