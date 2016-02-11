#!/usr/bin/python2
# -*- coding: utf-8 -*-

from datasets import DataSet
from training_data import TrainingData
import sys
import numpy as np
import json
import os
import pickle 
from math import sqrt

def main():
	needed_params=['learning_params','hidden_geometry','pretraining_geometry','init_weights','save_init_weights','outfolder']
	config = getConfig()
	used_config={}
	for aParam in needed_params:
		if not( aParam in config.keys()):
			raise ValueError("Experience configuration does not contain %s parameter"%(aParam))
		if aParam!='init_weights':
			used_config[aParam]=config[aParam]
		elif not (config[aParam] is None):
			used_config['init_weights'] = []

	learning_params=config['learning_params']
	hidden_geometry=config['hidden_geometry']
	pretraining_geometry=config['pretraining_geometry']
	init_weights=config['init_weights']
	save_init_weights=config['save_init_weights']
	outfolder=config['outfolder']

	absoutfolder=os.path.abspath(outfolder)
	if not os.path.exists(absoutfolder):
		os.mkdir(absoutfolder)

	print('... saving used configuration')
	json.dump(used_config,open(os.path.join(absoutfolder,"dnn_configuration.json"),'wb'),indent=2)
    
        x_train, y_train=load_data('dataSauvola')

def getConfig():
    config={}


    #Learning parameters of the input pretraining
    input_pretraining_params={
            'learning_rate': 10.0,
            'batch_size' : 10000,
            'epochs' : 50
    }
    #Learning parameters of the output pretraining
    output_pretraining_params={
            'learning_rate': 10.0,
            'batch_size' : 10000,
            'epochs' : 10
    }
    
    #Learning parameters of the supervised training + pretrainings
    config['learning_params']={
        'learning_rate' : 10.0,
        'batch_size' : 10000,
        'epochs' : 150,
        'input_pretraining_params' : input_pretraining_params,
        'output_pretraining_params' : output_pretraining_params,
        'link_pretraining' : False
    }
    
    #Size of one hidden representation
    hidden_size = 2*64
    #Geometry of all hidden representations 
    config['hidden_geometry'] = [hidden_size, hidden_size]

    #How many layers are pretrained
    # (here 1 at input and 1 at output) 
    config['pretraining_geometry']={
        'nInputLayers': 1,
        'nOutputLayers': 1
    }

    #Shall we used known init weights (here no)
    config['init_weights'] = None #pickle.load(open('./results/starting_params.pck'))
    #Shall we save init weights
    config['save_init_weights'] = False

    #Where to store results
    config['outfolder']='./results/'

    return config

def loadDataConfig(name='configuration.json',path='data'):
	print('Loading used configuration')
	return json.load(open(os.path.join(path,name),'rb'))

def load_data(dataDir):
    print 'Loading datasets...'
    ds_train_samples = DataSet(dataDir+'/train/samples')
    ds_train_gt = DataSet(dataDir+'/train/groundTruth')
    print 'Loading data generation parameters...'
    data = TrainingData(ds_train_samples, ds_train_gt)
    try:
        data.load_config_file('config.json')
    except IOError:
        data.config['patch_size'] = 100
        try:
            data.save_config_file('config.json')
        except IOError:
            print 'Cannot save configuration file.'
    try:
        genconf = data.load_gen_config('generation_params.pck')
    except IOError:
        genconf = data.random_gen_config(1000)
        try:
            data.save_gen_config('generation_params.pck', genconf)
        except IOError:
            print 'Cannot save generation parameters.'
    print 'Generating data...'
    samples_data = data.generate_input_data(genconf)
    gt_data = data.generate_ground_truth_data(genconf)
    print samples_data.shape[0], 'training samples were generated.'
    return samples_data, gt_data

if __name__=='__main__':
	main()
