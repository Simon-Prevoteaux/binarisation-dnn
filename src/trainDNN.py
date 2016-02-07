#!/usr/bin/python2
# -*- coding: utf-8 -*-

from datasets import DataSet
import sys
import numpy as np
import json
import os
import pickle 
from math import sqrt

def main(dataDir):
	"""
	training_set=pickle.load(open(dataDir+'/training_set.pck'))
	x_train=training_set['x_train']
	y_train=training_set['y_train']
	"""

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
	json.dump(used_config,open(os.path.join(absoutfolder,"configuration.json"),'wb'),indent=2)


def getConfig():
    config={}

    dataConfig=loadDataConfig()
    batch_size=dataConfig['bloc_size']
    batch_size=batch_size*batch_size

    #Learning parameters of the input pretraining
    input_pretraining_params={
            'learning_rate': 10.0,
            'batch_size' : batch_size,
            'epochs' : 50
    }
    #Learning parameters of the output pretraining
    output_pretraining_params={
            'learning_rate': 10.0,
            'batch_size' : batch_size,
            'epochs' : 100
    }
    
    #Learning parameters of the supervised training + pretrainings
    config['learning_params']={
        'learning_rate' : 10.0,
        'batch_size' : batch_size,
        'epochs' : 150,
        'input_pretraining_params' : input_pretraining_params,
        'output_pretraining_params' : output_pretraining_params,
        'link_pretraining' : False
    }
    
    #Size of one hidden representation
    hidden_size = sqrt(batch_size)*2
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


if __name__=='__main__':

	if (len(sys.argv)<2):
		dataDir='./results'
	else:
		dataDir=sys.argv[1:][0]

	main(dataDir)
