#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from datasets import DataSet
from training_data import TrainingData
import sys
import numpy as np
import json
import os
import pickle
from math import sqrt
import theano
import theano.tensor as T
import crino
from crino.network import PretrainedMLP
from crino.criterion import MeanSquareError
from crino.criterion import CrossEntropy
from math import sqrt

from matplotlib import pyplot as plt


def main():
    # DEFINE PATCH SIZE
    patch_size=20

    needed_params=['learning_params','hidden_geometry','pretraining_geometry','init_weights','save_init_weights','outfolder']
    config = getConfig(patch_size)
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

    #decomment the two lines above if you want to generate you own data and save it

    #x_train, y_train=load_data('dataSauvola/train/',patch_size=patch_size,nb_random_patchs=25) # generate data
    #save_training_data('training_data',x_train,y_train) #save the generated data
    x_train,y_train=load_data_from_file('training_data') #load the saved data

    #test_data(x_train,y_train)

    nApp = x_train.shape[0] # number of training examples
    nFeats = x_train.shape[1] # number of features per input image
    nLabels = y_train.shape[1] # number of labels per groundtruth
    
    # Compute the full geometry of the ioda
    geometry=[nFeats] + hidden_geometry + [nLabels]
    # Compute the number of layers
    nLayers=len(geometry)-1

    print '... building and learning a network'
    ioda = PretrainedMLP(geometry, outputActivation=crino.module.Sigmoid,**pretraining_geometry)
    
    # bake the ioda and set the criterion 
    ioda.linkInputs(T.matrix('x'), nFeats)
    ioda.prepare()
    #ioda.criterion = MeanSquareError(ioda.outputs, T.matrix('y'))
    ioda.criterion= CrossEntropy(ioda.outputs, T.matrix('y'))
    # set initial weights if they exists
    if not(init_weights is None):
        ioda.setParameters(init_weights)
    # save initial weights if ask
    if save_init_weights:
        pickle.dump(ioda.getParameters(),open(os.path.join(absoutfolder,"starting_params.pck"),'w'),protocol=-1)

    delta = ioda.train(x_train, y_train, **learning_params)
    print '... learning lasted %s (s) ' % (delta)
    
    print '... saving results'
    
    # Save parameters in pythonic serialization
    pickle.dump(ioda.getParameters(),open(os.path.join(absoutfolder,"learned_params.pck"),'w'),protocol=-1)
    
    # Save some history of the learning phase in pythonic serialization
    results={
        'I':pretraining_geometry['nInputLayers'],
        'L':nLayers-pretraining_geometry['nInputLayers']-pretraining_geometry['nOutputLayers'],
        'O':pretraining_geometry['nOutputLayers'],
        'train_criterion':ioda.finetune_history[-1],
        'train_history':ioda.finetune_history,
        'train_full_history':ioda.finetune_full_history,
        }
    pickle.dump(results,open(os.path.join(absoutfolder,'results.pck'),'w'),protocol=-1)

def getConfig(patch_size):
    """
        Enable to get the configuration for the Neural Network
    """
    config={}


    #Learning parameters of the input pretraining
    input_pretraining_params={
            'learning_rate': 10,
            'batch_size' : 500,
            'epochs' : 10
    }
    #Learning parameters of the output pretraining
    output_pretraining_params={
            'learning_rate': 10,
            'batch_size' : 100,
            'epochs' : 0
    }

    #Learning parameters of the supervised training + pretrainings
    config['learning_params']={
        'learning_rate' : 0.1,
        'batch_size' : 1,
        'epochs' : 50,
        'input_pretraining_params' : input_pretraining_params,
        'output_pretraining_params' : output_pretraining_params,
        'link_pretraining' : False
    }

    #Size of one hidden representation
    hidden_size = 4*patch_size
    #hidden_size=600
    #Geometry of all hidden representations
    config['hidden_geometry'] = [hidden_size, hidden_size]

    #How many layers are pretrained
    # (here 1 at input and 1 at output)
    config['pretraining_geometry']={
        'nInputLayers': 0,
        'nOutputLayers': 0
    }

    #Shall we used known init weights (here no)
    config['init_weights'] = None #pickle.load(open('./results/starting_params.pck'))
    #Shall we save init weights
    config['save_init_weights'] = False

    #Where to store results
    config['outfolder']='./results/'

    return config

def loadDataConfig(name='dnn_configuration.json',path='results'):
    """
        Loads the neural network configuration from a Json configuration file.

        Args:
            name (str) : the name of the file to load, default 'dnn_configuration.json'
            path (str) : the name of the folder in which we look for the file, default 'results'
        Raises:
            IOError: If the file could not be read/opened.
    """
    print('Loading used configuration')
    return json.load(open(os.path.join(path,name),'rb'))

def load_data(dataDir,patch_size=25,nb_random_patchs=50):
    """
        Load the data for a use

        Args:
            dataDir (str) : the directory we want to use for generating the dataDir
            nb_random_patchs (int) : the number of patchs per image if we use a random generation of data

        Returns:
            numpy.array : Array containing the normalised generated data
    """
    print 'Loading datasets...'
    ds_train_samples = DataSet(dataDir+'samples')
    ds_train_gt = DataSet(dataDir+'groundTruth')
    print 'Loading data generation parameters...'
    data = TrainingData(ds_train_samples, ds_train_gt)
    try:
        data.load_config_file(dataDir+'config.json')
        if (data.config['patch_size']!=patch_size):
            data.config['patch_size']= patch_size
    except IOError:
        data.config['patch_size'] = patch_size
        try:
            data.save_config_file(dataDir+'config.json')
        except IOError:
            print 'Cannot save configuration file.'
    try:
        genconf = data.load_gen_config(dataDir+'generation_params.pck')
    except IOError:
        genconf = data.random_gen_config(nb_random_patchs)
        try:
            data.save_gen_config(dataDir+'generation_params.pck', genconf)
        except IOError:
            print 'Cannot save generation parameters.'
    print 'Generating data...'
    samples_data = data.generate_input_data(genconf)
    gt_data = data.generate_ground_truth_data(genconf)
    print samples_data.shape[0], 'training samples were generated.'
    print 'Each sample contains', samples_data.shape[1], 'pixels.'
    samples_data=normalise(samples_data)
    gt_data=normalise(gt_data)
    return samples_data, gt_data

def load_data_from_file(filename):
    """
    Load the file correspoding to a specified data set

    Args:
        filename (str) : the data file we want to load. Must contain a x_train and a y_train.

    Returns:
        x_train, y_train (numpy array) : the data extracted from the data file
    """

    print 'Loading ', filename
    temp=pickle.load(open('./results/'+filename))
    x_train=temp['x_train']
    y_train=temp['y_train']
    return x_train, y_train

def save_training_data(filename,x_train,y_train):
    """
    Save the specified data into a filename

    Args
        filename (str) : the name file in which we want to save the data
        x_train, y_train (numpy array) : the training data we want to save,
    """
    temp={'x_train': x_train, 'y_train' : y_train}
    pickle.dump(temp,open(os.path.join('./results/',filename),'w'),protocol=-1)

def test_data(x,y):
    """
        Test function allowing to visualize the data and printing the values

        Args
            x, y (numpy.array) : a test matrix

    """
    for i in xrange(0, len(x)):
        n=x.shape[1]
        sqrtn=sqrt(n)
        image = x[i].reshape(sqrtn,sqrtn)
        gt = y[i].reshape(sqrtn,sqrtn)
        print(np.unique(image))
        print(np.unique(gt))

        plt.subplot(1,2,1)
        plt.imshow(image, interpolation='nearest', cmap=plt.get_cmap('gray'), vmin=0.0, vmax=1.0)
        plt.title('Original image')
        plt.subplot(1,2,2)
        plt.imshow(gt, interpolation='nearest', cmap=plt.get_cmap('gray'), vmin=0.0, vmax=1.0)
        plt.title('Groundtruth binarized image')
        plt.show()

def normalise(temp):
	"""
		Allow to normalise a matrix and convert it the float type used by theano (float32)

        Args:
            input (numpy.array) : the matrix to normalise

        Returns:
            numpy.array : the normalized array
	"""
        temp=temp/float(255)
        temp.astype(theano.config.floatX)
        return temp

if __name__=='__main__':
    main()
