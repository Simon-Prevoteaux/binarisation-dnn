
import os;
import theano;
import theano.tensor as T;
import cPickle as pickle;
import numpy as np;
import scipy.io as sio
from matplotlib import pyplot as plt
import json;
import pdb;
import crino;
from crino.network import MultiLayerPerceptron;
from train_dnn import load_data
from train_dnn import normalise

outfolder = "./results/"

print('... loading used configuration')
config = json.load(open(os.path.join(outfolder,"dnn_configuration.json"),'r'))

#loading test data
x_test,y_test=load_data('dataSauvola/test/',nb_random_patchs=1)

x_test = np.asarray(x_test, dtype=theano.config.floatX) # We convert to float32 to 
y_test = np.asarray(y_test, dtype=theano.config.floatX) # compute on GPUs with CUDA

x_test=normalise(x_test)
y_test=normalise(y_test)

nTest = x_test.shape[0] # number of test examples
nFeats = x_test.shape[1] # number of features per input image
nLabels = y_test.shape[1] # number of labels per groundtruth

nn = MultiLayerPerceptron([nFeats] + config['hidden_geometry'] + [nLabels], outputActivation=crino.module.Sigmoid)
nn.linkInputs(T.matrix('x'), nFeats)
nn.prepare()
nn.setParameters(pickle.load(open('./results/learned_params.pck')))


for i in xrange(0, nTest):
    image = x_test[i].reshape(100,100)
    estimated_binarisation = nn.forward(x_test[i:i+1])
    print(np.unique(estimated_binarisation))
    estimated_image = estimated_binarisation.reshape(100,100)
    gt = y_test[i].reshape(100,100)
    
    plt.subplot(2,2,1)
    plt.imshow(image, interpolation='nearest', cmap=plt.get_cmap('gray'), vmin=0.0, vmax=1.0)
    plt.title('Original image')
    plt.subplot(2,2,2)
    plt.imshow(estimated_image, interpolation='nearest', cmap=plt.get_cmap('gray'), vmin=0.0, vmax=1.0)
    plt.title('Estimated binarized image')
    plt.subplot(2,2,4)
    plt.imshow(gt, interpolation='nearest', cmap=plt.get_cmap('gray'), vmin=0.0, vmax=1.0)
    plt.title('Groundtruth binarized image')
    plt.show()

