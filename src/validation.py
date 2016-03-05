
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
from train_nn import load_data
from train_nn import normalise
from math import sqrt

from os import walk
from os.path import relpath
from os.path import splitext




def main():
    # DEFINE PATCH SIZE
    patch_size=20

    outfolder = "./results/"
    absoutfolder=os.path.abspath(outfolder)

    weight_folder="./results/dumps"

    print('... loading used configuration')
    config = json.load(open(os.path.join(outfolder,"dnn_configuration.json"),'r'))

    #loading test data

    x_test,y_test=load_data('dataSauvola/valid/',patch_size=patch_size,nb_random_patchs=10)

    x_valid = np.asarray(x_test, dtype=theano.config.floatX) # We convert to float32 to 
    y_valid = np.asarray(y_test, dtype=theano.config.floatX) # compute on GPUs with CUDA


    nTest = x_test.shape[0] # number of test examples
    nFeats = x_test.shape[1] # number of features per input image
    nLabels = y_test.shape[1] # number of labels per groundtruth

    nn = MultiLayerPerceptron([nFeats] + config['hidden_geometry'] + [nLabels], outputActivation=crino.module.Sigmoid)

    nn.linkInputs(T.matrix('x'), nFeats)
    nn.prepare()


    k=0
    for (path, _, files) in walk(weight_folder):
        validation_error=np.zeros((len(files)))
        files.sort()
        print(files)
        for param in files:
            print('Weight file :',param)
            nn.setParameters(pickle.load(open(weight_folder+'/'+param)))
            for i in xrange(0, nTest):
                estimated_binarisation = nn.forward(x_test[i:i+1])
                y_est=treshold(estimated_binarisation)
                validation_error[k]+=jaccard_index(y_est[0],y_test[i])
            k+=1

        pickle.dump(validation_error,open(os.path.join(absoutfolder,'jaccard_index.pck'),'w'),protocol=-1)

        print(validation_error)
        plt.subplot(1,1,1)
        plt.plot(validation_error)
        plt.title('jaccard index')
        plt.show()

def jaccard_index(y_est,y_true):
    return sum(np.logical_and(y_est,y_true))/float(sum(np.logical_or(y_est,y_true)))

def treshold(nn_output):
    treshold=0.5
    nn_output[nn_output<treshold]=0
    nn_output[nn_output>=treshold]=1
    return nn_output

if __name__=='__main__':
    main()
