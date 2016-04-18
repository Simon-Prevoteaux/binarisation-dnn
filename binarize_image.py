#!/usr/bin/env python2

from binarisation.project import Project
from binarisation.datasets import DataSet
from binarisation.training_data import TrainingData
import sys
import os
import numpy as np
import theano
import matplotlib.pyplot as plt

def get_model_from_name(project, model_name):
    for i in range(project.get_models_count()):
        if project.get_model_name(i) == model_name:
            return project.get_model(i)
    raise LookupError('Could not find model with name ' + model_name)

def load_weights(model, model_name, model_filename, best_epoch_number):
    # Unfortunately, the model system is designed in a way wich requires us to iterate through all the epochs before finding the right one
    # TODO: Fix this issue
    with open(model_filename, 'rb') as model_file:
        while True:
            try:
                k = model.load_weights(model_file)
            except EOFError:
                break
            if k == best_epoch_number:
                return model.load_weights(model_file)
    raise IOError('Could not find epoch number ' + str(best_epoch_number))

def binarize(patches, patch_size, gen_config, model, image_size):
    print 'Binarizing image...'
    binpatches = model.apply(patches)
    print 'Reconstructing image...'
    data = np.empty((image_size[1], image_size[0]))
    for i in range(len(gen_config)):
        x = gen_config[i][0]
        y = gen_config[i][1]
        data[y : y + patch_size, x : x + patch_size] = binpatches[i].reshape(patch_size, patch_size)
    return data

def main():
    # Parse command line args
    if len(sys.argv) < 2:
        print 'USAGE: binarize_image.py project_file model_name best_epoch_number test_image_file [models_dir]'
        sys.exit(1)
    project_file = sys.argv[1]
    model_name = sys.argv[2]
    best_epoch_number = int(sys.argv[3])
    test_image_file = sys.argv[4]
    if len(sys.argv) >= 6:
        models_dir = sys.argv[5]
    else:
        models_dir = '.'

    # Load the project file and the model
    project = Project(project_file)
    model_filename = os.path.join(models_dir, model_name + '.model')
    model = get_model_from_name(project, model_name)
    # Load the weight corresponding to the right epoch
    load_weights(model, model_name, model_filename, best_epoch_number)
    # Load the image
    dataset = DataSet('.')
    dataset.imagespaths[test_image_file] = test_image_file
    data = TrainingData(dataset, dataset)
    patch_size = project.get_training_dataset().config['patch_size']
    data.set_config({'patch_size': patch_size})
    genconfig = data.exhaustive_gen_config([patch_size, patch_size])
    patches = data.generate_input_data(genconfig)
    # Binarize the image
    binarized_image = binarize(patches, patch_size, genconfig[test_image_file], model, dataset.open_image(test_image_file).size)
    # Draw the image
    print(binarized_image.shape)
    plt.imshow(binarized_image, cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()

