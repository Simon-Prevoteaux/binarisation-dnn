#!/usr/bin/env python2

from binarisation.project import Project
from binarisation.datasets import DataSet
from binarisation.training_data import TrainingData
import sys
import os
import numpy as np
from numpy import linspace
import matplotlib.pyplot as plt

def jaccard_distance(estimate, ground_truth):
    return 1.0 - np.logical_and(estimate, ground_truth).sum(dtype=np.float_) / np.logical_or(estimate, ground_truth).sum(dtype=np.float_)

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

def threshold(threshold_value,estimate):
    out = np.zeros(estimate.shape)
    out[estimate > threshold_value] = 1
    return out

def plot_validation_results(thresholds, jaccard_distances):
    plt.figure()
    plt.title('threshold validation')
    plt.xlabel('Treshold')
    plt.ylabel('Jaccard distance')
    plt.plot(thresholds, jaccard_distances)
    plt.show()

def validate_threshold(model,input_data, output_data):
    jaccard_distances = []
    # Compute the estimate
    estimate = model.apply(input_data)
    threshold_range=linspace(0.1,0.9,9)
    for threshold_value in threshold_range:
        # Compute Jaccard distance between the estimate and the ground truth
        binarised_estimate = threshold(threshold_value,estimate)
        jaccard_distances.append(jaccard_distance(binarised_estimate, output_data))


    plot_validation_results(threshold_range,jaccard_distances)
    #Find the best threshold according to the jaccard criterion
    jaccard_distances=np.array(jaccard_distances)
    best_threshold=threshold_range[jaccard_distances.argmin()]
    print 'The best threshold is ' + str(best_threshold)


def main():
    # Parse command line args
    if len(sys.argv) < 2:
        print 'USAGE: validate-treshold.py project_file model_name best_epoch_number [models_dir]'
        sys.exit(1)
    project_file = sys.argv[1]
    model_name = sys.argv[2]
    best_epoch_number = int(sys.argv[3])
    if len(sys.argv) >= 5:
        models_dir = sys.argv[4]
    else:
        models_dir = '.'

    # Load the project file and the model
    project = Project(project_file)
    model_filename = os.path.join(models_dir, model_name + '.model')
    model = get_model_from_name(project, model_name)
    # Load the weight corresponding to the right epoch
    load_weights(model, model_name, model_filename, best_epoch_number)

    # Generate validation data
    validation_dataset = project.get_validation_dataset()
    validation_genconfig = project.get_validation_genconfig() #using the same validation dataset for threshold fine tuning
    input_data = validation_dataset.generate_input_data(validation_genconfig)
    output_data = validation_dataset.generate_ground_truth_data(validation_genconfig)


    validate_threshold(model,input_data,output_data)




if __name__ == '__main__':
    main()
