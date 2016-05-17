#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from binarisation.project import Project
import os.path
import sys
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

def jaccard_distance(estimate, ground_truth):
    return 1.0 - np.logical_and(estimate, ground_truth).sum(dtype=np.float_) / np.logical_or(estimate, ground_truth).sum(dtype=np.float_)

def mse_criterion(estimate, ground_truth):
    return np.mean(np.square(estimate - ground_truth))

def threshold(estimate):
    threshold_value = 0.5
    out = np.zeros(estimate.shape)
    out[estimate > threshold_value] = 1
    return out

def plot_validation_results(model_name, epoch_ids, jaccard_distances):
    plt.figure()
    plt.title('Model ' + model_name)
    plt.xlabel('Epoch')
    #plt.ylabel('Jaccard distance')
    plt.ylabel('mse distance')
    plt.plot(epoch_ids, jaccard_distances)

def validate(model, model_filename, model_name, input_data, output_data):
    model_file = open(model_filename, 'rb')
    epoch_ids = []
    jaccard_distances = []
    mse_errors = []
    # For each logged set of weights until we reach EOF
    while True:
        try:
            epoch_ids.append(model.load_weights(model_file))
        except EOFError:
            break
        print 'Epoch ' + str(epoch_ids[-1])
        # Compute the estimate
        for i in xrange(0, len(input_data)):
            image = input_data[i].reshape(sqrt(input_data.shape[1]),sqrt(input_data.shape[1]))
            estimated_binarisation = model.apply(input_data[i:i+1])
            #estimated_binarisation=threshold(estimated_binarisation)
            estimated_image = estimated_binarisation.reshape(sqrt(output_data.shape[1]),sqrt(output_data.shape[1]))
            gt = output_data[i].reshape(sqrt(output_data.shape[1]),sqrt(output_data.shape[1]))

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
            ## TODO : implement something that will take the middle of the patch for input data
            
        estimate = model.apply(input_data)
        # In a first time, we used a jaccard criterion to compute the error, but we changed it to a mse one.
        # Compute Jaccard distance between the estimate and the ground truth
        # /
        #binarised_estimate = threshold(estimate)
        #jaccard_distances.append(jaccard_distance(binarised_estimate, output_data))
        # \
        mse_errors.append(mse_criterion(estimate, output_data))

    model_file.close()
    plot_validation_results(model_name, epoch_ids, mse_errors)
    mse_errors = np.array(mse_errors)
    best_model = epoch_ids[mse_errors.argmin()]
    print 'The best model for ' + model_name + ' is the number ' + str(best_model)

def main():
    # Parse command line args
    if len(sys.argv) < 2:
        print 'USAGE: validate.py project_file.json [models_dir]'
        sys.exit(1)
    project_file = sys.argv[1]
    if len(sys.argv) >= 3:
        models_dir = sys.argv[2]
    else:
        models_dir = '.'
    # Check that the models directory exists
    if not os.path.exists(models_dir):
        raise IOError('The models directory does not exist.')
    # Load project
    project = Project(project_file)
    # Generate validation data
    validation_dataset = project.get_validation_dataset()
    in_validation_genconfig = project.get_input_validation_genconfig()
    out_validation_genconfig = project.get_output_validation_genconfig()

    input_data = validation_dataset.generate_input_data(in_validation_genconfig)
    output_data = validation_dataset.generate_ground_truth_data(out_validation_genconfig)
    print str(input_data.shape[0]) + " training samples were generated."
    print "Each input sample contains " + str(input_data.shape[1]) + " pixels."
    print "Each out sample contains " + str(output_data.shape[1]) + " pixels."
    # Validate each model
    for i in range(project.get_models_count()):
    #TODO : implement a validation method that doesn't need training to be finished, so that validation error could be visualized while the app is still running
        model = project.get_model(i)
        model_name = project.get_model_name(i)
        print 'Validating model ' + model_name + '...'
        model_filename = os.path.join(models_dir, model_name + '.model')
        validate(model, model_filename, model_name, input_data, output_data)
    # Show all plots
    plt.show()

if __name__ == '__main__':
    main()
