#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from binarisation.project import Project
import os.path
import sys
import numpy as np

def jaccard_distance(estimate, ground_truth):
    return 1.0 - np.logical_and(estimate, ground_truth).sum(dtype=np.float_) / np.logical_or(estimate, ground_truth).sum(dtype=np.float_)

def threshold(estimate):
    threshold_value = 0.5
    out = np.zeros(estimate.shape)
    out[estimate > threshold_value] = 1
    return out

def validate(model, model_filename, input_data, output_data):
    model_file = open(model_filename, 'rb')
    # For each logged set of weights until we reach EOF
    while True:
        try:
            epoch_id = model.load_weights(model_file)
        except EOFError:
            break
        print "Epoch " + str(epoch_id)
        # Compute the estimate
        estimate = model.apply(input_data)
        # Compute Jaccard distance between the estimate and the ground truth
        binarised_estimate = threshold(estimate)
        print "Jaccard distance: " + str(jaccard_distance(binarised_estimate, output_data))
    model_file.close()

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
        raise IOError("The models directory does not exist.")
    # Load project
    project = Project(project_file)
    # Generate validation data
    validation_dataset = project.get_validation_dataset()
    validation_genconfig = project.get_validation_genconfig()
    input_data = validation_dataset.generate_input_data(validation_genconfig)
    output_data = validation_dataset.generate_ground_truth_data(validation_genconfig)
    print str(input_data.shape[0]) + " validation samples were generated."
    print "Each sample contains " + str(input_data.shape[1]) + " pixels."
    # Validate each model
    for i in range(project.get_models_count()):
        model = project.get_model(i)
        model_filename = project.get_model_name(i)
        print "Validating model " + model_filename + "..."
        model_filename = os.path.join(models_dir, model_filename + ".model")
        validate(model, model_filename, input_data, output_data)

if __name__ == '__main__':
    main()

