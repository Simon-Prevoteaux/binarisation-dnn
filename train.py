#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from binarisation.project import Project
from binarisation.neuralnet import NeuralNetwork
from binarisation.datasets import DataSet
import sys
import os
import os.path
import numpy as np

def main():
    # Parse command line args
    if len(sys.argv) < 2:
        print 'USAGE: train.py project_file.json [output_dir]'
        sys.exit(1)
    project_file = sys.argv[1]
    if len(sys.argv) >= 3:
        output_dir = sys.argv[2]
    else:
        output_dir = '.'
    # Load project
    project = Project(project_file)
    # Generate training data
    training_dataset = project.get_training_dataset()
    in_training_genconfig = project.get_input_training_genconfig()
    out_training_genconfig = project.get_output_training_genconfig()
    input_data = training_dataset.generate_input_data(in_training_genconfig)
    output_data = training_dataset.generate_ground_truth_data(out_training_genconfig)
    print str(input_data.shape[0]) + " training samples were generated."
    print "Each input sample contains " + str(input_data.shape[1]) + " pixels."
    print "Each out sample contains " + str(output_data.shape[1]) + " pixels."

    # Create output directory if necessary
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Train each model
    for i in range(project.get_models_count()):
        model = project.get_model(i)
        model_name = project.get_model_name(i)
        print 'Training model ' + model_name + '...'
        model_path = os.path.join(output_dir, model_name + '.model')
        model.train(input_data, output_data, open(model_path, 'wb+'))

if __name__ == '__main__':
    main()
