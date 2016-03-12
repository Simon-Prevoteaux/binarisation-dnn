#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from binarisation.project import Project
from binarisation.neuralnet import NeuralNetwork
from binarisation.datasets import DataSet
import sys
import os
import os.path
import numpy as np

#SAMPLES_DIR = "data/train/samples"
#GROUND_TRUTH_DIR = "data/train/groundTruth"
#DATASET_CONFIG_FILE = "config/data.json"
#NETWORK_CONFIG_FILE = "config/default_configuration.json"
#WEIGHTS_SAVE_FILE = "config/weights.sav"
#
#def create_neural_network(features_count):
#    neural_net = NeuralNetwork()
#    neural_net.initialise_from_file(features_count, NETWORK_CONFIG_FILE)
#    return neural_net
#
#def create_training_data():
#    samples_ds = DataSet(SAMPLES_DIR)
#    gt_ds = DataSet(GROUND_TRUTH_DIR)
#    data = TrainingData(samples_ds, gt_ds)
#    data.load_config_file(DATASET_CONFIG_FILE)
#    genconf = data.random_gen_config(patches_per_image=10)
#    input_data = data.generate_input_data(genconf)
#    output_data = data.generate_ground_truth_data(genconf)
#    print str(input_data.shape[0]) + " training samples were generated."
#    print "Each sample contains " + str(input_data.shape[1]) + " pixels."
#    return (input_data, output_data)
#
#def main():
#    print "Loading data..."
#    (samples, ground_truth) = create_training_data()
#    print "Initialising neural network..."
#    neural_net = create_neural_network(samples.shape[1])
#    print "Training network..."
#    neural_net.train(samples, ground_truth)
#    print "Saving network state..."
#    neural_net.save_weights(WEIGHTS_SAVE_FILE)

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
    training_genconfig = project.get_training_genconfig()
    input_data = training_dataset.generate_input_data(training_genconfig)
    output_data = training_dataset.generate_ground_truth_data(training_genconfig)
    print str(input_data.shape[0]) + " training samples were generated."
    print "Each sample contains " + str(input_data.shape[1]) + " pixels."
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

