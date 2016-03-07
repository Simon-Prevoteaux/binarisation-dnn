#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from datasets import DataSet
from training_data import TrainingData
from neuralnet import NeuralNetwork
import numpy as np

SAMPLES_DIR = "data/train/samples"
GROUND_TRUTH_DIR = "data/train/groundTruth"
DATASET_CONFIG_FILE = "config/data.json"
NETWORK_CONFIG_FILE = "config/default_configuration.json"
WEIGHTS_SAVE_FILE = "config/weights.sav"

def create_neural_network(features_count):
    neural_net = NeuralNetwork()
    neural_net.initialise_from_file(features_count, NETWORK_CONFIG_FILE)
    return neural_net

def create_training_data():
    samples_ds = DataSet(SAMPLES_DIR)
    gt_ds = DataSet(GROUND_TRUTH_DIR)
    data = TrainingData(samples_ds, gt_ds)
    data.load_config_file(DATASET_CONFIG_FILE)
    genconf = data.random_gen_config(patches_per_image=10)
    input_data = data.generate_input_data(genconf)
    output_data = data.generate_ground_truth_data(genconf)
    print str(input_data.shape[0]) + " training samples were generated."
    print "Each sample contains " + str(input_data.shape[1]) + " pixels."
    return (input_data, output_data)

def main():
    print "Loading data..."
    (samples, ground_truth) = create_training_data()
    print "Initialising neural network..."
    neural_net = create_neural_network(samples.shape[1])
    print "Training network..."
    neural_net.train(samples, ground_truth)
    print "Saving network state..."
    neural_net.save_weights(WEIGHTS_SAVE_FILE)

if __name__ == '__main__':
    main()

