Binarization Using Neural Networks
==================================

This library aims to provide a method for training, validating and testing a neural network for image binarization.

Setup
======

The key packages required are :
  * numpy
  * theano
  * crino
  * pillow
  * matplotlib

Crino
======

For more details for the installation, see https://github.com/jlerouge/crino

Usage
=====

In order to train a model, you will need to write a config file as shown in the directory config/default_project.
This file contains every parameter you need to fix concerning the data, the patch size and the network.
It should be noted that it can contains several network model.

Once this is done, you can run the training phase using $ python train.py ./config/default_project

Then, $ python validate.py ./config/default_project will allow you to visualize the curve corresponding to the validation error.

Finally, you can run a binarization on a whole image. For that, you will have to use this :
$ python binarize_image.py config/default_project.json name_of_the_model_you_want_to_use number_of_the_best_epoch path_to_image
