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

Data
=====
In order to train the network, we first used the data from the **DIBCO** competition available here : http://users.iit.demokritos.gr/~kntir/HDIBCO2014/resources.html

Considering the huge difficulty of the dataset, we first decided to use a "customize" dataset, corresponding to binarized newpapers using Sauvola's method. The idea is to pretrain
our network on these non-perfect data and then fine-tune the network on the **DIBCO** dataset.

Usage
=====

In order to train a model, you will need to write a config file as shown in the directory config/default_project.
This file contains every parameter you need to fix concerning the data, the patch size and the network.
It should be noted that it can contains several network model.

Once this is done, you can run the training phase using **$ python train.py ./config/default_project**

Then, **$ python validate.py ./config/default_project** will allow you to visualize the curve corresponding to the validation error.

An intermediary step consists of choosing the best threshold for image binarization. It can be done using the following command :
**$ python validate-threshold config/default_project name number** where :
  * name : the name of the model you want to use
  * number : the number of the epoch which you consider to be the best (result provided by the validate.py file)

Finally, you can run a binarization on a whole image. For that, you will have to use this :
**$ python binarize_image.py config/default_project name number image** where :
  * image : the path of the image you want to binarize
