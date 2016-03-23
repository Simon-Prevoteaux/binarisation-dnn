#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from binarisation.project import Project
import os.path
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import theano

from cv2 import imshow,waitKey

def get_modele_from_name(project, model_name):
	
    for i in range(project.get_models_count()):
    	if (project.get_model_name(i) == model_name):
    		print 'The model '+model_name+' has been correctly found' 
    		return project.get_model(i)
	return None     

def load_best_epoch(model, model_name, model_filename,best_epoch_number):
    model_file = open(model_filename, 'rb')

    k=0
    while k!=best_epoch_number:
        try:
            k=model.load_weights(model_file)
        except EOFError:
            break
    print 'Successfully found the best weights for '+model_name
    model_file.close()

def exhaustive_gen_config(test_image,patches_padding, psize):
    """
    Creates a generation configuration containing the whole images.
    Args:
        patches_padding (numpy.ndarray): (x, y) padding between each patch of the image.
    Returns:
        Generation configuration.
    """
    (width, height) = test_image.size
    curlist = []
    for y in range(0, height - psize, patches_padding[1]):
        for x in range(0, width - psize, patches_padding[0]):
            curlist.append([x, y])
        gen_config = np.array(curlist, np.int16)
    return gen_config

def binarize(psize, gen_config, model,test_image):
    pcount = gen_config.shape[0]
    data = np.empty([pcount, psize * psize], theano.config.floatX)
    i = 0

    data = np.empty([pcount, psize * psize], theano.config.floatX)

    (width, height) = test_image.size

    #image = np.zeros([width,height])
    image=np.zeros([height,width])
    print pcount

    print gen_config

    for j in range(0,pcount-1):
    #for j in range(80000,85000):
        data[j,:] = test_image.crop((gen_config[j][0], gen_config[j][1], gen_config[j][0] + psize, gen_config[j][1]+ psize)).getdata()
        temp=data[j,:]
        estimate = model.apply(temp)
        binarised_estimate = threshold(estimate)

        print j
        '''
        plt.subplot(2,1,1)
        plt.imshow(binarised_estimate.reshape((psize,psize)), interpolation='nearest', cmap=plt.get_cmap('gray'), vmin=0.0, vmax=1.0)
        plt.subplot(2,1,2)
        plt.imshow(temp.reshape((psize,psize)), interpolation='nearest', cmap=plt.get_cmap('gray'), vmin=0.0, vmax=1.0)
        plt.show()
        '''

        image[gen_config[j][1]:gen_config[j][1] + psize,gen_config[j][0]:gen_config[j][0]+ psize] = binarised_estimate.reshape((psize,psize))
    return image

def threshold(estimate):
    threshold_value = 0.5
    out = np.zeros(estimate.shape)
    out[estimate > threshold_value] = 1
    return out

def main():
    # Parse command line args
    if len(sys.argv) < 2:
        print 'USAGE: binarize_image.py project_file.json model_name best_epoch_number test_image_file [models_dir]'
        sys.exit(1)
    project_file = sys.argv[1]

    model_name = sys.argv[2]

    best_epoch_number = sys.argv[3]

    if len(sys.argv) >= 5:
        test_image_file = sys.argv[4]
    else:
    	test_image_file = './dataSauvola/test/samples/29/FRAD076_JPL3_81_0001.tif'

    if len(sys.argv) >= 6:
        models_dir = sys.argv[5]
    else:
        models_dir = '.'

	project = Project(project_file)

	model_filename = os.path.join(models_dir, model_name + '.model')
	model=get_modele_from_name(project,model_name)

    load_best_epoch(model, model_name,model_filename,best_epoch_number)

    test_image=Image.open(test_image_file).convert('F')

    gen_config=exhaustive_gen_config(test_image,[10,10],10)


    binarize_image=binarize(10,gen_config,model,test_image)

    print np.unique(binarize_image)
    print binarize_image.shape # pas la bonne taille
    im = Image.fromarray(np.uint8(binarize_image*255))
    im.show()

if __name__ == '__main__':
    main()
