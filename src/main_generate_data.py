#!/usr/bin/python
# -*- coding: utf-8 -*-

from datasets import DataSet
import sys
import numpy as np
import json
import os

def main(filename):

	# creating the datasets
	ds_train_sample=DataSet(filename+'/train/samples')
	ds_train_GT=DataSet(filename+'/train/groundTruth')

	ds_test_sample=DataSet(filename+'/test/samples')
	ds_test_GT=DataSet(filename+'/test/groundTruth')

	#loading the images
	ds_train_sample.preload()
	ds_train_GT.preload()

	ds_test_sample.preload()
	ds_test_GT.preload()

	print('Datasets created')

	config=getConfig()
	saveConfig(config)

	print('Creation of the data')
	generate_data(ds_train_sample,ds_train_GT,config)

def getConfig():
    config={}
    config['bloc_size'] = 16
    config['space'] = 16

    return config

def saveConfig(config,path='results'):
	print('Saving used configuration')
	json.dump(config,open(os.path.join(path,"configuration.json"),'wb'),indent=2)

    
def generate_data(sample,gt,config):
	"""
	Permit to generate data from two sets

	Args:
		sample : the set corresponding containing all the raw images we want to crop
		gt : the groundTruth of the binarisation of the sample set
	"""

	temp=sample.common_images(gt)

	print(len(temp))



def generate_data_from_image(image,bloc_size,space):
	"""
		Allow to get a list of arrays, each one containing a bloc_size*bloc_size part of the image.

		Args
			image : the image we want go get data on
			config : the configuration containing different parameters on how generating data.
	"""

	width, height = image.size   # Get dimensions
	result=[]
	for i in range(0,width-bloc_size,space):
		for j in range(0,height-bloc_size,space):
			temp=image.crop((i,j,i+bloc_size,j+bloc_size))
			result.append(np.asarray(temp.getdata()))
	
	return result



if __name__=='__main__':

	if (len(sys.argv)<2):
		dataDir='./data'
	else:
		dataDir=sys.argv[1:][0]

	main(dataDir)

