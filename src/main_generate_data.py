#!/usr/bin/python
# -*- coding: utf-8 -*-

from datasets import DataSet
import sys
import numpy as np
import json
import os
import pickle 

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
	print('Training data set')
	x_train, y_train=generate_data(ds_train_sample,ds_train_GT,config)
	print('Test data set')
	x_test, y_test=generate_data(ds_test_sample,ds_test_GT,config)

	
	#faire une selection aléatoire des patchs ???	

	x_train=np.asarray(x_train)
	y_train=np.asarray(y_train)

	x_test=np.asarray(x_test)
	y_test=np.asarray(y_test)

	train_set={}
	train_set['x_train']=x_train
	train_set['y_train']=y_train

	test_set={}
	test_set['x_test']=x_test
	test_set['y_test']=y_test

	print('Saving data')

	pickle.dump(train_set,open(os.path.join('results',"training_set.pck"),'w'),protocol=-1)
	pickle.dump(test_set,open(os.path.join('results',"testing_set.pck"),'w'),protocol=-1)



def getConfig():
    config={}
    config['bloc_size'] = 50
    config['space'] = 25

    return config

def saveConfig(config,name='configuration.json',path='results'):
	print('Saving used configuration')
	json.dump(config,open(os.path.join(path,name),'wb'),indent=2)



def generate_data(sample,gt,config):
	"""
	Permit to generate data from two sets

	Args:
		sample : the set corresponding containing all the raw images we want to crop
		gt : the groundTruth of the binarisation of the sample set
		config : the configuration containing different parameters on how generating data.
	"""

	temp=sample.common_images(gt)

	x=[]
	y=[]
	for name in temp:
		print('loading image ',name)
		sampleImTemp=sample.open_image(name)
		sampleImTemp=sampleImTemp.convert('L') #convert into greyscale image

		gtImTemp=gt.open_image(name)
		generate_data_from_image(sampleImTemp,config,x)
		generate_data_from_image(gtImTemp,config,y)

	return x,y


def generate_data_from_image(image,config,result):
	"""
		Allow to get a list of arrays, each one containing a bloc_size*bloc_size part of the image.

		Args
			image : the image we want go get data on
			config : the configuration containing different parameters on how generating data.
	"""

	bloc_size=config['bloc_size']
	space=config['space']

	width, height = image.size   # Get dimensions

	for i in range(0,width-bloc_size,space):
		for j in range(0,height-bloc_size,space):
			temp=image.crop((i,j,i+bloc_size,j+bloc_size))
			result.append(np.asarray(temp.getdata()))




if __name__=='__main__':

	if (len(sys.argv)<2):
		dataDir='./dataSauvola'
	else:
		dataDir=sys.argv[1:][0]

	main(dataDir)

