#!/usr/bin/python
# -*- coding: utf-8 -*-

from datasets import DataSet
import sys
import numpy as np
import json
import os
import pickle 


def main(dataDir):
	training_set=pickle.load(open(dataDir+'/training_set.pck'))
	x_train=training_set['x_train']
	y_train=training_set['y_train']


def loadDataConfig(name='configuration.json',path='results'):
	print('Loading used configuration')
	return json.load(open(os.path.join(path,name),'rb'))



if __name__=='__main__':

	if (len(sys.argv)<2):
		dataDir='./results'
	else:
		dataDir=sys.argv[1:][0]

	main(dataDir)
