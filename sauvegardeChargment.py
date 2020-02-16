#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 17:50:42 2019

@author: okimb
"""

from data_process import *
from gbdt_model import gbdt_model, svm, gbdt_model_grid1, gbdt_model_default
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from main import *
    # save model to file
    # pickle.dump(model, open("pima.pickle.dat", "wb"))       
#    pickle.dump(method, open("docclassifier.pickle.date", "wb"))
    
    	
    # load model from file
    # loaded_model = pickle.load(open("pima.pickle.dat", "rb"))
#    loaded_model = pickle.load(open("docclassifier.pickle.dat", "rb"))

