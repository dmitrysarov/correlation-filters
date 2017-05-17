# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 14:57:25 2017

@author: temp
"""

import corFilter

qcf = corFilter.QCF(corFilterSize = 64, posEigen = 1, negEigen = 1, fastQC = False) 
# corFilterSize - filter size in pixels, posEigen/negEigen - number of positive and negative eigen vectors
# fastQC=True in case of train set less then image size
qcf.getTrueClassFilesList(r'example_set\true')
qcf.getFalseClassFilesList(r'example_set\false')
qcf.assembleFilter() # simple assemble filter from all available images. 
# or you can train filter to prevent overfitting
#qcf.train(r'example_set\true', r'example_set\false')
qcf.get_correlations()
rocauc = qcf.get_roc_auc()
qcf.plots()