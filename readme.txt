#readme file
I. Requirements: python 3, with packagest: scipy, numpy, sklearn
II. Main Code: sample.py
#This code file contains the 'sample' class, which is the implementation of the unified sampling 
function
a. initializtion: 
	clf: a classifier object
	data: data (x)
	Y: label (t)
	train_index: indices of initial training set L_0
	candidate_index: indices of initial unlabeled pool U_0
	test_index: indices of test set
	M: max active learning iterations	
	gam: initial balancing parameter gamma_0
	lam: initial balancing parameter lambda_0
	gam_ur: update rate of balancing parameter delta_gamma
	lam_ur: updata rate of balancing parameter delta_lambda
	gam_clf: gamma parameter for RBF kernel (=1/2l^2, l = length scale)
b. method:
		dtrustSample( self,n_batch = 1 ): perform dtrust sampling on the current unlabeled pool,
			re-train model on the new training set
			n_batch: the batch size, default = 1 
		evaluate( self ): return the current test accuracy 
III. Sample Code:
#below is a sample of what the active learning code should look like
#the capitalized keywords should be replaced with real code
from DIRECTORY import sample
from sklearn import svm;
from sklearn.multiclass import OneVsRestClassifier

data, y = READ_DATA( DATASET )
train_index, candidate_index, test_index = SPLIT_DATA( data, y )
gam_clf = 1.0
c_clf = 1.0
M = 500
clf_binary = svm.SVC( C = c_clf, probability = True, tol = 0.001, kernel = 'rbf', gamma = gam_clf)
clf = OneVsRestClassifier( clf_binary )
new_sample = sample.sample( clf, data, y, train_index, candidate_index, test_index, M,\
							gam = GAM, gam_ur = GAMUR, lam = LAM, lam_ur = LAMUR, gam_clf)
accuracy = []
for i in range(500):
	new_sample.dtrustSample()
	accuracy.append(new_sample.evaluate())
	