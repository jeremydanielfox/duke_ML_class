# Jeremy Fox

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import neighbors, datasets
from random import shuffle
# import data



# read file and preprocess the data
def read_and_process(filename):
	file = open(filename, 'r')
	data = file.readlines()
	# strip out newline characters
	data = map(lambda d: d.rstrip(), data)
	# convert to tuple of appropriate form
	data = map(lambda d: d.split(), data)
	data = map(lambda line: (list(line[0:3]),line[-1]), data)
	data = map(lambda d: (map(int,d[0]),int(d[1])),data)
	# switch labels to first spot
	data = map(lambda d: (d[1],d[0]), data)
	shuffle(data)
	return data[:2000]
	# change to 

def nested_cross_validate(param_algo,data,algo_name):
	n = len(data)
	folds = chunkify(data, 10)
	parameters = map(lambda tuple: tuple[0], param_algo)
	algorithms = map(lambda tuple: tuple[1], param_algo)
	d = {}
	for par in parameters:
		d[par] = []
	#d = dict.fromkeys(parameters, [])
	count = 0
	measures = []
	for i, test in enumerate(folds):
		first_folds=folds[:]
		first_folds.pop(i)
		test_df = get_dataframe(test)
		for j, val in enumerate(first_folds):
			train_folds = first_folds[:]
			train_folds.pop(j)
			# flatten
			train_folds = [item for sublist in train_folds for item in sublist]
			train_df = get_dataframe(train_folds) # convert to dataframe
			val_df = get_dataframe(val)
			for tupl in param_algo:
				param = int(tupl[0])
				algorithm = tupl[1]
				algorithm.fit(train_df.iloc[:,1:],train_df.iloc[:,0])
				hard_pred = algorithm.predict(val_df.iloc[:,1:])
				acc = np.isclose(hard_pred,val_df.iloc[:,0]).sum()/float(len(hard_pred))
			
				d[param].append(acc)
		keys = []
		means = []

		for key, value in d.iteritems():
			means.append(sum(value) / float(len(value)))
			keys.append(key)
	
		d = {}
		for par in parameters:
			d[par] = []
		best = np.argmax(means)
		
		best_algo = algorithms[best]
		
		best_algo.fit(train_df.iloc[:,1:],train_df.iloc[:,0])
		hard_pred = best_algo.predict(test_df.iloc[:,1:])
		acc = np.isclose(hard_pred,test_df.iloc[:,0]).sum()/float(len(hard_pred))
	
		measures.append(acc)


		soft_pred = best_algo.predict_proba(test_df.iloc[:,1:])
	
		fpr,tpr,thresh = roc_curve(test_df.iloc[:,0],soft_pred[:,1])
		auc = roc_auc_score(test_df.iloc[:,0],soft_pred[:,1])
		plt.plot(fpr,tpr,label='ROC fold %d (area = %0.2f)' % (i, auc))

	plt.plot([-.1,1.1],[-.1,1.1],"r--",alpha=.5)
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.axhline(0)
	plt.axvline(0)
	plt.legend(loc="lower right")
	plt.savefig("nested" + algo_name + '.png')
	plt.clf()
	mean = sum(measures) / float(len(measures))
	std_dev = np.std(measures)
	return (mean, std_dev)

#data should come in as a list of tuples, where first value is the datapoint and second value is the label
def cross_validate(algorithm, data,algo_name):
	n = len(data)
	folds = chunkify(data, 10)
	auc_list = []
	for index, test in enumerate(folds):
		train_folds = folds[:]
		train_folds.pop(index)
		# flatten
		train_folds = [item for sublist in train_folds for item in sublist]
		# train
		train_df = get_dataframe(train_folds) # convert to dataframe
		
		test_df = get_dataframe(test)
		algorithm.fit(train_df.iloc[:,1:],train_df.iloc[:,0]) # may need to make a copy of the algorithm
		# test
		hard_pred = algorithm.predict(test_df.iloc[:,1:])
		acc = np.isclose(hard_pred,test_df.iloc[:,0]).sum()/float(len(hard_pred))
	

		# use predicted probabilities to construct ROC curve and AUC score
		soft_pred = algorithm.predict_proba(test_df.iloc[:,1:])
	
		fpr,tpr,thresh = roc_curve(test_df.iloc[:,0],soft_pred[:,1])
		auc = roc_auc_score(test_df.iloc[:,0],soft_pred[:,1])
		
		plt.plot(fpr,tpr,label='ROC fold %d (area = %0.2f)' % (index, auc))
		
		
		auc_list.append(auc)
	plt.plot([-.1,1.1],[-.1,1.1],"r--",alpha=.5)
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.axhline(0)
	plt.axvline(0)
	plt.legend(loc="lower right")
	plt.savefig(algo_name + '.png')
	plt.clf()
	mean = sum(auc_list) / float(len(auc_list))
	std_dev = np.std(auc_list)
	return (mean, std_dev)



def get_dataframe(tuples):
	tuples = map(lambda tuple: (tuple[0] if tuple[0]==1 else -1 , tuple[1][0],tuple[1][1],tuple[1][2]), tuples)
	df =  pd.DataFrame(tuples)
	df.columns = ['y', 'R','G','B']
	return df

def chunkify(data, k):
	chunks=[]
	num = len(data)/k
	for i in range(k):
		chunks.append(data[i*num:i*num+num])
	final = chunks[-1][-1]
	if final==data[-1]:
		return chunks
	else:
		index = 0
		for chunk in chunks:
			index = index + len(chunk)
		extra = data[index:]
		for idx, value in enumerate(extra):
			chunks[idx].append(value)
	return chunks

def true_positive_rate(prediction,actual):
	n = np.sum(actual)
	true_pos = 0.0
	for i in range(len(prediction)):
	    if prediction[i] == 1 and actual[i] == 1:
	        true_pos = true_pos +1
	return true_pos/n

def false_positive_rate(prediction,actual):
    n = 0.0
    false_pos = 0.0
    for i in range(len(actual)):
        if actual[i]==0:
            n += 1
        if actual[i]==0 and prediction[i] == 1:
            false_pos += 1
    return false_pos/n

def roc_feature(data, feature, fold):
	values = map(lambda d: (d[0],d[1][feature]), data)
	values = sorted(values, key=lambda value: value[1])
	y_values = map(lambda a: 1 if a[0]==1 else 0,values)

	# use a plane sweep to generate tpr/fpr data
	roc_fpr = []
	roc_tpr = []
	offset = 0.00001
	for pair in values:
	    y_val = pair[0]
	    feature = pair[1]
	    predictions = map(lambda val: 1 if val > feature else 0, map(lambda a: a[1],values))
	    tpr = true_positive_rate(predictions,y_values)
	    fpr = false_positive_rate(predictions,y_values)
	    roc_fpr.append(fpr)
	    roc_tpr.append(tpr)

	plt.plot(roc_fpr,roc_tpr, label='Fold %d ' % fold )
	plt.axis([-.2, 1.2, -.2, 1.2])
	plt.axhline(0)
	plt.axvline(0)
	plt.legend(loc="lower right")
	plt.xlabel('False Positive Rate', fontsize=18)
	plt.ylabel('True Positive Rate', fontsize=16)

def cross_validate_feature(data,feature, lbl):
	n = len(data)
	folds = chunkify(data, 10)
	for index, test in enumerate(folds):
		train_folds = folds[:]
		train_folds.pop(index)
		# flatten
		train_folds = [item for sublist in train_folds for item in sublist]
		# train
		train_df = get_dataframe(train_folds) # convert to dataframe
		
		test_df = get_dataframe(test)

		roc_feature(test, feature, index)
	plt.savefig('feature' + lbl +  '.png')
	plt.clf()


data = read_and_process('/Users/Jeremy/Downloads/Skin_NonSkin.txt')
cross_validate_feature(data,0, 'B')
cross_validate_feature(data,1,'G')
cross_validate_feature(data,2,'R')
# Normal cross validation testing on the algorithms
'''
print "Starting Decision Tree..."
dt = DecisionTreeClassifier()
mean, std = cross_validate(dt,data,"Decision_Tree")
print "mean: " + str(mean)
print "standard deviation: " + str(std)
'''

print "Random Forest"
rf = RandomForestClassifier()
mean, std = cross_validate(rf,data,"Random_Forest")
print "mean: " + str(mean)
print "standard deviation: " + str(std)

print "Gradient Boosted Trees"
gb = GradientBoostingClassifier()
mean, std = cross_validate(gb,data,"Gradient_Boosted_Trees")
print "mean: " + str(mean)
print "standard deviation: " + str(std)

print "Logistic Regression"
lr = LogisticRegression()
mean, std = cross_validate(lr,data,"Logistic_Regression")
print "mean: " + str(mean)
print "standard deviation: " + str(std)

print "Decision Tree Nested"
algo_tuples = []
dt1 = DecisionTreeClassifier(max_leaf_nodes=2)
dt2 = DecisionTreeClassifier(max_leaf_nodes=10)
dt3 = DecisionTreeClassifier(max_leaf_nodes=100)
dt4 = DecisionTreeClassifier(max_leaf_nodes=1000)
algo_tuples.append((2,dt1))
algo_tuples.append((10,dt2))
algo_tuples.append((100,dt3))
algo_tuples.append((1000,dt4))
mean, std = nested_cross_validate(algo_tuples, data, "DecisionTree")
print "mean: " + str(mean)
print "standard deviation: " + str(std)

print "K nearest neighbors Nested"
algo_tuples = []
for k in [1,10,100,1000]:
	algo_tuples.append((k,neighbors.KNeighborsClassifier(k)))
mean, std = nested_cross_validate(algo_tuples,data,"KNN")
print "mean: " + str(mean)
print "standard deviation: " + str(std)








