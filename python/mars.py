#from os.path import isfile, join
#import string
import sys
#import os
#import getopt
#import random
import plotting as plt
import preprocessing as prproc
from sklearn.decomposition import PCA,KernelPCA
from sklearn import preprocessing
#from sklearn.pipeline import Pipeline
#from sklearn import metrics
from sklearn.cluster import KMeans
#from sklearn import cross_validation
#from sklearn.cross_validation import train_test_split
#from sklearn.grid_search import GridSearchCV
import numpy as np
import feature_man as fm
import classification as cl
# loads dataset from alldatafile
# returns the segment data ( each element of segmdata is a segment of the data of length wl )

def load_dataset(wl,ol,framefile='../script/stepstats.csv',alldatafile='/home/ilaria/Scrivania/marsupio/acquisizione20062014/acquisizione_20062014/Data_115818.txt',):
	

	data = fm.read_data_file(alldatafile,keep_marker=False)

#	data = prproc.median_filter(data)
	#print data.shape
	# calc theta value as arccos(z_norm) for each item
	
	th = fm.calc_theta(data)
	data = fm.add_features(data,th)
	#print data.shape
	segmdata = fm.segment( data,wl,ol)
	
	return data,segmdata
	

# takes the segmented dataset as input
# returns a new dataset where each item represents a segment via some features (ie mean, stdev ecc.)

def build_dataset_features(segmdata,silent=True):
	dataset = []
	header = []
	for (feat,feat_fun) in fm.feature_list:
		
		features = []
		if not silent:
			print "Calculating ", feat,"...."
		for sitem in segmdata:
			features.append( feat_fun(sitem))
		if not len(dataset):
				dataset = np.array(features)
		else:
				dataset = fm.add_features(dataset,features)
		if not silent:
			print "Dataset shape ", dataset.shape
	header = fm.create_header()
	return dataset,header

def write_feature_data_to_file(dataset,header,outputfile = './featuredata.csv'):
	fd = open(outputfile,'w')
	line = ""
	for label in header:
		line+=(label+",")
	line = line[:-1]
	line+='\n'
	fd.write(line)
	for item in dataset:
		line = ""
		for feat in item:
			line+=(str(feat)+",")
		line = line[:-1]
		line+='\n'
		fd.write(line)
	return
def select_indexes(varlist,suff):
	indexs=[]
	if suff   in ('_avg','_stdev','_median','_min','_max','_mad'):
		for tpl in varlist:
				s = (tpl+suff)
				print s
				indexs.append(fm.label_index[s])
	else:
		for tpl in varlist:
				s = ""
				for t in tpl:
					s+=(fm.original_header[t]+"_")
				s=s[:-1]
				s = (s+suff)
				print s
				indexs.append(fm.label_index[s])
	return indexs
# MAIN
def main(argv):
	# defaults
	
	window_length = 50
	overlap = window_length/2
	featdim = 10
	
	training_data,training_sgmdata = load_dataset(window_length,overlap)
	
	training_featdata,header = build_dataset_features(training_sgmdata)
	training_targets = fm.assign_target(training_featdata)

	"""
	data1,sgmdata1 = load_dataset(window_length,overlap,alldatafile='/home/ilaria/Scrivania/marsupio/acquisizione20062014/acquisizione_20062014/Data_120250.txt')
	featdata1,_ = build_dataset_features(sgmdata1)
	targets1 = fm.assign_target(featdata1)
	"""
	
	#write_feature_data_to_file(featdata,header)
	#print featdata[0,idxs]
	#plt.plot_in_subplots(featdata,idxs)
	#plt.plot_all(featdata1[:,idxs])
	
	
	#X_r=preprocessing.scale(featdata)
	#pca = PCA(n_components=featdim)
	
		
	#kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=0.1)
	#X_r = kpca.fit_transform(X_r)
	#X_r = pca.fit(X_r).transform(X_r)
	
	X_r = training_featdata
	targets = training_targets
	pca = PCA(n_components=2)
	X_r = preprocessing.scale(X_r)
	X_r = pca.fit(X_r).transform(X_r)
	kmeans = KMeans(n_clusters=10)
	kmeans.fit(X_r)
	plt.plot_clustering_and_targets(X_r,kmeans,0,1,targets)
	return
	pars =[{'clf__kernel': ['rbf'], 'clf__gamma': [1e-3, 1e-5,1e-2,1e-1,1e-4],
						 'clf__C': [0.001,0.01,0.1,1, 10, 100], 'pca__n_components': [5,10,20,50,80]},
						{'clf__kernel': ['linear'], 'clf__C': [0.001,0.01,0.1,0.5,1, 10, 100], 'pca__n_components': [5,10,20,50,80]}]

	#evaluation set
	cl.cross_model_selection(X_r,targets,pars,save=True)
	c = cl.load_model('model.pkl')
	print c
	return
	
	#print X_train.shape, X_test.shape
	clf = svm.SVC(kernel='rbf', gamma=0.7,C=0.8)
	pca = PCA(n_components=featdim)
	pca_svm = Pipeline([('pca', pca), ('svm',clf),])
	scores = cross_validation.cross_val_score( clf,X_r, targets, cv=5,scoring='acc')
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	#pca_svm.fit(X_train, y_train)
	#print pca_svm.score(X_test,y_test)
	return
	#X_r = pca.fit(sint).transform(sint)
	
	#X_r = preprocessing
	pca = PCA(n_components=featdim)
	
		
	#kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=0.1)
	#X_r = kpca.fit_transform(X_r)
	X_r = pca.fit(X_r).transform(X_r)
	ncluster = 10
	"""
	from sklearn.cluster import DBSCAN
	dbscan = DBSCAN()
	
	plt.plot_DBSCAN_clustering_result(X_r,dbscan,0,1)
	return
	"""
	#X_r = preprocessing.scale(X_r)
	kmeans = KMeans(n_clusters=ncluster)
	#print X_r
	kmeans.fit(X_r)
	plt.plot_clustering_and_targets(X_r,kmeans,0,1,target)

	return
	"""
	test = open('./test.csv','w')
	for dt in sint:
		for ft in dt:
			test.write(str(ft)+',')
		
		test.write('\n')
	"""
	#colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
	#colors = np.hstack([colors] * 20)
	
	featdim = 10
	
	Y= randomtargets(sint)
	clf = svm.SVC(kernel='rbf', gamma=0.7)
	pca = PCA(n_components=featdim)
	pca_svm = Pipeline([('pca', pca), ('svm',clf),])
	
	pca_svm.fit(sint, Y) 
	X_r = pca.fit(sint).transform(sint)
	cX_r = pca.fit(sint).transform(cint)
	#th1 = [l[1] for l in sint]
	#accx1 = [l[2] for l in sint]
	#print(th1)
	#plt.scatter(th1, accx1, 50,c=Y)
	#plt.show()
	
	features = []
	for i in range(0,featdim):
		features.append([l[i] for l in cX_r])
	Yp = [int(i) for i in pca_svm.predict(cint)]
	print Yp
	s = 411
	for  f in features[1:5]:
	#	plt.subplot(s) 
	#	plt.scatter(features[0], f, 50,c=Yp)
		i+=1
		s+=1
	
	#plt.show()
	s = 511
	for  f in features[5:10]:
	#	plt.subplot(s) 
	#	plt.scatter(features[0], f, color=colors[Yp].tolist())
		i+=1
		s+=1
	
	#plt.show()
	print clf.support_vectors_
#	plt.scatter(clf.support_vectors_,range(0,3), color=colors[range(0,3)].tolist())
	# create a mesh to plot in
	sint = np.array(sint)
	Y =(np.array(Y))

	x_min, x_max = sint[:, 2].min() - 1, sint[:, 2].max() + 1
	y_min, y_max = Y.min() - 1, Y.max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),np.arange(y_min, y_max, .02))
	#print len(Y), yy.shape
	#Z = Y.reshape(yy.shape)
	pl.contourf(xx, yy, Y, cmap=pl.cm.Paired)
	pl.axis('off')

    # Plot also the training points
	pl.scatter(X[:, 1], X[:, 2], c=Y, cmap=pl.cm.Paired)
	pl.show()
	return
	#intervalslist=scale(intervalslist)
	#print intervalslist
	featdim = 5
	ncluster = 8
	clusters = range(1,ncluster+1)

	pca = PCA(n_components=featdim)
	X_r = pca.fit(intervalslist).transform(intervalslist)
	features = []
	for i in range(0,featdim):
		features.append([l[i] for l in X_r])
	
	#return
	kmeans = KMeans()
	#print X_r
	pca_clustering = Pipeline([('pca', pca), ('minmaxnorm',preprocessing.Normalizer()), ('kmeans', kmeans)])
	clustering = Pipeline([ ('kmeans', kmeans)])
	print pca_clustering.fit(intervalslist)
	#return
	pca_clusters = pca_clustering.predict(intervalslist)

	clustering.fit(intervalslist)
	nopca_clusters = clustering.predict(intervalslist)
	clustered = []
	i = 0
	s = 411
	for  f in features[1:]:
		plt.subplot(s) 
		plt.scatter(features[0], f, color=colors[pca_clusters].tolist())
		i+=1
		s+=1
	
	plt.show()
	
	"""
	try:
	opts, args = getopt.getopt(argv, "p:o:", [ "perc=", "all"])
	except getopt.GetoptError:
	#usage()
	return

	for opt, arg in opts:
	if opt in ("-p", "--perc"):
	perc = float(arg)

	if opt in ("-o"):
	output = arg
	if opt in ("--all"):
	_all = 1
	"""

if __name__ == "__main__":
	main(sys.argv[1:])
