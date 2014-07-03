import feature_man as fm	
import matplotlib.pyplot as pl
import matplotlib.cm as cm
import numpy as np

def plot_clustering_result(data,clustering,i,j,size = (10,6)):
	pl.figure(figsize=size,dpi=80)
	centroids = clustering.cluster_centers_
	clusters = clustering.predict(data)
	pl.scatter(data[:,i], data[:,j], c=clusters,cmap = cm.gist_rainbow,s = 30)
	
	pl.scatter(centroids[:,i],centroids[:,j],marker='x', c ='r',linewidths=2,s = 100)
	pl.show()

# plotta i risultati del clustering e i dati etichettati in due subplot
def plot_clustering_and_targets(data,clustering,i,j,targets,size = (10,6)):
	f, axarr = pl.subplots(2, sharex=True,figsize=size)
	centroids = clustering.cluster_centers_
	clusters = clustering.predict(data)
	#for t,c in zip(targets,clusters):
	#	print t," ",c
	axarr[0].scatter(data[:,i], data[:,j], c=clusters,cmap = cm.gist_rainbow,s = 80)
	axarr[0].scatter(centroids[:,i],centroids[:,j],marker='x', c ='r',linewidths=2,s = 100)
	
	axarr[1].scatter(data[:,i], data[:,j], c=targets,cmap = cm.gist_rainbow,s = 80)
	axarr[1].scatter(centroids[:,i],centroids[:,j],marker='x', c ='r',linewidths=2,s = 100)
	
	pl.show()
"""
def plot_DBSCAN_clustering_result(data,clustering,i,j):
	pl.figure(figsize=(10,6),dpi=80)
	clusters = clustering.fit_predict(data)
	centroids = clustering.components_
	pl.scatter(data[:,i], data[:,j], c=clustering.labels_)
	centr_col = range(0,len(centroids))
	pl.scatter(centroids[:,i],centroids[:,j],marker='x', c ='r',linewidths=2,s = 100)
	pl.show()
"""
def plot_all(data):
	pl.figure(figsize=(10,6),dpi=80)
	if len(data.shape)==1:
		pl.plot(data)
	else: 
		for i in range(0,data.shape[1]):
			pl.plot(data[:,i])
	pl.show()
	
def plot_in_subplots(data,varstart,varend):
	nsubplots = varend-varstart
	f, axarr = pl.subplots(2, nsubplots/2, sharex=True,figsize=(18,8))
	for (i,ax) in zip(range(varstart,varend),axarr.flatten()):
		ax.plot(data[:,i])
		ax.set_title(fm.original_header[i])
	pl.show()


def plot_confusion_matrix(cm):
	pl.matshow(cm)
	pl.title('Confusion matrix')
	pl.colorbar()
	pl.ylabel('True label')
	pl.xlabel('Predicted label')
	pl.show()



def plot_in_subplots_varlist(data,varlist):
	nsubplots =len(varlist)
	f, axarr = pl.subplots(2, nsubplots/2, sharex=True,figsize=(18,8))
	for (i,ax) in zip(varlist,axarr.flatten()):
		ax.plot(data[:,i])
		ax.set_title(fm.header[i])
	pl.show()

# not working
def contour_plot(X,Y,model):
		h = 1  # step size in the mesh
		# create a mesh to plot in
		gr = []
		fl_gr = []
		for i in range(0, model.steps[1][1].n_components):
			gr.append( (X[:, i].min() - 1, X[:, i].max() + 1) )
			fl_gr.append(np.arange(X[:, i].min() - 1,  X[:, i].max() + 1, h))
		#x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
		#y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
		#print y_min, y_max
		ms = np.meshgrid(fl_gr[0],fl_gr[1],fl_gr[2],fl_gr[3],fl_gr[4])
		rr =[r.ravel() for r in ms]
		#print len(rr)
		r =np.transpose(rr)
		#print r.shape
		#print xx.shape, '\n',np.c_[xx.ravel(), yy.ravel()]
		Z =model.steps[2][1].predict(r)
		#ms = np.meshgrid(fl_gr[0],fl_gr[1])
		xx = ms[0]
		yy = ms[1]
		print xx.shape
		print X.shape
		# Put the result into a color plot
		Z = Z.reshape(xx.shape)
		print xx.shape
		print X.shape
		pl.contourf(xx[:,:,0,0,0], yy[:,:,0,0,0], Z[:,:,0,0,0], cmap=pl.cm.hsv)
		#pl.contourf(xx, yy, Z, cmap=pl.cm.hsv)
		
		pl.axis('off')

		# Plot also the training points
		pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.hsv,s = 100)
		pl.show()
		
		
		
		
		
