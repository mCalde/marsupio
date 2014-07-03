import random
from sklearn.decomposition import PCA
from sklearn import preprocessing
import numpy as np
from scipy.signal import medfilt

def median_filter(data,ksize=3):
	fdata = []
	for i in range(0,data.shape[1]):
		fdata.append(medfilt(data[:,i], kernel_size=ksize))
	return np.transpose(np.array(fdata))

def scale(dataset,feature_range=(0,1)):
	min_max_scaler = preprocessing.MinMaxScaler(feature_range=feature_range)
	
	X_scaled =  min_max_scaler.fit_transform(dataset)
	return X_scaled
