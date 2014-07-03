from os.path import isfile, join
import matplotlib.pyplot as pl
import matplotlib.cm as cm
import string
import sys
import os
import random
from sklearn.decomposition import PCA
from sklearn import preprocessing
import numpy as np
from collections import Counter	
from scipy.fftpack import fft,rfft
from numpy import linalg as LA
"""
indici per file svid vecchio

th1 = 0
accX1 = 1
accY1 = 2
accZ1 = 3
SMA1 = 4

th2 = 5
accX2 = 6
accY2 = 7
accZ2 = 8
SMA2 = 9

th3 = 10
accX3 = 11
accY3 = 12
accZ3 = 13
SMA3 = 14
"""

"""
dataset_var = {
"th1" = 0,
"accX1" = 1,
"accY1" = 2,
"accZ1" = 3,
"SMA1" = 4,

"th2" = 5,
"accX2" = 6,
"accY2" = 7,
"accZ2" = 8,
"SMA2" = 9,

"th3" = 10,
"accX3" = 11,
"accY3" = 12,
"accZ3" = 13,
"SMA3" = 14
}
"""

accX1 = 0
accY1 = 1
accZ1 = 2
mgX1 = 3
mgY1 = 4
mgZ1 = 5

accX2 = 6
accY2 = 7
accZ2 = 8
mgX2 = 9
mgY2 = 10
mgZ2 = 11

accX3 = 12
accY3 = 13
accZ3 = 14
mgX3= 15
mgY3 = 16
mgZ3 = 17

th1=18
th2=19
th3=20

def assign_target(dataset):
	i= 0
	idx = label_index['accX2_stdev']
	#print idx
	#print dataset.shape
	st = preprocessing.scale(dataset[:,idx])
	oldval=100
	cl =[]
	count=0
	n = 0
	stop = len(st)
	for val in st:
		if n <= stop:
			if val>0.1:
				if   oldval<0.1:#( (val > oldval and oldval >0.1) or (val<oldval and count<1) ):
					#print val,oldval
					i+=1
					count = 0
				else:
					count+=1
		else:
			if val>0.1 :
				if   oldval<0.1 or abs(val-oldval)>2:#( (val > oldval and oldval >0.1) or (val<oldval and count<1) ):
					#print val,oldval
					i+=1
					count = 0
				else:
					count+=1
		if n==stop:
			i = 0
		n+=1
		cl.append(i)
		oldval = val
	# repeated positions
	
	cl = [0 if c in [0,5,10,13] else c for c in cl ]
	cl = [4 if c in [4,9] else c for c in cl ]
	cl = [7 if c in [7,11] else c for c in cl ]
	#print cl
	#pl.scatter(range(0,len(st[:])),st[:],s = 100, c=cl[:],cmap = cm.rainbow)
	#pl.plot(st[:])
	#pl.show()
	return cl
	#pl.scatter(range(0,len(st)),st,s = 100, c=cl,cmap = cm.hot)
	#pl.plot(st)
	#pl.show()
	return
	"""
	oldvals = []
	targets = []
	n= 20
	c = 0
	for val in ref:		
		if c < 180:
			targets.append(0)
		elif c < 500:
			targets.append(1)
		elif c < 750:
			targets.append(2)
		elif c < 950:
			targets.append(3)
		elif c < 1280:
			targets.append(4)
		elif c < 1650:
			targets.append(5)
		elif c < 1900:
			targets.append(6)
		elif c < 2200:
			targets.append(7)
		elif c < 2420:
			targets.append(8)
		elif c < 2820:
			targets.append(9)
		elif c < 3120:
			targets.append(10)
		elif c < 3450:
			targets.append(11)
		elif c < 3750:
			targets.append(12)
		elif c < 4020:
			targets.append(13)
		elif c < 4100:
			targets.append(14)
		else:
			targets.append(15)
		c+=1
		oldvals.append(val)
		if abs(oldvals[max(step,-len(oldvals))] - val) > thres:
			count+=1
		else:
			count = 0
		if count > n:
			count = 0
			print oldvals[max(step,-len(oldvals))] , val
			i+=1
		targets.append(i)
		"""
	
	return


def read_interval_file(f):	
	intfile = open(f,'r')
	lines = [ l.rstrip() for l in intfile.readlines()]
	#header =[l.replace('"','') for l in lines[0].split(',')]
	lines = lines[1:]
	#intervalsdict = []
	intervalslist = []
	# carica in intervals tutti gli intervalli creati
	# in ordine di indice (ie temporale)

	for l in lines:
		llist = [float(ll.replace("'",'')) for ll in l.split(',')]
	#	tmpdict = {}
		i = 0
		intervalslist.append([f for f in llist[1:]])
	#	for h in header:

	#		tmpdict[h] = llist[i]
	#		i+=1
	#		intervalsdict.append(tmpdict)	
	return np.array(intervalslist)

# reads first data files (s.vid1.csv etc.)
def read_original_files_svid_version(f):
	data = []
	for i in range(1,4):
			rfile = open(f+str(i)+'.csv','rU')
			lines = [ [np.float64(label) for label in l.rstrip().split(';')][1:] for l in rfile.readlines()[1:]]
			data.append( lines[1:])
	alldt = []
	for f1,f2,f3 in zip(data[0],data[1],data[2]):
		alldt.append(np.array(f1+f2+f3))
	return np.array(alldt)

# data file format : time, (accx, accy,accz, mx,my,mz)*3, marker

def read_data_file(f, keep_marker = False):
	rfile = open(f,'rU')
	lines = []
	if(keep_marker):
		lines = [ [np.float64(label.replace(',','.')) for label in l.rstrip().split()][1:] for l in rfile.readlines()[1:]]
	else:
		# remove marker
		lines = [ [np.float64(label.replace(',','.')) for label in l.rstrip().split()][1:-1] for l in rfile.readlines()[1:]]
	
	#print lines
	data = np.array(lines)
	return data
	
def segment(dt,wl,ol):
	n =len(dt)
	i = 0
	sdata = []
	while i < n: 
		fr = dt[i:i+wl]
		i += wl-ol
		sdata.append(fr)
	return np.array(sdata)
	
def calc_theta(data,deg=True):
	min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
	X_scaled =  min_max_scaler.fit_transform(data[:,(accZ1,accZ2,accZ3)])
	"""
	for z1,z2,z3 in X_scaled:
		if abs(z1)> 1 or abs(z2)> 1 or abs(z3)> 1:
			print  z1,z2,z3 
			print np.arccos([z1,z2,z3 ])
	"""
	X_scaled[X_scaled>1] = 1
	X_scaled[X_scaled<-1] = -1
	th = np.arccos(X_scaled)
	if deg:
		th = th*360/(2*np.pi)
	return th

# featlist contiene una lista della stessa dimensione del dataset
# ogni elemento e a sua volta una lista delle features che si vogliono aggiungere

def add_features(intdata,featlist):
	ndata = []
	for it,newf in zip(intdata,featlist):
		ndata.append( np.append(it,newf))
	return np.array(ndata)
	
def entropy(fl):
	entr = [0 for v in fl]
	
	i = 0
	for f in fl:
		c = Counter(f)
		ndistval = len(c)
		probs = [ float(count)/float(len(f)) for count in c.values()]
		e = - sum(probs * np.log2(probs))
		e/=len(f)
		entr[i] = e
		i+=1
	return entr
	
# calcola fft(frame), entr(fft(frame))
# x ogni frame calcola la somma dei quadrati del risultato diviso per la lunghezza del frame
# calcola anche information entropy
def frame_fft_and_entropy(dframe):
	rf = []
	enf = []
	#for dframe in dataset:
	f = []
	for tpl in var_to_fft:
		frame_slice = (dframe[:,tpl])
		ft = [ LA.norm(ff)**2 for ff in fft(frame_slice)]
		temprf = sum(ft)/len(ft)
		rf.append(temprf)
		f.append(ft)
	enf = np.array(entropy(f))
	return np.append(np.array(rf),enf)
	"""
	dfr1 = dframe[:,(accX1,accY1,accZ1)]
	dfr2 = dframe[:,(accX2,accY2,accZ2)]
	dfr3 = dframe[:,(accX3,accY3,accZ3)]
	f1 = [ LA.norm(ff)**2 for ff in fft(dfr1)]
	f2 = [ LA.norm(ff)**2 for ff in fft(dfr2)]
	f3 = [ LA.norm(ff)**2 for ff in fft(dfr3)]
	
	rf1 = sum(f1)/len(f1)
	rf2 = sum(f2)/len(f2)
	rf3 = sum(f3)/len(f3)
	rf = (np.append( np.append( rf1,rf2 ),rf3))
	enf = (np.array(entropy(f1,f2,f3)))
	return (rf, enf)
	"""
def frame_avg(frame):
	return np.mean(frame,axis=0,dtype=np.float64)
def frame_std(frame):
	return np.std(frame,axis=0,dtype=np.float64)

def frame_max(frame):
	return np.max(frame,axis=0)

def frame_min(frame):
	return np.min(frame,axis=0)

def frame_median(frame):
	return np.median(frame,axis=0)
	
def frame_MAD(frame):
	medians = np.median(frame,axis=0)
	i = 0
	MAD = []
	for attr in medians:
		var = frame[:,i]
		MAD.append( np.median(np.abs(var-attr)))
		i+=1
	return np.array(MAD)

def frame_cor(frame):
	corr_list = []
	for (var1,var2) in var_to_cor:
		corr_list.append(np.correlate(frame[:,var1],frame[:,var2]))
	return np.array(corr_list)

def frame_sma(frame):
	sma_list = []
	for axes in var_to_sma:
		T = len(frame)
		axes_sum = 0
		for var in axes:
			axes_sum+=sum(abs(frame[:,var]))
		axes_sum/=T
		sma_list.append(axes_sum)
	return np.array(sma_list)

# header

original_header = [
'accX1','accY1','accZ1','mgX1','mgY1','mgZ1',
'accX2','accY2','accZ2','mgX2','mgY2','mgZ2',
'accX3','accY3','accZ3','mgX3','mgY3','mgZ3',
'th1','th2','th3',
]

# defines for which variables the correlation will be computed

var_to_cor = [
(accX1,accY1),
(accX2,accY2),
(accX3,accY3),
(accX1,accZ1),
(accX2,accZ2),
(accX3,accZ3),
(accY1,accZ1),
(accY2,accZ2),
(accY3,accZ3),
(mgX1,mgY1),
(mgX2,mgY2),
(mgX3,mgY3),
(mgX1,mgZ1),
(mgX2,mgZ2),
(mgX3,mgZ3),
(mgY1,mgZ1),
(mgY2,mgZ2),
(mgY3,mgZ3),
]
# defines for which sets of axes the SMA will be computed

var_to_sma = [
(accX1,accY1,accZ1),
(accX2,accY2,accZ2),
(accX3,accY3,accZ3),
]

# defines for which sets of variables the fft & fft entropy will be computed

var_to_fft = [
(accX1,accY1,accZ1),
(accX2,accY2,accZ2),
(accX3,accY3,accZ3),
(mgX1,mgY1,mgZ1),
(mgX2,mgY2,mgZ2),
(mgX3,mgY3,mgZ3),
]

# defines which features will be constructed from the data

feature_list = [
 ('mean',frame_avg),
 ('stdev',frame_std),
 ('median',frame_median),
 ('min',frame_min),
 ('max',frame_max),
 ('mad',frame_MAD),
 ('sma',frame_sma),
 ('cor',frame_cor),
 ('fft_entr',frame_fft_and_entropy),
]

new_labels = [
'_avg',
'_stdev',
'_median',
'_min',
'_max',
'_mad',
'_sma',
'_cor',
'_fft',
'_entr',
]

header = []

label_index = {

}
# reads newlabels list to create header
def create_header():
	idx = 0
	for label in new_labels:
		if label in ('_avg','_stdev','_median','_min','_max','_mad'):
			for  var in original_header:
				header.append((var+label))
				label_index[(var+label)]=idx
				idx +=1 
		elif label in ('_cor'):
			for (var1,var2) in var_to_cor:
				header.append(original_header[var1]+"_"+original_header[var2]+label)
				
				label_index[original_header[var1]+"_"+original_header[var2]+label]=idx
				idx +=1 
		elif label in ('_sma'):
			for (var1,var2,var3) in var_to_sma:
				l =original_header[var1]+"_"+original_header[var2]+"_"+original_header[var3]+label
				header.append(l)
				label_index[l]=idx
				idx +=1 
		elif label in ('_fft','_entr'):
			for tpl in var_to_fft:
				s = ""
				for t in tpl:
					s+=(original_header[t]+"_")
				s=s[:-1]
				header.append(s+label)
				label_index[s+label] = idx
				idx +=1 
		
	return header
	
	
	
	
	
	
	
	
	
