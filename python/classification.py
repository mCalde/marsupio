
import plotting as plt
from sklearn.decomposition import PCA,KernelPCA
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.externals import joblib
from sklearn import svm
	
import matplotlib.pyplot as pl
import numpy as np
import feature_man as fm
def cross_model_selection(X,Y,pars, _test_size=0.3,save = False):
	
	#evaluation set
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=_test_size, random_state=0)
	
	# Set the parameters by cross-validation (on x train only!)
	
	tuned_parameters = pars
	scores = ['accuracy']
	
	for score in scores:
		print("# Tuning hyper-parameters for %s" % score)
		print()
		scaler = preprocessing.StandardScaler()
		pca = PCA()
		clf = svm.SVC(C=1)
		pca_svm = Pipeline([('scaler',scaler),('pca', pca), ('clf',clf),])
		best_pipe = GridSearchCV(pca_svm, tuned_parameters, cv=5, scoring=score,verbose=0)
		best_pipe.fit(X_train, y_train)

		print("Best parameters set found on development set:")
		print()
		print(best_pipe.best_estimator_)
		print()
		print("Grid scores on development set:")
		print()
		for params, mean_score, scores in best_pipe.grid_scores_:
			print("%0.3f (+/-%0.03f) for %r"
				  % (mean_score, scores.std() / 2, params))
		print()

		print("Detailed classification report:")
		print()
		print("The model is trained on the full development set.")
		print("The scores are computed on the full evaluation set.")
		print()
		y_true, y_pred = y_test, best_pipe.predict(X_test)
		print("Confusion Matrix:\n")
		cm = (confusion_matrix(y_true, y_pred))
		print cm
		
		print()
		print ("Accuracy:\n")
		print accuracy_score(y_true,y_pred)
		print()
		print(classification_report(y_true, y_pred))
		print()
		plt.plot_confusion_matrix(cm)
# riallena su tutti i dati e salva su file
	if save:
		best_pipe.best_estimator_.fit(X,Y)
		joblib.dump(best_pipe.best_estimator_, 'model.pkl') 
		#scale_pipe = Pipeline(best_pipe.best_estimator_.steps[0:2])
		#X_scaled = scale_pipe.fit_transform(X)
		#plt.contour_plot(X_scaled,Y,best_pipe.best_estimator_)
	return

def tree_cross_model_selection(X,Y, _test_size=0.3,save = False):
	
	#evaluation set
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=_test_size, random_state=0)
	
	# Set the parameters by cross-validation (on x train only!)
	
	tuned_parameters = [{#'clf__criterion': ['gini','entropy'], 'clf__min_samples_split': [2,5,10],
				 'pca__n_components': [5]}
						]

	scores = ['accuracy']
	
	for score in scores:
		print("# Tuning hyper-parameters for %s" % score)
		print()
		scaler = preprocessing.StandardScaler()
		pca = PCA()
		clf =  tree.DecisionTreeClassifier()
		pca_tree = Pipeline([('scaler',scaler),('pca', pca), ('clf',clf),])
		best_pipe = GridSearchCV(pca_tree, tuned_parameters, cv=5, scoring=score,verbose=0)
		best_pipe.fit(X_train, y_train)

		print("Best parameters set found on development set:")
		print()
		print(best_pipe.best_estimator_)
		print()
		print("Grid scores on development set:")
		print()
		for params, mean_score, scores in best_pipe.grid_scores_:
			print("%0.3f (+/-%0.03f) for %r"
				  % (mean_score, scores.std() / 2, params))
		print()

		print("Detailed classification report:")
		print()
		print("The model is trained on the full development set.")
		print("The scores are computed on the full evaluation set.")
		print()
		y_true, y_pred = y_test, best_pipe.predict(X_test)
		print("Confusion Matrix:\n")
		print (confusion_matrix(y_true, y_pred))
		print(classification_report(y_true, y_pred))
		print()

# riallena su tutti i dati e salva su file
	if save:
		best_pipe.best_estimator_.fit(X,Y)
		joblib.dump(best_pipe.best_estimator_, 'model.pkl') 

	return
		

	
def load_model(f):
	clf2 = joblib.load(f)
	return clf2
