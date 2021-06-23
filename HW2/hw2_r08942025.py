import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn import ensemble, metrics
from sklearn.tree import DecisionTreeClassifier

def create_model(neurons=80):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(units=neurons, input_dim=30, kernel_initializer='normal', activation='relu')) 
    model.add(keras.layers.Dense(units=1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
    return model

def gen_file():

	df = pd.read_csv('Data.csv')
	target_var = 'Class'
	features = list(df.columns)
	features.remove(target_var)

	train_x, test_x, train_y, test_y = train_test_split(df[features], df[target_var], train_size=0.8, test_size=0.2, random_state=7)

	return train_x, test_x, train_y, test_y

def grid_search(train_x, test_x, train_y, test_y):

	model = KerasClassifier(build_fn=create_model, verbose=0)
	b_size = [5,10]
	max_epochs = [20,50,100]
	neurons_select = [60, 70, 80, 90]
	param_grid = dict(batch_size=b_size, nb_epoch=max_epochs, neurons = neurons_select)
	grid = GridSearchCV(estimator=model, param_grid=param_grid) 
	print("Starting training")
	grid_result = grid.fit(train_x, train_y) 
	print("Training finished")

	# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)) 

	# display(pd.DataFrame(grid_result.cv_results_).sort_values(by='rank_test_score').head(5))

	# print(grid_result.cv_results_)

	# for params, mean_test_score in grid_result.cv_results_: 

	#     print(mean_test_score, params)

	print(f"best accuracy: {grid_result.best_score_}，best parameter set：{grid_result.best_params_}")
	
	means = grid_result.cv_results_['mean_test_score']
	stds = grid_result.cv_results_['std_test_score']
	params = grid_result.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):
	    print(f"average accuracy: {mean}, standard diviation: {stdev}, parameter set: {param}")

	return grid_result.best_score_, grid_result.best_params_



def model_evaluation(model, epochs_p, b_size, train_x, test_x, train_y, test_y):

	train_history = model.fit(x=train_x, y=train_y, validation_split=0.2, epochs=epochs_p, batch_size=b_size, verbose=2)
	pred_y = model.predict(test_x)
	accuracy = metrics.accuracy_score(test_y, pred_y)
	# print(accuracy)
	# summarize history for accuracy
	plt.plot(train_history.history['accuracy'])
	plt.plot(train_history.history['val_accuracy'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left') 
	plt.show()
	# summarize history for loss 
	plt.plot(train_history.history['loss']) 
	plt.plot(train_history.history['val_loss']) 
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left') 
	plt.show()
	return accuracy

def plot_confusion_matrix(model, train_x, test_x, train_y, test_y):

	pred_y = model.predict(test_x)
	confmat_test = confusion_matrix(test_y, pred_y)

	fig, ax = plt.subplots(figsize=(2.5, 2.5))
	ax.matshow(confmat_test, cmap=plt.cm.Blues, alpha=0.3)
	for i in range(confmat_test.shape[0]):
	    for j in range(confmat_test.shape[1]):
	        ax.text(x=j, y=i, s=confmat_test[i,j], va='center', ha='center')
	plt.xlabel('predicted label')        
	plt.ylabel('true label')
	plt.show()

	pred_y_train = model.predict(train_x)
	confmat_train = confusion_matrix(train_y, pred_y_train)

	fig, ax = plt.subplots(figsize=(2.5, 2.5))
	ax.matshow(confmat_train, cmap=plt.cm.Blues, alpha=0.3)
	for i in range(confmat_train.shape[0]):
	    for j in range(confmat_train.shape[1]):
	        ax.text(x=j, y=i, s=confmat_train[i,j], va='center', ha='center')
	plt.xlabel('predicted label')        
	plt.ylabel('true label')
	plt.show()

	return confmat_test, confmat_train

def conf_statistics_0(confmat_test):

	#class_0
	TP = confmat_test[0][0]
	FP = confmat_test[1][0]
	FN = confmat_test[0][1]
	TN = confmat_test[1][1]

	precision = TP/(TP+FP)
	recall = TP/(TP+FN)
	F1_score = 2 * precision * recall / (precision + recall)
	# print(precision, recall, F1_score)
	return precision, recall, F1_score

def conf_statistics_1(confmat_test):
	#class_1
	TN = confmat_test[0][0]
	FN = confmat_test[1][0]
	FP = confmat_test[0][1]
	TP = confmat_test[1][1]

	precision = TP/(TP+FP)
	recall = TP/(TP+FN)
	F1_score = 2 * precision * recall / (precision + recall)
	# print(precision, recall, F1_score)
	return precision, recall, F1_score

def plot_roc_curve(model, test_x, test_y):

	ns_probs = [0 for _ in range(len(test_y))]
	# predict probabilities
	lr_probs = model.predict_proba(test_x)
	# keep probabilities for the positive outcome only
	lr_probs = lr_probs[:, 1]
	# calculate scores
	ns_auc = metrics.roc_auc_score(test_y, ns_probs)
	lr_auc = metrics.roc_auc_score(test_y, lr_probs)
	# summarize scores
	print('No Skill: ROC AUC=%.3f' % (ns_auc))
	print('Logistic: ROC AUC=%.3f' % (lr_auc))
	# calculate roc curves
	ns_fpr, ns_tpr, _ = metrics.roc_curve(test_y, ns_probs)
	lr_fpr, lr_tpr, _ = metrics.roc_curve(test_y, lr_probs)
	# plot the roc curve for the model
	plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
	plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
	# axis labels
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	# show the legend
	plt.legend()
	# show the plot
	plt.show()

def plot_prc_curve(model, test_x, test_y):

	lr_probs = model.predict_proba(test_x)
	# keep probabilities for the positive outcome only
	lr_probs = lr_probs[:, 1]
	# predict class values
	# yhat = model.predict(testX)
	lr_precision, lr_recall, _ = metrics.precision_recall_curve(test_y, lr_probs)
	lr_auc = metrics.auc(lr_recall, lr_precision)
	# summarize scores
	print('Logistic: auc=%.3f' % lr_auc)
	# plot the precision-recall curves
	no_skill = len(test_y[test_y==1]) / len(test_y)
	plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
	plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')
	# axis labels
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	# show the legend
	plt.legend()
	# show the plot
	plt.show()

def plot_lift_curve(model, test_x, test_y):

	lr_probs = model.predict_proba(test_x)
	lr_probs = lr_probs[:, 1]
	dict_s = {"score": lr_probs}
	cg = pd.DataFrame(dict_s)
	test_y_ri = test_y.reset_index()
	test_y_ri = test_y_ri['Class']
	cg['target'] = test_y_ri
	pred_y_test = model.predict(test_x)
	confmat = confusion_matrix(test_y, pred_y_test)
	print(cg)
	get_cum_gains(cg, 'score', 'target', 'title', confmat)


def get_cum_gains(df, score, target, title, cm):
    df1 = df[[score,target]].dropna()

    TP = cm[0][0]
    FN = cm[0][1]
    FP = cm[1][0]
    TN = cm[1][1]
    fpr, tpr, thresholds = metrics.roc_curve(df1[target], df1[score])
    # ppr=(tpr*df[target].sum()+fpr*(df[target].count()-df[target].sum()))/df[target].count()
    ppr=(tpr*df[target].sum()+fpr*(df[target].count()-df[target].sum()))
    print(ppr)
    print(tpr)

    ttt = [i*(TP+FN) for i in tpr]
    print(ttt)
    # plt.figure(figsize=(12,4))
    # plt.subplot(1,2,1)
    # metrics.recall_score

    plt.plot(ppr, ttt, label='')
    plt.plot([0,df[target].count()], [0,ttt[-1]])
    plt.plot([0, FN+TN,df[target].count()],[0, ttt[-1],ttt[-1]])
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    plt.grid(b=True, which='both', color='0.65',linestyle='-')
    plt.xlabel('%Population')
    plt.ylabel('%Target')
    plt.title(title+'Cumulative Gains Chart')
    plt.legend(loc="lower right")
    plt.show()

def main_dnn(train_x, test_x, train_y, test_y):

	print("dnn")
	# grid search
	print("-"*40)
	print("grid search")
	best_score, best_param = grid_search(train_x, test_x, train_y, test_y)
	print(best_score)
	print(best_param)
	print("-"*40)

	## model evaluation
	print("-"*40)
	print("model evaluation")
	# print(best_param['neurons'], best_param['nb_epoch'], best_param['batch_size'])
	
	model = KerasClassifier(build_fn=create_model, verbose=0)
	accuracy = model_evaluation(model, best_param['nb_epoch'], best_param['batch_size'], train_x, test_x, train_y, test_y)
	
	# model = KerasClassifier(build_fn=create_model, verbose=0)
	# accuracy = model_evaluation(model, 50, 5, train_x, test_x, train_y, test_y)
	print("-"*40)

	## confusion matrix
	print("-"*40)
	print("confusion matrix")
	confmat_test, confmat_train = plot_confusion_matrix(model, train_x, test_x, train_y, test_y)
	print("-"*40)

	## model statistics
	print("-"*40)
	print("model statistics")
	precision_0, recall_0, F1_score_0 = conf_statistics_0(confmat_test)
	precision_1, recall_1, F1_score_1 = conf_statistics_1(confmat_test)
	print("-"*40)

	## roc prc curve
	print("-"*40)
	print("roc prc curve")
	plot_roc_curve(model, test_x, test_y)
	plot_prc_curve(model, test_x, test_y)
	print("-"*40)

	## lift curve
	print("-"*40)
	print("lift curve")
	plot_lift_curve(model, test_x, test_y)
	print("-"*40)
	
	pass

def main_dt(train_x, test_x, train_y, test_y):

	print("decision tree")
	## model evaluation
	print("-"*40)
	print("model evaluation")

	clf = DecisionTreeClassifier()
	model = clf.fit(train_x, train_y)
	pred_y = model.predict(test_x)
	accuracy = metrics.accuracy_score(test_y, pred_y)
	print("-"*40)

	## confusion matrix
	print("-"*40)
	print("confusion matrix")
	confmat_test, confmat_train = plot_confusion_matrix(model, train_x, test_x, train_y, test_y)
	print("-"*40)

	## model statistics
	print("-"*40)
	print("model statistics")
	precision_0, recall_0, F1_score_0 = conf_statistics_0(confmat_test)
	precision_1, recall_1, F1_score_1 = conf_statistics_1(confmat_test)
	print("-"*40)

	## roc prc curve
	print("-"*40)
	print("roc prc curve")
	plot_roc_curve(model, test_x, test_y)
	plot_prc_curve(model, test_x, test_y)
	print("-"*40)

	## lift curve
	print("-"*40)
	print("lift curve")
	plot_lift_curve(model, test_x, test_y)
	print("-"*40)

	pass

def main_rf(train_x, test_x, train_y, test_y):

	print("random forest")
	## model evaluation
	print("-"*40)
	print("model evaluation")

	model = ensemble.RandomForestClassifier(n_estimators = 100)
	forest_fit = model.fit(train_x, train_y)
	pred_y = model.predict(test_x)
	accuracy = metrics.accuracy_score(test_y, pred_y)
	print("-"*40)

	## confusion matrix
	print("-"*40)
	print("confusion matrix")
	confmat_test, confmat_train = plot_confusion_matrix(model, train_x, test_x, train_y, test_y)
	print("-"*40)

	## model statistics
	print("-"*40)
	print("model statistics")
	precision_0, recall_0, F1_score_0 = conf_statistics_0(confmat_test)
	precision_1, recall_1, F1_score_1 = conf_statistics_1(confmat_test)
	print("-"*40)

	## roc prc curve
	print("-"*40)
	print("roc prc curve")
	plot_roc_curve(model, test_x, test_y)
	plot_prc_curve(model, test_x, test_y)
	print("-"*40)

	## lift curve
	print("-"*40)
	print("lift curve")
	plot_lift_curve(model, test_x, test_y)
	print("-"*40)
	pass

if __name__ == '__main__':

	run = "dnn"
	# run = "dt"
	# run = "rf"

	train_x, test_x, train_y, test_y = gen_file()

	if run == 'dnn':
		main_dnn(train_x, test_x, train_y, test_y)
	elif run == 'dt':
		main_dt(train_x, test_x, train_y, test_y)
	elif run == 'rf':
		main_rf(train_x, test_x, train_y, test_y)
	else:
		pass