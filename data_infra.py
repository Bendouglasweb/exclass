from sklearn import linear_model, svm, tree
import sklearn.metrics
import numpy as np

SVM_LINEAR_MODEL = 'SVM_LINEAR'
EVALUATION_TYPE = 'ACCURACY'
FILE = '../data/1CSurr.csv'
TRAIN_RATIO = 0.25
import random


# reads from file and returns X, Y
def ReadFromFile(filename,shuffle=None):
    with open(filename) as f:
        data = np.loadtxt(f, delimiter=",")
    if shuffle is not None:
        np.random.shuffle(data)
    X = np.array((data[:, 0:-1]))
    X.tolist()
    Y = np.array(data[:, -1])
    return X, Y


def SplitTrainAndTest(train_ratio, X, Y):
    train_size = int(len(Y) * train_ratio)
    X_train = X[:train_size]
    Y_train = Y[:train_size]
    X_test = X[train_size:]
    Y_test = Y[train_size:]
    return [X_train, Y_train, X_test, Y_test]


def TrainModel(X, Y, model_type='SVM_LINEAR',op=0,n_features=0):
    if model_type == SVM_LINEAR_MODEL:
        if op == 0:
            model = svm.SVC(kernel='linear')
        elif op == 1:
            model = svm.LinearSVC()
        elif op == 2:
            model = svm.SVC(kernel='rbf')
        elif op == 3:
            model = svm.SVC(kernel='rbf',gamma=2/n_features)
        elif op == 4:
            model = svm.SVC(kernel='rbf',gamma=1/(2*n_features))
        elif op == 5:
            model = svm.SVC(kernel='poly',degree=2)
        elif op == 6:
            model = svm.SVC(kernel='poly',degree=3)
        elif op == 7:
            model = svm.SVC(kernel='poly',degree=5)
        elif op == 8:
            model = svm.SVC(kernel='poly',degree=6)

    #elif model_type=='SVM_NONLINEAR':
	#    model=#add function for non linear
    model.fit(X, Y)

    return model


def ComputePerf(Y_actual, Y_pred):
    conf_matrix = sklearn.metrics.confusion_matrix(Y_actual, Y_pred)
    if EVALUATION_TYPE == 'ACCURACY':
        metric = sklearn.metrics.accuracy_score(Y_actual, Y_pred)
    return {'metric': metric, 'conf_matrix': conf_matrix}


def PredictModel(model, X_test, Y_test=None):
    Y_pred = model.predict(X_test)
    return Y_pred

def TestPipeline():
    X, Y = ReadFromFile(FILE)
    [X_train, Y_train, X_test, Y_test] = SplitTrainAndTest(TRAIN_RATIO, X, Y)
    model = TrainModel(X_train, Y_train)

    performance = ComputePerf(PredictModel(model, X_train), Y_train)
    print('Train Performance on %s: %s' % (EVALUATION_TYPE, performance['metric']))

    performance = ComputePerf(PredictModel(model, X_test), Y_test)
    print('Test Performance on %s: %s' % (EVALUATION_TYPE,performance['metric']))

def WriteToFile(filename,X,Y=None):
    Y=[int(y) for y in Y]
    X=np.concatenate((X,np.array([list(Y)]).T),axis=1)

    with open(filename,'w') as f:
        data = np.savetxt(f,X,delimiter=",")


# Added for clustering portion
# ToDo

if __name__ == '__main__':
    TestPipeline()
