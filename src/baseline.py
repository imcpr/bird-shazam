import numpy as np 
import csv
import os
from scipy import io
from random import randint
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

def load_test_files():
    files = {}
    for i in range(1, 1001):
        fname = "../data/test/cepst_conc_cepst_nips4b_birds_testfile"
        if i < 10:
            fname+= "000"
        elif i<100:
            fname+="00"
        elif i<1000:
            fname+="0"
        fname+= "%d.txt" % i
        f= open(fname, 'r')
        lines = f.readlines()
        x = []
        for line in lines:
            tok = line.split()
            tok = [float(j) for j in tok]
            x.append(tok)
        x = np.array(x)
        files[i] = x
    return files


# returns a list of size 687 with each position holding the matrix from file i
def load_train_files():
    files = {}
    for i in range(1, 688):
        fname = "../data/train/cepst_conc_cepst_nips4b_birds_trainfile"
        if i < 10:
            fname+= "00"
        elif i<100:
            fname+="0"
        fname+= "%d.txt" % i
        f= open(fname, 'r')
        lines = f.readlines()
        x = []
        for line in lines:
            tok = line.split()
            tok = [float(j) for j in tok]
            x.append(tok)
        x = np.array(x)
        files[i] = x
    return files

# loads multilabel matrix from numero_file_train.csv
def load_labels():
    f = open('../data/numero_file_train.csv')
    labels = {}
    lines = f.readlines()
    for line in lines:
        tok = line.split(",")
        tok = [float(i) for i in tok]
        labels[int(tok[0])] = tok[1:-1] # last col is duration, not using for now
    return labels

def vector_mean(m):
    return np.sum(m, axis=1)/1288

def create_lgr_dataset(files, labels):
    X = []
    Y = []
    c = 0
    for i in range(1, 500):
        if i == 60: # skip file 60 cuz there is some infinity numbers
            continue
        for j in range(len(labels[i])):
            if labels[i][j] ==1:
                X.append(vector_mean(files[i]))
                Y.append(j+1)
                c += 1
    return np.array(X), np.array(Y)

# takes the 17x1288 matrix and averages out column vectors to get 1x17 avg vector
def transform_X(files, start, end, skip=True):
    X = []
    for i in range(start, end):
        if i == 60 and skip:
            continue
        X.append(np.sum(files[i], axis=1)/1288)
    return np.array(X)

def write_submission(filename, preds):
    f = open(filename, 'w')
    for i in range(1,1001):
        fname = "nips4b_birds_testfile"
        if i < 10:
            fname+= "000"
        elif i<100:
            fname+="00"
        elif i<1000:
            fname+="0"
        fname += "%d.wav_classnumber_" %i
        for j in range(1, 88):
            filename = fname + "%d" % j
            line = "%s,%f"%(filename,preds[i-1][j-1])
            f.write(line+"\n")
    f.close()

print "Loading dataset"
files = load_train_files()
labels = load_labels()
print "Formatting data"
# formats dataset in lgr format
lgrX, lgrY = create_lgr_dataset(files, labels)

lgr = LogisticRegression()
print "Fitting LogisticRegression Classifier"
lgr = lgr.fit(lgrX, lgrY)

# formats data for decision tree format
X_train = transform_X(files, 1, 500)
Y_train = []
for i in range(1, 500):
    if i == 60:
        continue
    Y_train.append(labels[i])
Y_train = np.array(Y_train)
dt = DecisionTreeClassifier()
dt = dt.fit(X_train, Y_train)

# transform validation set
X_val = transform_X(files, 500, 688)
Y_val = []
for i in range(500, 688):
    Y_test.append(labels[i])
Y_val = np.array(Y_val)

preds = lgr.predict_proba(X_val)

print "LogisticRegression on test: "
print "ROC AUC %f " % metrics.roc_auc_score(Y_val, preds)

preds2= dt.predict(X_test)

print "DecisionTreeClassifier on test: "
print "ROC AUC %f " % metrics.roc_auc_score(Y_val, preds2)


# read actual test files and transform into correct format
test_files = load_test_files()
X_sub = transform_X(test_files, 1, 1001, False)
preds = lgr.predict_proba(X_sub)
preds2 = dt.predict(X_sub)

# write to files
# write_submission('lgr_sub', preds)
# write_submission('dt_sub', preds2)