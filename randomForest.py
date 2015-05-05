import numpy as np
import nltk
import string
import pickle
from scipy.sparse import csc_matrix
from sklearn import ensemble
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
def combineText(infile):
    exclude = set(string.punctuation + "0123456789")
    f = open("stop-word-list.csv", "r")
    stop_words = f.read().split(",")
    f.close()
    stop_words = [i.strip() for i in stop_words]
    #save training data as a list of lists of strings
    #save labels as a list of integers
    data = []
    labels = []
    #save the corpus in a list of strings
    fo = open(infile, 'r')
    for line in fo:
        text,label = line.strip().split('\t')
        text = ''.join(ch for ch in text if ch not in exclude)
        data.append(' '.join([w for w in text.split() if w not in stop_words]))
        labels.append(int(label))
    fo.close()
    return data, labels
def tfidf(train_corpus, val_corpus, train_label, val_label):
    vectorizer = TfidfVectorizer(min_df=20)
    print 'TF-IDF training data'
    X_train = vectorizer.fit_transform(train_corpus)
    X_train.dtype = np.float16
    X_train = X_train.todense()
    y_train = train_label
    print 'TF-IDF validation data'
    X_val = vectorizer.transform(val_corpus)
    X_val.dtype = np.float16
    X_val = X_val.todense()
    y_val = val_label
    return X_train,y_train,X_val, y_val
def randomForest(X_train, y_train, X_val, y_val, depth, n):
    print 'defining classifier'
    clf = ensemble.RandomForestClassifier(n_estimators = n, criterion = 'entropy', max_depth = depth)
    print 'traiing classifier'
    clf.fit(X_train, y_train)
    print 'predicting values'
    pred_train = clf.predict(X_train)
    pred_val = clf.predict(X_val)
    print 'calculating error'
    err_train = np.sum(pred_train != y_train)/float(len(y_train))
    err_val = np.sum(pred_val != y_val)/float(len(y_val))
    return err_train, err_val

if __name__ == '__main__':
    print 'processing train data'
    train_corpus, train_labels = combineText('100000.txt')
    print 'processing validation data'
    val_corpus, val_labels =combineText('val_20000.txt')
    X_train,y_train,X_val, y_val = tfidf(train_corpus, val_corpus, train_labels, val_labels)
    n_estimators = [200]
    for n in n_estimators:
        print 'n_estimators = ', n
        print randomForest(X_train, y_train, X_val, y_val, 60, n)
