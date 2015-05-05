import numpy as np
import nltk
import string
import pickle
from scipy.sparse import csc_matrix
from sklearn.externals import joblib
from sklearn import ensemble
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
def combineText(infile):
    exclude = set(string.punctuation + "0123456789")
    #f = open("stop-word-list.csv", "r")
    #stop_words = f.read().split(",")
    #f.close()
    #stop_words = [i.strip() for i in stop_words]
    #save training data as a list of lists of strings
    #save labels as a list of integers
    data = []
    labels = []
    #save the corpus in a list of strings
    fo = open(infile, 'r')
    for line in fo:
        text,label = line.strip().split('\t')
        text = ''.join(ch for ch in text if ch not in exclude)
        data.append(text)
        #data.append(' '.join([w for w in text.split() if w not in stop_words]))
        labels.append(int(label))
    fo.close()
    return data, labels
def tfidf(train_corpus, val_corpus, train_label, val_label):
    vectorizer = TfidfVectorizer(min_df=1,stop_words='english')
    print 'TF-IDF training data'
    X_train = vectorizer.fit_transform(train_corpus)
    y_train = train_label
    print "dumping vectorizer"
    joblib.dump(vectorizer, './pickle/vectorizer.pkl')
    print 'TF-IDF validation data'
    X_val = vectorizer.transform(val_corpus)
    y_val = val_label
    return X_train,y_train,X_val, y_val
def svm(X_train, y_train, X_val, y_val, c):
    print 'defining classifier'
    clf = LinearSVC(C = c)
    print 'traiing classifier'
    clf.fit(X_train, y_train)
    print 'predicting values'
    pred_train = clf.predict(X_train)
    pred_val = clf.predict(X_val)
    print 'calculating error'
    err_train = np.sum(pred_train != y_train)/float(len(y_train))
    err_val = np.sum(pred_val != y_val)/float(len(y_val))
    print 'dumping model to pickle'
    joblib.dump(clf, './pickle/clf.pkl')
    return err_train, err_val

if __name__ == '__main__':
    print 'processing train data'
    train_corpus, train_labels = combineText('train_data.txt')
    print 'processing validation data'
    val_corpus, val_labels =combineText('val_data.txt')
    X_train,y_train,X_val, y_val = tfidf(train_corpus, val_corpus, train_labels, val_labels)
    '''
    cs = np.linspace(0.1, 2, 10)
    for c in cs:
        print c
        print svm(X_train, y_train, X_val, y_val,c)
    '''
    print svm(X_train, y_train, X_val, y_val, 0.064)
