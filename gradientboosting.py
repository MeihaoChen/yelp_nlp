import numpy as np
import nltk
import string
import pickle
from scipy.sparse import csc_matrix
from sklearn import ensemble
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
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
        labels.append(label)
    fo.close()
    return data, labels
def tfidf(train_corpus, val_corpus, train_label, val_label, depth, n):
    vectorizer = TfidfVectorizer(min_df=10)
    print 'TF-IDF training data'
    X_train = vectorizer.fit_transform(train_corpus).todense()
    y_train = train_label
    print 'TF-IDF validation data'
    X_val = vectorizer.transform(val_corpus).todense()
    y_val = val_label
    print 'defining classifier'
    clf = GradientBoostingClassifier(n_estimators = n, learning_rate=1.0, max_depth = depth, random_state=0)
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
    train_corpus, train_labels = combineText('small.txt')
    print 'processing validation data'
    val_corpus, val_labels =combineText('val_small.txt')
    depths = [10, 20, 30, 40, 50]
    for d in depths:
        print 'depth = ', d
        print tfidf(train_corpus, val_corpus, train_labels, val_labels, d, 10)

