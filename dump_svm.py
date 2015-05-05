import numpy as np
import nltk
import string
from sklearn.externals import joblib
from sklearn import ensemble
from sklearn.svm import LinearSVC
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
def tfidf_forest_pickle(train_corpus, train_label):
    print 'defining TF-IDF vectorizer'
    vectorizer = TfidfVectorizer(min_df=1)
    print 'TF-IDF on training set'
    X_train = vectorizer.fit_transform(train_corpus).todense()
    y_train = [int(i) for i in train_label]
    print 'defining random forest classifier, n = 20, depth = 10'
    clf = LinearSVC(C = 0.064)
    print 'training classifier'
    clf.fit(X_train, y_train)
    print 'dumping model to pickle'
    joblib.dump(clf, './testPickle/clf.pkl') 
    joblib.dump(vectorizer, './testPickle/vectorizer.pkl') 
if __name__ == '__main__':
    print 'processing 10000 training data'
    train_corpus, train_labels = combineText('small.txt')
    tfidf_forest_pickle(train_corpus, train_labels)
