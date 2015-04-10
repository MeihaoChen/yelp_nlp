import numpy as np
import nltk
import string
import pickle
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
        labels.append(label)
    fo.close()
    return data, labels
def tfidf(train_corpus, val_corpus, train_label, val_label):
    vectorizer = TfidfVectorizer(min_df=1)
    X_train = vectorizer.fit_transform(train_corpus).todense()
    y_train = train_label
    X_val = vectorizer.transform(val_corpus).todense()
    y_val = val_label
    clf = ensemble.RandomForestClassifier(n_estimators = 300, criterion = 'entropy', max_depth = 5)
    clf.fit(X_train, y_train)
    pred_train = clf.predict(X_train)
    pred_val = clf.predict(X_val)
    err_train = np.sum(pred_train != y_train)/float(len(y_train))
    err_val = np.sum(pred_val != y_val)/float(len(y_train))
    return err_train, err_val

if __name__ == '__main__':
    train_corpus, train_labels = combineText('small.txt')
    val_corpus, val_labels =combineText('val_small.txt')
    print tfidf(train_corpus, val_corpus, train_labels, val_labels)
