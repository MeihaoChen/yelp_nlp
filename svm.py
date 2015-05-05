import numpy as np
import nltk
import string
import pickle
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

def combineText(infile):
    exclude = set(string.punctuation + "0123456789")
    #save data as a list of lists of strings
    #save labels as a list of integers
    data = []
    labels = []
    #save the corpus in a list of strings
    fo = open(infile, 'r')
    for line in fo:
        text,label = line.strip().split('\t')
        text = ''.join(ch for ch in text if ch not in exclude)
        data.append(text)
        labels.append(int(label))
    fo.close()
    return data, labels

def tfidf(train_corpus, val_corpus, train_label, val_label):
    # take the train and validation data
    # return the tfidf matrix and dump it into a pickle
    vectorizer = TfidfVectorizer(min_df=1,stop_words='english', analyzer = 'word', ngram_range = (1,2))
    print 'TF-IDF training data'
    X_train = vectorizer.fit_transform(train_corpus)
    y_train = train_label
    print 'dumping vectorizer'
    joblib.dump(vectorizer, './pickle_2gram/vectorizer.pkl')
    print 'TF-IDF validation data'
    X_val = vectorizer.transform(val_corpus)
    y_val = val_label
    return X_train,y_train,X_val, y_val

def svm(X_train, y_train, X_val, y_val, c):
    # define a classifier and train it against training set
    # dump the classifier into pickle
    print 'defining classifier'
    clf = LinearSVC(C = c)
    print 'training classifier'
    clf.fit(X_train, y_train)
    print 'dumping classifier'
    joblib.dump(clf, './pickle_2gram/clf_'+str(c)+'.pkl')
    print 'predicting values'
    pred_train = clf.predict(X_train)
    pred_val = clf.predict(X_val)
    print 'calculating error'
    err_train = np.sum(pred_train != y_train)/float(len(y_train))
    err_val = np.sum(pred_val != y_val)/float(len(y_val))
    return err_train, err_val

if __name__ == '__main__':
    print 'processing train data'
    train_corpus, train_labels = combineText('train_data.txt')
    print 'processing validation data'
    val_corpus, val_labels =combineText('val_data.txt')
    X_train,y_train,X_val, y_val = tfidf(train_corpus, val_corpus, train_labels, val_labels)
    #print 'loading vectorizer'
    #vectorizer = joblib.load('./pickle/vectorizer.pkl')
    #print "vectorizing data"
    #X_train = vectorizer.transform(train_corpus)
    #y_train = train_labels
    #X_val = vectorizer.transform(val_corpus) 
    #y_val = val_labels
    cs = np.linspace(0.01,0.3, 20)
    for c in cs:
        print c
        print svm(X_train, y_train, X_val, y_val,c)
