import numpy as np
import nltk
import math
import string
import pickle

def combineText(infile):
    exclude = set(string.punctuation + "0123456789")
    f = open("stop-word-list.csv", "r")
    stop_words = f.read().split(",")
    f.close()
    stop_words = [i.strip() for i in stop_words]
    '''
    #save training data as a list of lists of strings
    #save labels as a list of integers
    data = []
    labels = []
    '''
    #save the corpus in a list of strings
    corpus = []
    fo = open(infile, 'r')
    for line in fo:
        text,label = line.strip().split('\t')
        text = ''.join(ch for ch in text if ch not in exclude)
        corpus.append(text)
        #data.append([w for w in text.split() if w not in stop_words])
        #labels.append(label)
    fo.close()
    corpus = ' '.join(corpus)
    corpus = ' '.join([word for word in corpus.split() if word not in stop_words])
    '''
    #dump data into pickle
    output = open('corpus.pkl', 'wd')
    pickle.dump(corpus, output)
    output.close()

    data_pickle = open('train_data.pkl', 'wd')
    pickle.dump(data, data_pickle)
    data_pickle.close()
    
    label_pickle = open('train_label.pkl', 'wd')
    pickle.dump(labels, label_pickle)
    label_pickle.close()
    '''
    out = open('glove_yelp.txt', 'w')
    out.write(corpus)
    out.close()
if __name__ == '__main__':
    combineText('train_data.txt')
