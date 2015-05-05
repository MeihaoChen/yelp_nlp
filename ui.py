import sys
from parseString import *
from sklearn.externals import joblib
def main():
    clf = joblib.load('./model/clf.pkl')
    vectorizer = joblib.load('./model/vectorizer.pkl')
    while True:
        try:
            inputNum = raw_input('Input the number of examples:')
            
            if inputNum == 'quit':
                raise KeyboardInterrupt
            num = int(inputNum)
            break
        except KeyboardInterrupt:
            print 'terminated!'
            sys.exit()
        except nullNumber:
            print 'Not an integer!'
    while True:
        try:
            stringList = []
            for i in range(num):
                while True:
                    inputString = raw_input()
                    if len(inputString) > 0:
                        break
                if len(inputString.strip().split('\n')) == num:
                    stringList = inputString.strip().split('\n')
                    break
                string = parseString(inputString)
                X = vectorizer.transform([string])
                pred = clf.predict(X)
                print pred[0]
            if len(stringList) > 0:
                for string in stringList:
                    string = parseString(string)
                    X = vectorizer.transform([string])
                    pred = clf.predict(X)
                    print pred[0]
                print "Prediction finished!"
                sys.exit()
            else:
                print "Prediction finished!"
                sys.exit()
        except KeyboardInterrupt:
            print 'terminated!'
            sys.exit()
if __name__ == '__main__':
    main()
