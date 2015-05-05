import string

def parseString(inputString):
    exclude = set(string.punctuation + "0123456789")
    inputStr = inputString
    cleanText = ''.join(ch for ch in inputStr if ch not in exclude)
    return cleanText
