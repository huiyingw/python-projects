#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import logging
import sys
from time import time
import math
np.set_printoptions(threshold=np.inf)


class MyBayesClassifier():
    #implement Bernoulli Bayes
    def __init__(self, smooth=1):
        self._smooth = smooth # This is for add one smoothing, don't forget!
        self._feat_prob = []
        self._class_prob = []
        self._Ncls = []
        self._Nfeat = []

    def train(self, X, y):

        class_counts = []
        
        for theClass in y:
            if theClass not in self._Ncls:
                self._Ncls.append(theClass)
                class_counts.append(0) # class count starts from 0

        # P(y)
        self._class_prob = np.zeros(len(self._Ncls))
        for classNum, theClass in enumerate(self._Ncls):
            count = 0
            for thingsInY in y:
                if (thingsInY == theClass):
                    count += 1
            self._class_prob[classNum] = ((float(count) + self._smooth)/(len(y) + self._smooth*2))
            class_counts[classNum] = float(count)

        # P(xi=0|y)
        self._Nfeat = np.zeros((len(self._Ncls), len(X[0])));
        self._feat_prob = np.zeros((len(self._Ncls), len(X[0])));

        for classNum, theClass in enumerate(self._Ncls):
            for xNum, thingsInX in enumerate(X[0]):
                self._feat_prob[classNum][xNum] = (class_counts[classNum] + (self._smooth*2));
 
        for classNum, classValue in enumerate(self._Ncls): #looping over classes

            for yNum, thingsInY in enumerate(y): #looping over the documents classifications
                if (thingsInY == classValue):
                    #add up number of '1's in the Xs corresponding to that y index
                    for xNum, xi in enumerate(X[yNum]): #looping over a documents words
                        if(xi > 0):
                            self._Nfeat[classNum][xNum] += 1

        self._Nfeat = np.add(self._Nfeat, self._smooth)
        self._feat_prob = np.true_divide(self._Nfeat, self._feat_prob)
        return 


    def predict(self, X):
        # P(0 | X), P(1| X), P (2 | X)...
        ProbX = np.zeros((len(self._Ncls), len(X)))
        Xpercent = np.zeros((len(self._Ncls), len(X)))

        for classIndex, theClass in enumerate(self._Ncls):
            for xListNum, xList in enumerate(X): #looping over documents in document list
                ProbXi = math.log(self._class_prob[classIndex])
                for miu, thingsInX in enumerate(xList):   #looping over words in document
                    if(thingsInX > 0):
                        ProbXi += math.log(self._feat_prob[classIndex][miu]) #P(xi=1 | Class=theClass)
                    else: 
                        ProbXi += math.log(1 - self._feat_prob[classIndex][miu]) #P(xi=0 | Class=theClass)
                ProbX[classIndex][xListNum] = ProbXi

        predic = np.zeros(len(X))
        for xListNum, xList in enumerate(X):
            HighNum = ProbX[0][xListNum]
            HighNumClass = self._Ncls[0]
            for classIndex, theClass in enumerate(self._Ncls):
                NewNum = ProbX[classIndex][xListNum]
                if (NewNum > HighNum):
                    HighNum = NewNum
                    HighNumClass = theClass
            predic[xListNum] = HighNumClass

        return predic


""" 
Here is the calling code

"""

categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]
remove = ('headers', 'footers', 'quotes')

data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=remove)

data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=remove)
print('data loaded')

y_train, y_test = data_train.target, data_test.target

print("Extracting features from the training data using a count vectorizer")
t0 = time()

vectorizer = CountVectorizer(stop_words='english', binary=False)#, analyzer='char', ngram_range=(1,3))
X_train = vectorizer.fit_transform(data_train.data).toarray()
X_test = vectorizer.transform(data_test.data).toarray()
feature_names = vectorizer.get_feature_names()
"""

alpha = 1
clf = MyBayesClassifier(alpha)
clf.train(X_train,y_train)
y_pred = clf.predict(X_test)
print 'alpha=%i accuracy = %f' %(alpha, np.mean((y_test-y_pred)==0))
print 'Uncomment the code at the bottem to see the additive smooting'
"""
#additive smoothing:(uncomment the following to see the additive smoothing)
alpha = 0.01
while alpha < 1.01:
    clf = MyBayesClassifier(alpha)
    clf.train(X_train,y_train)
    y_pred = clf.predict(X_test)
    print '%f %f' %(alpha, np.mean((y_test-y_pred)==0))
    alpha += 0.01
    
