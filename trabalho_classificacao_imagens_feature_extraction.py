# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:32:09 2019

@author: jean
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 13:20:33 2019

@author: jean
"""
import numpy as np
import os 
import cv2
import re

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

path = "/home/jean/git/ia09/trabalho_visao_computacional/Treinamento/"
path_test = "/home/jean/git/ia09/trabalho_visao_computacional/Teste/"

hists_train = []
hists_cls_train = []
hists_test = []
hists_cls_test = []

files_train = os.listdir(path)
files_test = os.listdir(path_test)

for f in files_train:    
    name = f.split('.')[0]    
    cls = re.sub(r'[0-9]+','',name)
    
    image = cv2.imread(path+f)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])        
    cv2.normalize(hist, hist)        
    h = hist.flatten()    

    hists_train.append( h )
    hists_cls_train.append( cls )
    

for f in files_test:    
    name = f.split('.')[0]    
    cls = re.sub(r'[0-9]+','',name)
    
    image = cv2.imread(path_test+f)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])        
    cv2.normalize(hist, hist)        
    h = hist.flatten()    

    hists_test.append( h )    
    hists_cls_test.append( cls )



classifier_svm = SVC()
classifier_svm.fit(hists_train,hists_cls_train)
predicted_cls = classifier_svm.predict(hists_test)

print("acuracia svm: " )
print(metrics.accuracy_score( hists_cls_test , predicted_cls ) )

print("imagens classificadas incorretamente ")
print( [ files_test[indx] for indx,x in enumerate(predicted_cls) for indy,y in enumerate(hists_cls_test) if indx==indy and x != y] )

classifier_rf = RandomForestClassifier()
classifier_rf.fit(hists_train,hists_cls_train)
predicted_cls = classifier_rf.predict(hists_test)

print("acuracia ramdon forest: " )
print(metrics.accuracy_score( hists_cls_test , predicted_cls ) )

print("imagens classificadas incorretamente")
print( [ files_test[indx] for indx,x in enumerate(predicted_cls) for indy,y in enumerate(hists_cls_test) if indx==indy and x != y] )

    