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


def process_image_class(image_name):
    name = image_name.split('.')[0]    
    return re.sub(r'[0-9]+','',name)
  
def process_image(path):
    image = cv2.imread(path)
    
#    image = cv2.resize(image,(400,300))
    
    hist  = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])        
    cv2.normalize(hist, hist)        
    
#    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    h = hist.flatten()    

    return h
    

hists_train = []
hists_cls_train = []

hists_test = []
hists_cls_test = []

files_train = os.listdir(path)
files_test = os.listdir(path_test)

for file in files_train:    
    hists_train.append( process_image(path+file) )
    hists_cls_train.append( process_image_class(file) )
    

for file in files_test:    
    hists_test.append( process_image(path_test+file) )    
    hists_cls_test.append( process_image_class(file) )



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

    