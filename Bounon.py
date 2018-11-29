#encoding=utf-8
#author: DIALLO Bassoma
#email: sanediallo2003@yahoo.fr
from __future__ import division
from sklearn.feature_extraction.text import CountVectorizer
import os
import numpy as np

# STEP1 Load the classic4 dataset
fold = os.path.join('classic')

corpus = []
for file in os.listdir(fold):
   with open(os.path.join(fold, file)) as f:
        text = f.read()
        text = text.strip()
        text = text.replace('\n',' ')
        corpus.append(text)
        #print(text)

with open('terms.txt') as f:
    voc = f.readlines()
    voc = [v.replace('\n', '') for v in voc]
vectorizer = CountVectorizer(vocabulary=voc)
X = vectorizer.fit_transform(corpus).todense()
#print(vectorizer.get_feature_names())
#np.save('X.npy', X)
np.savetxt('X.txt', X, fmt="%d")

print(X.shape)

#  STEP2  list of lists aka tfidf_vectorized_list

with open('docbyterm.tfidf.txt') as f:
    txt = f.readlines()

data = []
for i in range(7095):
    data.append([])
for i in txt:
    i_list = i.split(' ')
    data[int(i_list[0])-1].append(int(i_list[1]))

with open('tfidf_vectorized_list.txt','w+') as f:
    for i in data:
        f.write(str(i)[1:-1])
        f.write('\n')
#print(data)

tfidf_vectorized_list = []
with open('tfidf_vectorized_list.txt') as f:
    for line in f:
        inner_list = [elt.strip() for elt in line.split(', ')]
        # in alternative, if you need to use the file content as numbers
        # inner_list = [int(elt.strip()) for elt in line.split(',')]
        tfidf_vectorized_list.append(inner_list)
tfidf_vectorized_list = np.array(tfidf_vectorized_list)
print(tfidf_vectorized_list)