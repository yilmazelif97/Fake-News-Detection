import nltk
import pandas as pd
import numpy as np
import csv
import pickle
import nltk as nt
import re

import seaborn as sns
import matplotlib.pyplot as plt
import visualizer as visualizer
from sklearn import metrics
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer, word_tokenize

#nt.download('punkt') # I installed first time for can use word tokinizer
#nt.download('stopwords') #Installed first time for can use stopwords library
#nltk.download('wordnet')

#Importing datataset
from sklearn.tree import DecisionTreeClassifier

dataset = pd.read_csv(r'C:\Users\BERA YILMAZ\Desktop\train.csv')


print(dataset.head()) #It shows first 5 data
print(dataset.shape) #It shows number of inputs and features

stop_words = stopwords.words('english')

lemmatizer = WordNetLemmatizer()

"""
for index, row in dataset.iterrows():
    filter_sentence = ''
    sentence = row['text']

    # Cleaning the sentence with regex
    sentence = re.sub(r'[^\w\s]', '', sentence, flags=re.I)

    # Tokenization
    words = nltk.word_tokenize(sentence)


   # Stopwords removal
    words = [w for w in words if not w in stop_words]

    # Lemmatization
    for words in words:
        filter_sentence = filter_sentence  + ' ' + str(lemmatizer.lemmatize(words)).lower()
    dataset.loc[index, 'text'] = filter_sentence

dataset = dataset[['text', 'label']]
"""




#print(dataset.isnull().any()) #check that

print(dataset.isna().sum())

#I have to take all data. Solve this problem!
dataset = dataset[500:1500]

#Converting 0 and 1 to mean of word
dataset.loc[(dataset['label'] == 1) , ['label']] = 'Fake'
dataset.loc[(dataset['label'] == 0) , ['label']] = 'Real'

x= dataset['text']
y=dataset['label']


z=dataset.iloc[:,:-1].values
print(z[0])



#Splitting data for train and test
x_train, x_test, y_train, y_test = train_test_split(x.values.astype('str'),y, test_size=0.2)

#Applying tfidf
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(x_train) #fit transform means do extra calculation and do tranformations
tdifd_test = tfidf_vectorizer.transform(x_test)

#Applying Passive Aggressive Classifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)  # fit is using for starts prediction in x_train
y_pred=pac.predict(tdifd_test)  #predict on set test and calculate accuracy
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')
print(classification_report(y_test,y_pred))

print(confusion_matrix(y_test,y_pred, labels=['True','Fake']))

c=confusion_matrix(y_test,y_pred)
TP, FP, FN, TN = c[0][0], c[0][1], c[1][0],c[1][1]

print('TP:', TP)
print('TN:', TN)
print('FP:', FP)
print('FN', FN)


print('########')

#Applying Naive-Bayes
pipeline= Pipeline([('tfidf',TfidfVectorizer(stop_words='english')),('nbmodel',MultinomialNB())])
pipeline.fit(x_train,y_train)
score2=pipeline.score(x_test,y_test)
print(f'Accuracy: {round(score2*100,2)}%')

predic = pipeline.predict(x_test)
print(classification_report(y_test,predic))

print(confusion_matrix(y_test,predic))

deneme = confusion_matrix(y_test,predic)
cr=classification_report(y_test,predic)

nn = metrics.confusion_matrix(y_test,predic)

c=confusion_matrix(y_test,predic)
TP, FP, FN, TN = c[0][0], c[0][1], c[1][0],c[1][1]

print('TP:', TP)
print('TN:', TN)
print('FP:', FP)
print('FN', FN)


print('########')

#print(dataset['title'][1552])



fig, ax = plt.subplots()
myscatterplot = ax.scatter(c, confusion_matrix(y_test,y_pred))
ax.set_xlabel("text")
ax.set_ylabel("label")

plt.show()



def plot_classification_report(cr, title='Classification report ', with_avg_total=False, cmap=plt.cm.Blues):

    lines = cr.split('\n')
    #print(lines) #classification reporstaki verileri ayrı ayrı yazdırıyor

    classes = []
    plotMat = []
    for line in cr[2 :len(lines)-2 ]:
        #print(line)
        t = line.split()
        #print(t)
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        print(v) #yazdırmak istediğim değerler
        plotMat.append(v)

    if with_avg_total:
        aveTotal = lines[len(lines) - 1].split()
        classes.append('avg/total')
        vAveTotal = [float(x) for x in t[1:len(aveTotal) - 1]]
        plotMat.append(vAveTotal)


    plt.imshow(plotMat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(3)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('Classes')
    plt.xlabel('Measures')

#plot_classification_report(deneme)




"""

clf = MultinomialNB()
clf.fit(tfidf_train,y_train)
pred= clf.predict(tdifd_test)
score3=metrics.accuracy_score(y_test,predic)
print(f'Accuracy: {round(score3*100,2)}%')
cm =metrics.confusion_matrix(y_test,predic)
print(cm)"""








