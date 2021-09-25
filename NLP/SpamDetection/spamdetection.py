# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 14:40:11 2021

@author: Anshul
"""
import pandas as pd
data = pd.read_csv('smsspamcollection/SMSSpamCollection', sep = '\t', names = ["label", "message"])


import re # Regular Expression
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


lemmatizer = WordNetLemmatizer()

#cleaning text and preprocessing
corpus=[]
for i in range(len(data)):
    review = re.sub('[^a-zA-Z]',' ',data['message'][i])
    review = review.lower() # Lowering the text
    review = review.split() # splitting sentences to words
    review = [lemmatizer.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review) # joining the words into sentences
    corpus.append(review)
    
# Creating TFIDF Model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features=(4000)) #Used to reduce column size of corpus
X = cv.fit_transform(corpus).toarray() #make matrix according to their frequencies in a sentence.


y = pd.get_dummies(data['label']) # It changes the ham and spam into 0 or 1(categorial input). Have two columns
y = y.iloc[:,1].values #Converted 2 columns into only 1 where 0-> ham and 1-> Spam


from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X,y,test_size=0.2,random_state=0)


from sklearn.naive_bayes import MultinomialNB
spam_detection = MultinomialNB().fit(train_x,train_y)

y_pred = spam_detection.predict(test_x)

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(test_y, y_pred)


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_y,y_pred)

print("Test Accuracy of Model : ", accuracy)