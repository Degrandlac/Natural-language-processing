import pandas as pd


message = pd.read_csv('/Users/macbookair/Desktop/datasets/smsspamcollection/SMSSpamCollection',sep='\t',names=['labels','messages'])


import re


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lem = WordNetLemmatizer()
lm = []
for i in range(0, len(message)):
    review = re.sub('[^a-zA-Z]', ' ', message['messages'][i])
    review = review.lower()
    review = review.split()
    review = [lem.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    lm.append(review)

from sklearn.feature_extraction.text import CountVectorizer

cv =CountVectorizer(max_features=7000)
x = cv.fit_transform(lm).toarray()
y = pd.get_dummies(message['labels'])
y = y.iloc[:,1].values

from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,accuracy_score


X_train ,X_test,y_train,y_test =train_test_split(x,y,test_size =0.30,random_state = 0)

spam_detect_model = MultinomialNB().fit(X_train,y_train)

y_pred = spam_detect_model.predict(X_test)

confunsion_m = confusion_matrix(y_test, y_pred)
accuracy =accuracy_score(y_test, y_pred)
