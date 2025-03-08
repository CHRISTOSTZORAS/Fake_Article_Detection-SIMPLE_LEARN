import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection  import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import re
import string

#data cleaning
fake_news=pd.read_csv('/workspaces/Fake_Article_Detection/datasets/Fake.csv')
true_news=pd.read_csv('/workspaces/Fake_Article_Detection/datasets/True.csv')
fake_news['class']=0
true_news['class']=1

#manual_testing
data_fake_manual_testing=fake_news.tail(10)
for i in range(23480,23470,-1):
    fake_news.drop([i],axis=0,inplace=True)
    
data_true_manual_testing=true_news.tail(10)
for i in range(21416,21406,-1):
    true_news.drop([i],axis=0,inplace=True)
    
data_fake_manual_testing['class']=0
data_true_manual_testing['class']=1


whole_data=pd.concat([fake_news,true_news], axis=0)
reduced_data=whole_data.drop(['title','subject','date'],axis=1)


#function to remove marks
def wordopt(text):
    text=text.lower()
    text=re.sub('\[.*?\]','',text)
    text=re.sub("\W","",text)
    #πρεπει να αλλαξω την απο πανω εντολη να αφαιρει μονο τα κενα η γενικα να δω γιατι αφαιρει ολες τις γραμμες
    text=re.sub('https?://\S+|www\.\S+','',text)
    text=re.sub('<.*?>+','',text)
    text=re.sub('[%s]' % re.escape(string.punctuation),'',text)
    text=re.sub('\n','',text)
    text=re.sub('\w*\d\w*','',text)
    return text

reduced_data['text']=reduced_data['text'].apply(wordopt)

#define variabels
x=reduced_data['text']
y=reduced_data['class']

#train and split data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
vectorization= TfidfVectorizer()
xv_train=vectorization.fit_transform(x_train)
xv_test=vectorization.transform(x_test)

#model logistic regression defining
logistic_regression=LogisticRegression()
logistic_regression.fit(xv_train,y_train)

prediction_logistic_regression=logistic_regression.predict(xv_test)
score_logistic_Regression=logistic_regression.score(xv_test,y_test)

print(prediction_logistic_regression);print(score_logistic_Regression)
print(classification_report(y_test,prediction_logistic_regression))    

#model decision tree defining
DT=DecisionTreeClassifier()
DT.fit(xv_train,y_train)
pred_dt=DT.predict(xv_test)
DT.score(xv_test,y_test)


