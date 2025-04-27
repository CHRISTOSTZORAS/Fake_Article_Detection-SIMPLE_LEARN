import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection  import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import re
import string

#data cleaning
fake_news=pd.read_csv('C:\\Users\\tzwrakos\\OneDrive\\Υπολογιστής\\Projects\\Fake_Article_Detection\\Simple Learn\\dataset\\Fake.csv')
true_news=pd.read_csv('C:\\Users\\tzwrakos\\OneDrive\\Υπολογιστής\\Projects\\Fake_Article_Detection\\Simple Learn\\dataset\\True.csv')
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

# Balance fake and real data
min_len = min(len(fake_news), len(true_news))
fake_news_balanced = fake_news.sample(n=min_len, random_state=42)
true_news_balanced = true_news.sample(n=min_len, random_state=42)

whole_data = pd.concat([fake_news_balanced, true_news_balanced], axis=0)
reduced_data = whole_data.drop(['title', 'subject', 'date'], axis=1)

whole_data=pd.concat([fake_news,true_news], axis=0)
reduced_data=whole_data.drop(['title','subject','date'],axis=1)


#function to remove marks
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

reduced_data['text'] = reduced_data['text'].apply(wordopt)

#define variabels
x=reduced_data['text']
y=reduced_data['class']

#train and split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
vectorization = TfidfVectorizer(ngram_range=(1,2), max_df=0.7, min_df=5)
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# #model logistic regression defining
logistic_regression = LogisticRegression(C=0.5, penalty='l2', max_iter=1000)
logistic_regression.fit(xv_train,y_train)

prediction_logistic_regression=logistic_regression.predict(xv_test)
score_logistic_Regression=logistic_regression.score(xv_test,y_test)

print(prediction_logistic_regression);print(score_logistic_Regression)
print(classification_report(y_test,prediction_logistic_regression))    

# #model decision tree defining
DT=DecisionTreeClassifier()
DT.fit(xv_train,y_train)

prediction_dt=DT.predict(xv_test)
dt_score=DT.score(xv_test,y_test)
print(prediction_dt);print(dt_score)
print(classification_report(y_test,prediction_dt))    


def output_label(n):
    if n==0 :
        return "Fake News"
    elif n==1:
        return "Not a Fake News"

def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test['text'].apply(wordopt)
    new_x_test = new_def_test['text']
    new_xv_test = vectorization.transform(new_x_test)
    pred_lr = logistic_regression.predict(new_xv_test)
    pred_dt = DT.predict(new_xv_test)
    
    return "\nLR Prediction: {} \nDT Prediction: {}".format(
        output_label(pred_lr[0]),
        output_label(pred_dt[0])
    )

news = "Olympiakos is the 2024 Conference league winner"
print(manual_testing(news))



