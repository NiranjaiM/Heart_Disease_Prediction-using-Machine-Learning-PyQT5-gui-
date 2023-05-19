#!/usr/bin/env python
# coding: utf-8

# In[49]:


#Diabetes Prediction Using Support Vector Machine
import pickle
import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Machine Learning
from sklearn.neighbors import KNeighborsClassifier


#For training the model
def train():
    heart = pd.read_csv("Dataset1.csv")
    # we have unknown values '?'
    # change unrecognized value '?' into mean value through the column
    #min_max = MinMaxScaler()
    #columns_to_scale = ['age', 'sex','cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    #heart[columns_to_scale ] = min_max.fit_transform(heart[columns_to_scale])
    y = heart['target']
    X = heart.drop(['target'], axis = 1)
    #y.head()
    #X.info()
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.32, random_state = 101)
    # heart = pd.get_dummies(heart, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])




    
    
    from sklearn.svm import SVC
    model = SVC(kernel='linear')
    svc=model.fit(X_train,Y_train)
    
    #Save Model As Pickle File
    with open('svc.pkl','wb') as m:
        pickle.dump(svc,m)
    test(X_test,Y_test)

#Test accuracy of the model
def test(X_test,Y_test):
    with open('svc.pkl','rb') as mod:
        p=pickle.load(mod)
    pre=p.predict(X_test)
    print (accuracy_score(Y_test,pre)) #Prints the accuracy of the model


def find_data_file(filename):
    if getattr(sys, "frozen", False):
        # The application is frozen.
        datadir = os.path.dirname(sys.executable)
    else:
        # The application is not frozen.
        datadir = os.path.dirname(__file__)

    return os.path.join(datadir, filename)


def check_input(data) ->int :
    df=pd.DataFrame(data=data,index=[0])
    with open(find_data_file('svc.pkl'),'rb') as model:
        p=pickle.load(model)
    op=p.predict(df)
    return op[0]
if __name__=='__main__':
    train()    


# In[ ]:


# In[ ]:




