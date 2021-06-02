import numpy as np
import pandas as pd
from sklearn import metrics
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df=pd.DataFrame()

data=pd.read_csv('train.csv')
data=data[data.isnull().sum(axis=1) < 2]

test_data=pd.read_csv('test.csv')
test_data=test_data[test_data.isnull().sum(axis=1) < 2]

df['Employee ID']=test_data['Employee ID']



corr_features= {'Age', 'Tenure','Vacations taken','Gender','Company Type'}

data=data.drop(corr_features,axis=1)
test_data=test_data.drop(corr_features,axis=1)



data=data.iloc[:,2:]
test_data=test_data.iloc[:,2:]




#data=data.interpolate()
#test_data=test_data.interpolate()
data=data.dropna()
test_data=test_data.dropna()



labels = data['Mental Fatigue Score']

labels=np.array(labels)
data=data.drop('Mental Fatigue Score', axis=1)
data=pd.get_dummies(data)

test_data=pd.get_dummies(test_data)

feature_list=list(data.columns)
data=np.array(data)

test_data=np.array(test_data)


train_features, test_features, train_labels, test_labels = train_test_split(data, labels, test_size = 0.01, random_state = 42)


rf = RandomForestRegressor(n_estimators = 100, oob_score = True,  n_jobs = -1, random_state = 42 , max_features="auto",min_samples_leaf =10)

rf.fit(train_features, train_labels);


predictn= rf.predict(test_features)


mse = metrics.mean_squared_error(test_labels, predictn)
rmse = math.sqrt(mse)
print('Accuracy for Random Forest',100 - rmse)

test_data_predict=rf.predict(test_data)


df['Mental Fatigue Score']=test_data_predict

df.to_csv("submission.csv",index=False)

