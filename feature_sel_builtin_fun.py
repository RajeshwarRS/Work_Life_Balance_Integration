import pandas as pd

from sklearn.feature_selection import SelectKBest, mutual_info_regression

data=pd.read_csv('train.csv')
data=data.iloc[:,2:]
data=data.dropna()
data=pd.get_dummies(data)
ydata=data['Mental Fatigue Score']
data=data.drop('Mental Fatigue Score',axis=1)
selector = SelectKBest(mutual_info_regression, k=5)
selector.fit(data,ydata)
print("SELECTING TOP 5 FEATURES ")
features=data.columns[selector.get_support()]
for i in features:
	print(i)
