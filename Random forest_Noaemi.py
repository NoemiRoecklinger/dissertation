
#######Noemi

# saving a classifier

from sklearn.externals import joblib
filename = 'randomforest_model.sav'
# instead of classifier, rename based on your file
joblib.dump(clf, filename)






###uploading the saved classifier

import pandas as pd
from sklearn.externals import joblib

filename = 'randomforest_model.sav'
loaded_model = joblib.load(filename)


### uploading your dataset for prediction 

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values



### optional######
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#################################
######



### perform prediction on your x_test, y_test
### accuracy
result = loaded_model.score(X_test, y_test)

### prediction of y 
# Predicting the Test set results
y_pred = loaded_model.predict(X_test)
y_pred = (y_pred > 0.5)





# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)




