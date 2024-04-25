import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import pickle

bank = pd.read_csv('bank.csv')

X = bank.iloc[:, [0, 5, 9, 11, 12, 13, 14]]
y = bank['deposit']

model = GradientBoostingClassifier()
param_search = {
  'max_depth' : [3, 5],
  'n_estimators': [50, 100]
}

print('Training...')
gsearch = GridSearchCV(estimator=model,
                        param_grid=param_search)
gsearch = gsearch.fit(X, y)

file = open('model.sav', 'wb')
pickle.dump(gsearch, file)
file.close()
print('Done')