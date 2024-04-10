
import pandas as pd
from sklearn.ensemble import VotingRegressor,RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm,tree
from sklearn import linear_model

data_train  = pd.read_csv('DataSet_train.csv')
label_train = pd.read_csv('zzc_Label_train.csv')

X_train = data_train.values[:,1:]
Y_train = label_train.values[:,1]

data_test  = pd.read_csv('New_data_7-16/Testing data/tcga.csv')
X_test = data_test.values[:,1:]


# model = linear_model.Lasso(alpha=0.1)
# model = linear_model.LogisticRegression()
# model = svm.NuSVR(tol=1e-9)
# model = tree.DecisionTreeRegressor(random_state=825,max_depth=3)
# model = RandomForestRegressor(n_estimators=1, random_state=1)
# model = KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)#

# r1 = linear_model.Lasso()
r1 = linear_model.SGDRegressor()
# r2 = svm.NuSVR(tol=1e-9)
r3 = KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
r4 = tree.DecisionTreeRegressor(random_state=825,max_depth=3)
r5 = RandomForestRegressor(n_estimators=1, random_state=1)
model = VotingRegressor([('lr', r1),   ('r3', r3), ('r4', r4),('r5', r5)])

model.fit(X_train, Y_train)


pred = model.predict(X_test)




