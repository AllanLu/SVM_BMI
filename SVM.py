import pandas as pd
import numpy as np
from sklearn import datasets,model_selection,svm

data = pd.read_csv("bmi.csv")
X=np.array(data[['height','weight']])
Y=np.array(data['label'])
xtrain,xtest,ytrain,ytest=model_selection.train_test_split(X,Y,test_size=0.1)

lsvc = svm.LinearSVC(dual=False)
svcrbf = svm.SVC(max_iter=1000)
svcpoly=svm.SVC(kernel='poly',max_iter=1000)
svcsig=svm.SVC(kernel='sigmoid',max_iter=1000)

lsvc.fit(xtrain,ytrain)
svcrbf.fit(xtrain,ytrain)
svcpoly.fit(xtrain,ytrain)
svcsig.fit(xtrain,ytrain)

# 0.952 0.948
print(lsvc.score(xtrain,ytrain))
print(lsvc.score(xtest,ytest))

# 0.999 0.999
print(svcrbf.score(xtrain,ytrain))
print(svcrbf.score(xtest,ytest))

# 0.945 0.936
print(svcpoly.score(xtrain,ytrain))
print(svcpoly.score(xtest,ytest))

# 0.317 0.315
print(svcsig.score(xtrain,ytrain))
print(svcsig.score(xtest,ytest))

