# BigDataAnalytics-HW2
## Xgboost 參數組合
import 所需套件，事前將Class欄位值改成由0~8
```python
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics  
from sklearn.grid_search import GridSearchCV   
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

data = pd.read_csv('LargeTrain1.csv')
train = pd.DataFrame(data)
target = 'Class'
``` 
### 初始參數
*   learning_rate =0.1
*   n_estimators=10
*   max_depth=5
*   min_child_weight=1
*   gamma=0
*   subsample=0.8
*   colsample_bytree=0.8
*   reg_alpha=0
```python
predictors = [x for x in train.columns if x not in [target]]
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=10,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softmax',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, train, predictors)
```
參數丟入XGBClassifier得到Accuracy為0.9985  
>Tune max_depth and min_child_weight
```python
param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=10, max_depth=5,min_child_weight=1,
 gamma=0, subsample=0.8, colsample_bytree=0.8,objective= 'multi:softmax', scale_pos_weight=1, seed=27),
 param_grid = param_test1 , n_jobs=4 , iid=False , cv=5)
gsearch1.fit(train[predictors],train[target])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
```
>得到最好 max_depth=9 , min_child_weight=1 後，Tune gamma
```python
param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=10, 
  max_depth=9,min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
  objective= 'multi:softmax', scale_pos_weight=1,seed=27), 
  param_grid = param_test3, n_jobs=4 , iid=False, cv=5)
gsearch3.fit(train[predictors],train[target])
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
```
>得到最好 gamma=0 後，Tune subsample與 colsample_bytree 
```python
param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=10, max_depth=9,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'multi:softmax', scale_pos_weight=1,seed=27), 
 param_grid = param_test4, n_jobs=4,iid=False, cv=5)
gsearch4.fit(train[predictors],train[target])
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_
```
>得到最好 subsample=0.9 與 colsample_bytree =0.6，後Tune reg_alpha
```python
param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=10, max_depth=9,
 min_child_weight=1, gamma=0, subsample=0.9 , colsample_bytree=0.6 ,
 objective= 'multi:softmax', scale_pos_weight=1,seed=27), 
 param_grid = param_test6 ,n_jobs=4,iid=False, cv=5)
gsearch6.fit(train[predictors],train[target])
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_
```
>得到最好reg_alpha=0.1 ，因此得到最佳參數組合為
*   max_depth=9
*   min_child_weight=1
*   gamma=0
*   subsample=0.9
*   colsample_bytree =0.6
*   reg_alpha=0.1
```python
xgb3 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=10,
 max_depth=9,
 min_child_weight=1,
 gamma=0,
 subsample=0.9,
 colsample_bytree=0.6,
 reg_alpha=0.1,
 objective= 'multi:softmax',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb3, train, predictors)
```
>透過Xgboost最佳參數組合得到 Accuracy0.9994
### Confusion Matrix 驗證
```python
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm , classes , normalize=False , title='Confusion matrix' , cmap=plt.cm.Blues):
    plt.imshow(cm , interpolation='nearest' , cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks , classes , rotation=45)
    plt.yticks(tick_marks , classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
    else:
        print('Confusion matrix , without normalization')
    print(cm)
    
    thresh = cm.max()/2.
    for i , j in itertools.product(range(cm.shape[0]) , range(cm.shape[1])):
        plt.text(j , i , cm[i,j] , horizontalalignment='center' , color='white' if cm[i,j]>thresh else 'black')
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predict label')
    
data = pd.read_csv('LargeTrain.csv')
target = 'Class'
train = [x for x in data.columns if x!= target]
class_name = ['Class' + str(x) for x in range(1,10)]
X = data[train]
y = data[target]

X_train , X_test , y_train , y_test = train_test_split(X, y , random_state=0)
clf = XGBClassifier(max_depth=9,min_child_weight=1,gamma=0,subsample=0.9,colsample_bytree=0.6,reg_alpha=0.1)
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)

cnf_matrix = confusion_matrix(y_test , y_pred)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix,classes=class_name,normalize=True,title='Normalized confusion matrix')
plt.show
```
![image](file:C:/Users/test/Downloads/confusion matrix.jpg)  
## GBM參數組合
```python
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

data = pd.read_csv('LargeTrain1.csv')
train = pd.DataFrame(data)
target = 'Class'
```
>Tune n_estimators  
```python
predictors = [x for x in train.columns if x not in [target]]
param_test1 = {'n_estimators':range(20,81,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1,
 min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10), 
 param_grid = param_test1,n_jobs=4,iid=False, cv=5)
gsearch1.fit(train[predictors],train[target])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
```
>得到最好n_estimators=80後，Tune max_depth 與 min_samples_split  
```python
param_test2 = {'max_depth':range(5,16,2), 'min_samples_split':range(200,1001,200)}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80,
 max_features='sqrt', subsample=0.8,random_state=10), 
 param_grid = param_test2,n_jobs=4,iid=False, cv=5)
gsearch2.fit(train[predictors],train[target])
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_
```
>得到最好max_depth=11，min_samples_split=200後，Tune min_samples_leaf  
```python
param_test3 = {'min_samples_leaf':range(30,71,10)}
gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80,
 max_depth=11,max_features='sqrt', subsample=0.8, random_state=10), 
 param_grid = param_test3,n_jobs=4,iid=False, cv=5)
gsearch3.fit(train[predictors],train[target])
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
```
>得到最好min_samples_leaf=40後，Tune max_features  
```python
param_test4 = {'max_features':range(7,20,2)}
gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80,max_depth=11, 
 min_samples_split=1600, min_samples_leaf=40, subsample=0.8, random_state=10),
 param_grid = param_test4,n_jobs=4,iid=False, cv=5)
gsearch4.fit(train[predictors],train[target])
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_
```
>得到最好max_features=19後，後Tune subsample
```python
param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80,
 max_depth=11,min_samples_split=1600, min_samples_leaf=40, subsample=0.8, random_state=10,max_features=19),
 param_grid = param_test5,n_jobs=4,iid=False, cv=5)
gsearch5.fit(train[predictors],train[target])
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_
```
>得到最好subsample=0.9，因此最佳參數組合為  
*   n_estimators=80
*   max_depth=11
*   min_samples_split=200
*   min_samples_leaf=40
*   max_features=19
*   subsample=0.9
>透過GBM最佳參數組合得到Accuracy=0.9979  
### Confusion Matrix 驗證
```python
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm , classes , normalize=False , title='Confusion matrix' , cmap=plt.cm.Blues):
    plt.imshow(cm , interpolation='nearest' , cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks , classes , rotation=45)
    plt.yticks(tick_marks , classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
    else:
        print('Confusion matrix , without normalization')
    print(cm)
    
    thresh = cm.max()/2.
    for i , j in itertools.product(range(cm.shape[0]) , range(cm.shape[1])):
        plt.text(j , i , cm[i,j] , horizontalalignment='center' , color='white' if cm[i,j]>thresh else 'black')
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predict label')
    
data = pd.read_csv('LargeTrain.csv')
target = 'Class'
train = [x for x in data.columns if x!= target]
class_name = ['Class' + str(x) for x in range(1,10)]
X = data[train]
y = data[target]

X_train , X_test , y_train , y_test = train_test_split(X, y , random_state=0)
clf = GradientBoostingClassifier(n_estimators=80,max_depth=11,min_samples_split=200,min_samples_leaf=40,subsample=0.9,max_features=19)
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)

cnf_matrix = confusion_matrix(y_test , y_pred)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix,classes=class_name,title='Confusion matrix , without normalization')
plt.show
```
![image](file:C:/Users/test/Downloads/cnf_matrix-GBM.jpg)  
