import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
#from matplotlib import pyplot as plt
inputs = pd.read_csv("inputs.csv")


inputs = inputs.drop(['Unnamed: 0'],axis='columns')
#inputs = inputs.drop(['RECOVD'],axis='columns')

targets = pd.read_csv("targets.csv")
targets.head()
targets = targets.drop(['Unnamed: 0'],axis='columns')
print('VACCINES')
print(inputs['VAX_MANU'].value_counts())
print('Allergies')
print(inputs['ALLERGIES'].value_counts())
print('AGE_YRS')
print(inputs['AGE_YRS'].value_counts())
print('SEX')
print(inputs['SEX'].value_counts())
from sklearn.preprocessing import LabelEncoder
# evaluate RFE for classification
from numpy import mean
from numpy import std
le = LabelEncoder()
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
# define dataset
le = LabelEncoder()
inputs['VAX_MANU']=le.fit_transform(inputs['VAX_MANU'])
inputs['ALLERGIES']=le.fit_transform(inputs['ALLERGIES'])
inputs['AGE_YRS']=le.fit_transform(inputs['AGE_YRS'])
inputs['SEX']=le.fit_transform(inputs['SEX'])
inputs=inputs.drop(['ALLERGIES'],axis=1)
# targets.loc[(targets.RECOVD=='y'),'RECOVD']=1
# targets.loc[(targets.RECOVD=='n'),'RECOVD']=0


X=inputs
y=targets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=8675309)






# inputs['VAX_MANU']=le.fit_transform(inputs['VAX_MANU'])
# inputs['ALLERGIES']=le.fit_transform(inputs['ALLERGIES'])
# inputs['AGE_YRS']=le.fit_transform(inputs['AGE_YRS'])
# inputs['SEX']=le.fit_transform(inputs['SEX'])
# print
from sklearn.preprocessing import OneHotEncoder
onehot=OneHotEncoder()
import numpy as np
from sklearn.compose import ColumnTransformer
columnTransformer = ColumnTransformer([('encoder',
                                         OneHotEncoder(),
                                         [0])],
                                       remainder='passthrough')
inputs=np.array(columnTransformer.fit_transform(inputs))
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
def run_exps(X_train: pd.DataFrame , y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame) -> pd.DataFrame:
    '''
    Lightweight script to test many models and find winners
:param X_train: training split
    :param y_train: training target vector
    :param X_test: test split
    :param y_test: test target vector
    :return: DataFrame of predictions
    '''
    
    dfs = []
    models = [
          ('LogReg', LogisticRegression()), 
          ('RF', RandomForestClassifier()),
          ('KNN', KNeighborsClassifier()),
          ('SVM', SVC()), 
          ('GNB', GaussianNB()),
          ('XGB', XGBClassifier())
        ]
    results = []
    names = []
    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
    target_names = ['no', 'yes']
    for name, model in models:
        kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=90210)
        cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)
        clf = model.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(name)
        print(classification_report(y_test, y_pred, target_names=target_names))
        results.append(cv_results)
        names.append(name)
        this_df = pd.DataFrame(cv_results)
        this_df['model'] = name
        dfs.append(this_df)
        final = pd.concat(dfs, ignore_index=True)
    return final
final=run_exps(X_train,y_train,X_test,y_test)
bootstraps = []
for model in list(set(final.model.values)):
    model_df = final.loc[final.model == model]
    bootstrap = model_df.sample(n=30, replace=True)
    bootstraps.append(bootstrap)
        
bootstrap_df = pd.concat(bootstraps, ignore_index=True)
results_long = pd.melt(bootstrap_df,id_vars=['model'],var_name='metrics', value_name='values')
time_metrics = ['fit_time','score_time'] # fit time metrics
## PERFORMANCE METRICS
results_long_nofit = results_long.loc[~results_long['metrics'].isin(time_metrics)] # get df without fit data
results_long_nofit = results_long_nofit.sort_values(by='values')
## TIME METRICS
results_long_fit = results_long.loc[results_long['metrics'].isin(time_metrics)] # df with fit data
results_long_fit = results_long_fit.sort_values(by='values')
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
metrics = list(set(results_long_nofit.metrics.values))
bootstrap_df.groupby(['model'])[metrics].agg([np.std, np.mean])
sgd=SGDClassifier()
sgd.fit(X_train,y_train)


# from sklearn import preprocessing
# scaler = preprocessing.StandardScaler().fit(inputs_train)


# x = scaler.transform(inputs_train)
# '''

# from sklearn.feature_selection import VarianceThreshold
# sel = VarianceThreshold(threshold=(.1 * (1 - .1)))
# inp=sel.fit_transform(inputs) 
# '''
