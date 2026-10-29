from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub


dagshub.init(repo_owner='Ahmad-Imran-id', repo_name='Mlflow-learning-for_expriement_tracking', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/Ahmad-Imran-id/Mlflow-learning-for_expriement_tracking.mlflow')

wine=load_wine()
X=wine.data
y=wine.target

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=2,test_size=0.2)

max_depth=10
n_estimators=5

#This statements autologs everything and there is no need for log statements
mlflow.autolog()
mlflow.set_experiment('Auto_log_exp')

with mlflow.start_run():
    #creating and fitting data to model
    rf=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators)
    rf.fit(X_train,y_train)
    y_pred=rf.predict(X_test)

    #Caculating metrics
    accuracy=accuracy_score(y_test,y_pred)
    precision=precision_score(y_true=y_test,y_pred=y_pred,average='weighted')

    # #logging metrics
    # mlflow.log_metrics({'accuracy':accuracy,'precision':precision})
    # mlflow.log_params({'max_depth':max_depth,'n_estiamtors':n_estimators})

    #plot
    cm=confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=wine.target_names,yticklabels=wine.target_names)
    plt.xlabel('True values')
    plt.ylabel('Predicted values')
    plt.title('Confusion Matrix')
    
    plt.savefig('Confusion_matrix.png')

    # #logging plot
    # mlflow.log_artifact('Confusion_matrix.png')

    mlflow.log_artifact(__file__) #(needs to be expicitly logged)

    #setting tags (needs to be expicitly mentioned)
    mlflow.set_tags({'Author':'Ahmad','Project_name':'Wine_classification'})

    # #logging model
    # mlflow.sklearn.log_model(rf,name='Random-forrest-model',skops_trusted_types=["sklearn.tree._tree.Tree"])