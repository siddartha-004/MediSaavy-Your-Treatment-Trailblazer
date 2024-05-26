import numpy as np 
import pandas as pd
import pickle 
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
files_and_directories = os.listdir('./medicine-recommendation-system')

# Display the files and directories
# for item in files_and_directories:
#     print(item)
# my_data=pd.read_csv("medicine-recommendation-system/Drug prescription Dataset.csv")
# my_data
# duplicate = my_data.duplicated()
 
# print("Duplicate Rows :")
 
# print(duplicate)
# duplicate_rows = my_data[my_data.duplicated()]
# print("Duplicate Values :", len(duplicate_rows))
# duplicate_vals = my_data.drop_duplicates()
# duplicate_vals
# my_data.isnull().sum()
# y = my_data["drug"]
# np.random.seed(42)

# X = my_data[["disease","age","gender","severity"]].values
# y = my_data["drug"].values 
# le_disease = preprocessing.LabelEncoder()
# le_disease.fit([  'diarrhea','gastritis','arthritis','migraine'])
# X[:,0] = le_disease.transform(X[:,0])

# le_gender = preprocessing.LabelEncoder()
# le_gender.fit(['female','male'])
# X[:,2] = le_gender.transform(X[:,2]) 

#   # le_age = preprocessing.LabelEncoder()
#   # le_age.fit([   '4',  '5',  '6',  '7',  '8',  '9', '10', '11', '12', '13', '14', '15', '16', '17' ,'18', '19', '20', '21', '22', '23', '24', '25', '26', '27',
#   # '28', '29', '30', '31'])
#   # X[:,1] = le_age.transform(X[:,1])

# le_severity = preprocessing.LabelEncoder()
# le_severity.fit([ 'LOW', 'NORMAL', 'HIGH'])
# X[:,3] = le_severity.transform(X[:,3])




# X[0:5]
# X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=1)
# train_data = pd.DataFrame(X_trainset)
# train_data['target'] = y_trainset
# train_data.to_csv('medicine-recommendation-system/train_data.csv', index=False)
# test_data = pd.DataFrame(X_testset)
# test_data['target'] = y_testset
# test_data.to_csv('medicine-recommendation-system/test_data.csv', index=False)
# drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 7)
# drugTree.fit(X_trainset,y_trainset)
# predTree = drugTree.predict(X_testset)
# train = drugTree.predict(X_trainset)
# predictions_test =  metrics.accuracy_score(y_testset, predTree)
# print("DecisionTrees's Testing Accuracy: ",predictions_test)
# predictions_train = drugTree.predict(X_trainset)
# Train_acc = metrics.accuracy_score(y_trainset,predictions_train)
# print("DecisionTrees's Training Accuracy: ",Train_acc)
# pickle.dump(drugTree,open("medicine-recommendation-system/drugTree.pkl","wb"))

def all_recommender():
    dataset = pd.read_csv('medicine-recommendation-system/Training.csv')
    X = dataset.drop('prognosis', axis=1)
    y = dataset['prognosis']

    le = LabelEncoder()
    le.fit(y)
    Y = le.transform(y)
        
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=20)
    models = {
    'SVC': SVC(kernel='linear'),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'KNeighbors': KNeighborsClassifier(n_neighbors=5),
    'MultinomialNB': MultinomialNB()
}
    for model_name, model in models.items():
 
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        print(f"{model_name} Accuracy: {accuracy}")

        cm = confusion_matrix(y_test, predictions)
        print(f"{model_name} Confusion Matrix:")
        print(np.array2string(cm, separator=', '))

        print("\n" + "="*40 + "\n")
    svc = SVC(kernel='linear')
    svc.fit(X_train,y_train)
    ypred = svc.predict(X_test)
    accuracy_score(y_test,ypred)
    pickle.dump(svc,open('medicine-recommendation-system/svc.pkl','wb'))
all_recommender()