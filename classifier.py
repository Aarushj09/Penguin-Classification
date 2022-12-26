import pandas as pd
import numpy as np
from sklearn.svm import SVC

def train_ovo(X_train, y_train):
    classifiers = []
    for i in range(0,3):
        for j in range(i+1,3):
            X_train1 = X_train[(y_train==i) | (y_train==j)]
            y_train1 = y_train[(y_train==i) | (y_train==j)]
            model = SVC(kernel="linear", C=0.025)
            model.fit(X_train1, y_train1)
            classifiers.append(model)
    return classifiers

def test_ovo(X_test, classifiers):
    y_pred = []
    for i in range(0,3):
            y_pred.append(classifiers[i].predict(X_test))
    y_pred = np.array(y_pred)
    y_final=[]
    for i in range(0,len(y_pred[0])):
        y_final.append(np.argmax(np.bincount(y_pred[:,i])))
    return y_final



df = pd.read_csv('penguins_train.csv')
unique_islands = df['Island'].unique()
for i in range(len(unique_islands)):
    df['Island'].replace(unique_islands[i],i,inplace=True)
unique_sex = df['Sex'].unique()
for i in range(len(unique_sex)):
    df['Sex'].replace(unique_sex[i],i,inplace=True)
unique_species = df['Species'].unique()
for i in range(len(unique_species)):
    df['Species'].replace(unique_species[i],i,inplace=True)
df['Clutch Completion'].replace('Yes',1,inplace=True)
df['Clutch Completion'].replace('No',0,inplace=True)
for i in range(len(df.columns)):
    df[df.columns[i]] = df[df.columns[i]].fillna(df[df.columns[i]].mode()[0])

features = df[['Island','Clutch Completion','Culmen Length (mm)','Culmen Depth (mm)','Flipper Length (mm)','Body Mass (g)','Sex','Delta 15 N (o/oo)','Delta 13 C (o/oo)']]
label = df['Species']
X_train = features
y_train = label
ovo = train_ovo(X_train, y_train)



df_test = pd.read_csv('penguins_test.csv')
for i in range(len(df_test['Island'])):
    if df_test['Island'][i] not in unique_islands:
        unique_islands = np.append(unique_islands, df_test['Island'][i])
for i in range(len(unique_islands)):
    df_test['Island'].replace(unique_islands[i],i,inplace=True)
        
for i in range(len(df_test['Sex'])):
    if df_test['Sex'][i] not in unique_sex:
        unique_sex = np.append(unique_sex, df_test['Sex'][i])
for i in range(len(unique_sex)):
    df_test['Sex'].replace(unique_sex[i],i,inplace=True)

df_test['Clutch Completion'].replace('Yes',1,inplace=True)
df_test['Clutch Completion'].replace('No',0,inplace=True)
for i in range(len(df_test.columns)):
    df_test[df_test.columns[i]] = df_test[df_test.columns[i]].fillna(df_test[df_test.columns[i]].mode()[0])
features = df_test[['Island','Clutch Completion','Culmen Length (mm)','Culmen Depth (mm)','Flipper Length (mm)','Body Mass (g)','Sex','Delta 15 N (o/oo)','Delta 13 C (o/oo)']]
X_test = features
# print(X_test)
y_pred = test_ovo(X_test, ovo)
# print(y_pred)y
y_final = []
for i in range(len(y_pred)):
    y_final.append(unique_species[y_pred[i]])
# print(y_final)
# print final values to csv file
df_new = pd.DataFrame(y_final, columns=['Species'])
df_new.to_csv('penguins_test_pred.csv', index=False)
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
