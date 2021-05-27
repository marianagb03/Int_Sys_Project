####################################################
#   Probabilites of scoring points on an F1 race 
#       using Machine Learning algorithms
#
#             Author: Mariana Garcia
#
####################################################

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Import datasets needed
df1 = pd.read_csv (r'results.csv')
df1 = df1[["raceId","constructorId","driverId","grid","position"]]
df1['position'] = df1['position'].replace(r'\N', 0)

df2 = pd.read_csv (r'races.csv')
df2 = df2[["raceId","year","circuitId"]]

df3 = pd.read_csv (r'drivers.csv')
df3 = df3[["driverId","driverRef"]]

df4 = pd.read_csv (r'constructors.csv')
df4 = df4[["constructorId","name"]]

df5 = pd.read_csv (r'circuits.csv')
df5 = df5[["circuitId","name","country"]]

# Merge datasets
df6 = pd.merge(df1,df2, on='raceId')
df6 = df6[["constructorId","driverId","grid","position","year","circuitId"]]
df_drv = df6

# Data preprocessing, change 0 to NaN in whole data
df6[["constructorId","driverId","grid","position","year","circuitId"]] = df6[["constructorId","driverId","grid","position","year","circuitId"]].replace(0, np.NaN)
# print(df6.isnull().sum())

# Discard all rows with null values and reset index
df6 = df6.dropna()
df6 = df6.reset_index(drop=True)
# print(df6.isnull().sum())
df6[["constructorId","driverId","grid","position","year","circuitId"]] = df6[["constructorId","driverId","grid","position","year","circuitId"]].astype(int)

# Use as maximum grid or initial position as 20 
df6['position'] = df6['position'].apply(lambda x: x if x <= 20 else x-(x-20))

# Define if a driver finshed in the top 10 and scored points add column to dataset
df6['score'] = df6['position'].apply(lambda x: 1 if x in range(1,11) else 0)

# Final data with dependent and independent values
df = df6.drop("position",axis=1)
print(df.tail(20))

# Split in train and test data
X_train, X_test, y_train, y_test = train_test_split(df.drop("score", axis=1), df.score, test_size=0.3)

# Logistic regression
def logRegression(const, driver, grid, year, circuit):
    model = LogisticRegression(max_iter=9000)
    model.fit(X_train, y_train)
    score=model.score(X_test, y_test)
    print('Score of the model:', score)
    print(score)
    y_pred = model.predict([[const, driver, grid, year, circuit]])
    
    return (int)(y_pred)

# #Random forest
def randomForest(const, driver, grid, year, circuit):
    # params = {
    #         'n_estimators':[100, 300, 500, 800, 1200],
    #         'max_depth': [5, 8, 15, 25, 30],
    #         'min_samples_split': [2, 5, 10, 15, 100],
    #         'min_samples_leaf': [1, 2, 5, 10] 
    #     }
    # clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=params)
    # clf.fit(X_train, y_train)
    # print(clf.best_params_)
    
    tree_clf = DecisionTreeClassifier(max_depth = 3)
    tree_clf.fit(X_train,y_train)
    y_pred_rf= tree_clf.predict(X_test)
    print("Tree Score", accuracy_score(y_test, y_pred_rf))

    rnd_clf = RandomForestClassifier(max_depth=25, min_samples_leaf=1, min_samples_split=15, n_estimators=1200)
    rnd_clf.fit(X_train, y_train)
    y_pred_rf = rnd_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred_rf)
    print("Random Forest Aaccuracy:", acc)
    y_pred = rnd_clf.predict([[const, driver, grid, year, circuit]])
  
    return (int)(y_pred)

# Start main function
print("\nPREDICT IF A DRIVER WITH THEIR TEAM CAN SCORE POINTS IN A CERTAIN F1 RACE")

start = "y"
while start == "y" or start == "Y":

    print("\n1-Logistic Regression \n2-Random Forest")
    mod = int(input("Select the method to use: "))
    yer = int(input("Select a year between 1980-2020: "))
    print("\n")

    df_drv = df_drv.loc[df_drv['year'] == yer]
    drv_list = df_drv.driverId.unique()
    drv = df3.loc[df3['driverId'].isin(drv_list)]
    print(drv)
    driv = int(input("Select the Id of a driver running that year: "))
    print("\n")

    cntr_list = df_drv.constructorId.unique()
    cntr = df4.loc[df4['constructorId'].isin(cntr_list)]
    print(cntr)
    constructor = int(input("Select the Id of a constructor from that year: "))
    print("\n")

    circ_list = df_drv.circuitId.unique()
    circ = df5.loc[df5['circuitId'].isin(circ_list)]
    print(circ)
    cir = int(input("Select the Id of a circuit from that year: "))

    init_pos = int(input("Starting grid position (1-20): "))
    
    if mod == 1:
        pred = logRegression(constructor,driv,init_pos,yer,cir)
        print(pred)

    elif mod == 2:
        pred = randomForest(constructor,driv,init_pos,yer,cir)
        print(pred)

    else:
        print("Invalid option")
        
    start = input("Do you want to keep predicting? y/n: ")
    

print("If you bet and lose money, don't blame the IA")