import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv('train.csv')
#test_df = pd.read_csv('test_data.csv')
#df = train_df.join(test_df)

df.head()

df["PassengerId"] = df["PassengerId"].astype("int32")
df.info()

df.describe()

train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).count().sort_values(by='Survived', ascending=False)


train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).count().sort_values(by='Survived', ascending=False)


train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)

train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).count().sort_values(by='Survived', ascending=False)


train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# only in titanic_df, fill the two missing values with the most occurred value, which is "S".
train_df["Embarked"] = train_df["Embarked"].fillna("S")

# plot
sns.factorplot('Embarked','Survived', data=train_df,size=4,aspect=3)

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))

# sns.factorplot('Embarked',data=titanic_df,kind='count',order=['S','C','Q'],ax=axis1)
# sns.factorplot('Survived',hue="Embarked",data=titanic_df,kind='count',order=[1,0],ax=axis2)
sns.countplot(x='Embarked', data=train_df)
sns.countplot(x='Survived', hue="Embarked", data=train_df, order=[1,0])

# group by embarked, and get the mean for survived passengers for each value in Embarked
embark_perc = train_df[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'])

# Either to consider Embarked column in predictions,
# and remove "S" dummy variable, 
# and leave "C" & "Q", since they seem to have a good rate for Survival.

# OR, don't create dummy variables for Embarked column, just drop it, 
# because logically, Embarked doesn't seem to be useful in prediction.

embark_dummies_titanic  = pd.get_dummies(train_df['Embarked'])
embark_dummies_titanic.drop(['S'], axis=1, inplace=True)

embark_dummies_test  = pd.get_dummies(train_df['Embarked'])
embark_dummies_test.drop(['S'], axis=1, inplace=True)

titanic_df = train_df.join(embark_dummies_titanic)

titanic_df.drop(['Embarked'], axis=1,inplace=True)
test_df.drop(['Embarked'], axis=1,inplace=True)


grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();

grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()



facet = sns.FacetGrid(train_df,row='Pclass',col="Sex", hue="Survived",size=2.2,aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train_df['Age'].max()))
facet.add_legend()

train_df['Family'] =  train_df["Parch"] + train_df["SibSp"]


facet = sns.FacetGrid(train_df,row='Family', hue="Survived",size=2.2,aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train_df['Age'].max()),ylim=(0,.1))
facet.add_legend()

train_df['Family'].loc[train_df['Family'] > 0] = 1
train_df['Family'].loc[train_df['Family'] == 0] = 0

facet = sns.FacetGrid(train_df,row='Family',col="Sex", hue="Survived",size=2.2,aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train_df['Age'].max()),ylim=(0,.06))
facet.add_legend()


# plot
fig= plt.subplots(figsize=(10,5))

# sns.factorplot('Family',data=titanic_df,kind='count',ax=axis1)
sns.countplot(x='Family', data=train_df, order=[1,0])

facet = sns.FacetGrid(train_df,col="Sex", hue="Survived",size=2.2,aspect=4)
facet.map(sns.countplot,'Family',data=train_df, order=[1,0])
facet.add_legend()


# average of survived for those who had/didn't have any family member
family_perc = train_df[["Family", "Survived"]].groupby(['Family'],as_index=False).mean()

family_perc.head()

sns.barplot(x='Family', y='Survived', data=family_perc, order=[1,0])


train_df['Age2'] = pd.qcut(train_df['Age'], 10)

train_df['Age2'] = train_df['Age2'].astype("str")

f = train_df['Age2'].sort_values().unique()

train_df['Age2'] = train_df['Age2'].map({"(0.419, 14.0]":0,"(14.0, 19.0]":1,"(19.0, 22.0]":2,
"(22.0, 25.0]":3,'(25.0, 28.0]':4,'(28.0, 31.8]':5,'(31.8, 36.0]':6,'(36.0, 41.0]':7,
'(41.0, 50.0]':8, '(50.0, 80.0]':9})



grid = sns.FacetGrid(train_df, row='Age2', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()




sns.boxplot(x=train_df["Fare"])

Q1 = train_df["Fare"].quantile(0.25)
Q3 =  train_df["Fare"].quantile(0.75)
IQR = Q3-Q1
low_limit = Q1 - 1.5*IQR
high_limit = Q3 + 1.5*IQR
train_df[train_df["Fare"] > high_limit]
train_df["Fare"].sort_values(ascending=False).head()
train_df["Fare"] = train_df["Fare"].replace(512.3292, 263)

train_df.drop(["Ticket"],axis =1, inplace = True)



facet = sns.FacetGrid(train_df,col="Sex", hue="Survived",size=2.2,aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train_df['Fare'].max()),ylim=(0,.06))
facet.add_legend()


facet = sns.FacetGrid(train_df,col="Family", hue="Survived",size=2.2,aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train_df['Fare'].max()),ylim=(0,.06))
facet.add_legend()


facet = sns.FacetGrid(train_df,col="Age2", hue="Survived",size=2.2,aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train_df['Fare'].max()),ylim=(0,.06))
facet.add_legend()


train_df["Title"] = train_df["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)

train_df.Title = train_df['Title'].replace(['Lady', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
train_df.Title = train_df['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
train_df.Title = train_df['Title'].replace('Mlle', 'Miss')
train_df.Title = train_df['Title'].replace('Ms', 'Miss')
train_df.Title = train_df['Title'].replace('Mme', 'Mrs')


#defining the figure size of our graphic
plt.figure(figsize=(12,5))

#Plotting the result
sns.countplot(x='Title', data=train_df, palette="hls")
plt.xlabel("Title", fontsize=16) #seting the xtitle and size
plt.ylabel("Count", fontsize=16) # Seting the ytitle and size
plt.title("Title Name Count", fontsize=20) 
plt.xticks(rotation=45)
plt.show()

train_df['Title'] = train_df['Title'].map({'Mr':0, 'Mrs':1, 'Miss':2, 'Master':3, 'Rare':4, 'Royal':5})
        
        
        
facet = sns.FacetGrid(train_df,col="Sex", hue="Survived",size=2.2,aspect=4)
facet.map(sns.kdeplot,'Title',shade= True)
facet.set(xlim=(0, train_df['Title'].max()),ylim=(0,.7))
facet.add_legend()


facet = sns.FacetGrid(train_df,row="Family", hue="Survived",size=2.2,aspect=4)
facet.map(sns.kdeplot,'Title',shade= True)
facet.set(xlim=(0, train_df['Title'].max()),ylim=(0,.6))
facet.add_legend()


facet = sns.FacetGrid(train_df,row="Age2", hue="Survived",size=2.2,aspect=4)
facet.map(sns.kdeplot,'Title',shade= True)
facet.set(xlim=(0, train_df['Title'].max()),ylim=(0,.6))
facet.add_legend()







df[["Title", "Survived"]].groupby(["Title"], as_index = False ).mean()

Title_mapping = {"Mr":1, "Miss":2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 5} 
df["Title"] = df["Title"].map(Title_mapping)

df.drop("Name", axis =1, inplace= True)

df = pd.get_dummies( df, columns = ["Title"], prefix = "Tit")
df = pd.get_dummies( df, columns = ["Embarked"], prefix = "Em")

df.drop(["Parch","SibSp"], axis =1, inplace =True)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


#Importing the auxiliar and preprocessing librarys 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import accuracy_score

#Models
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, SGDClassifier, LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier, VotingClassifier, RandomTreesEmbedding

clfs = []
seed = 3

clfs.append(("LogReg", 
             Pipeline([("Scaler", StandardScaler()),
                       ("LogReg", LogisticRegression())])))

clfs.append(("XGBClassifier",
             Pipeline([("Scaler", StandardScaler()),
                       ("XGB", XGBClassifier())]))) 
clfs.append(("KNN", 
             Pipeline([("Scaler", StandardScaler()),
                       ("KNN", KNeighborsClassifier())]))) 

clfs.append(("DecisionTreeClassifier", 
             Pipeline([("Scaler", StandardScaler()),
                       ("DecisionTrees", DecisionTreeClassifier())]))) 

clfs.append(("RandomForestClassifier", 
             Pipeline([("Scaler", StandardScaler()),
                       ("RandomForest", RandomForestClassifier())]))) 

clfs.append(("GradientBoostingClassifier", 
             Pipeline([("Scaler", StandardScaler()),
                       ("GradientBoosting", GradientBoostingClassifier(max_features=15, n_estimators=150))]))) 

clfs.append(("RidgeClassifier", 
             Pipeline([("Scaler", StandardScaler()),
                       ("RidgeClassifier", RidgeClassifier())])))

clfs.append(("BaggingRidgeClassifier",
             Pipeline([("Scaler", StandardScaler()),
                       ("BaggingClassifier", BaggingClassifier())])))

clfs.append(("ExtraTreesClassifier",
             Pipeline([("Scaler", StandardScaler()),
                       ("ExtraTrees", ExtraTreesClassifier())])))

#'neg_mean_absolute_error', 'neg_mean_squared_error','r2'
scoring = 'accuracy'
n_folds = 10

results, names  = [], [] 

for name, model  in clfs:
    kfold = KFold(n_splits=n_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, y_train, cv= 5, scoring=scoring, n_jobs=-1)    
    names.append(name)
    results.append(cv_results)    
    msg = "%s: %f (+/- %f)" % (name, cv_results.mean(),  cv_results.std())
    print(msg)
    
# boxplot algorithm comparison
fig = plt.figure(figsize=(15,6))
fig.suptitle('Classifier Algorithm Comparison', fontsize=22)
ax = fig.add_subplot(111)
sns.boxplot(x=names, y=results)
ax.set_xticklabels(names)
ax.set_xlabel("Algorithmn", fontsize=20)
ax.set_ylabel("Accuracy of Models", fontsize=18)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
plt.show()

