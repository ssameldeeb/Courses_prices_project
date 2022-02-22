# import modules
from numpy import dtype
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor


# readings the data
data = pd.read_csv("courses.csv")

# git information from the data file
print(data.shape)
print(data.columns.values)
print(data.head())
print(data.info())
print(data.isnull().sum())

# drop the columns are not important
data = data.drop(["course_id","course_title"],axis=1)
print(data.head())
print(data.info())

# clean data["is_paid"]
print(data["is_paid"].value_counts())
data.loc[data["is_paid"]==True , "is_paid"] = 1
data.loc[data["is_paid"]==False , "is_paid"] = 0
data["is_paid"] = data["is_paid"].astype(int)
print(data["is_paid"].value_counts())
print(data.head())

# clean data["level"]
print(data["level"].value_counts())
data["level"] = data["level"].map({"All Levels":1,"Beginner Level":2,"Intermediate Level":3,"Expert Level":4})
print(data["level"].value_counts())
print(data.head())

# clean data["price"]
print(data["price"].value_counts())
data.loc[data["price"]=="Free" , "price"] = 1
print(data["price"].value_counts())
data["price"] = data["price"].astype(int)

# clean data["content_duration"]
print(data["content_duration"].unique())
data.drop(data[data["content_duration"] == "0"].index,inplace=True)
print(data["content_duration"].unique())
data[["num_hours","hours"]] = data["content_duration"].str.split(" ",expand=True)
print(data.head())


# drop the columns are not important
data = data.drop(["hours","content_duration"], axis=1)
print(data.head())
print(data.dtypes)

# change data["num_hours"] dtype
data["num_hours"] = data["num_hours"].astype(float)
print(data.dtypes)


# clean data["published_timestamp"]
data["published_timestamp"] = pd.to_datetime(data["published_timestamp"])
data["published_year"] = data["published_timestamp"].dt.year
data = data.drop(["published_timestamp"],axis=1)
print(data.dtypes)

# clean data["subject"]
print(data["subject"].value_counts())
La = LabelEncoder()
data["subject"] = La.fit_transform(data["subject"]) 
print(data["subject"].value_counts())
print(data.dtypes)
print(data.head())
plt.figure(figsize=(10,6))
sns.countplot(data["subject"],palette="Greens")
plt.show()

# show the correlation in the data
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(),annot=True,cmap="Greens")
plt.show()


# split the data
x = data.drop("price",axis=1)
print(x.head())
y = data["price"]
print(y.head())

# split the data for testing and training
x_train , x_test , y_train , y_test = train_test_split(x , y , train_size=0.7)


# use different kind of regression to get the the best score and predict
# use LinearRegression 
m = LinearRegression()
m.fit(x_train,y_train)
train_score = m.score(x_train,y_train)
print(train_score)
test_score = m.score(x_test,y_test)
print(test_score)
print(m.coef_)

# nake line
print("-"*100)

# use MLPRegressor 
m1 =MLPRegressor(activation="identity",max_iter=3000)
m1.fit(x_train,y_train)
train_score1 = m1.score(x_train,y_train)
print(train_score1)
test_score1 = m1.score(x_test,y_test)
print(test_score1)

# nake line
print("-"*100)

# use KNeighborsRegressor 
m2 = KNeighborsRegressor(n_neighbors=101)
m2.fit(x_train,y_train)
train_score2 = m2.score(x_train,y_train)
print(train_score2)
test_score2 = m2.score(x_test,y_test)
print(test_score2)

# nake line
print("-"*100)

# use RandomForestRegressor
m3 = RandomForestRegressor(max_depth=65,n_estimators=100)
m3.fit(x_train,y_train)
train_score3 = m3.score(x_train,y_train)
print(train_score3)
test_score3 = m3.score(x_test,y_test)
print(test_score3)
y_pred = m3.predict(x_test)
print(y_pred[:5])

# create table from models
models = pd.DataFrame({"model":['m','m1','m2','m3'], "train_score":[test_score,test_score1,test_score2,test_score3],
                       "test_score":[test_score,test_score1,test_score2,test_score3]})
print(models)

# export the_autput_file to csv_file
the_autput = pd.DataFrame({"y_test":y_test,"y_pred":y_pred})
# the_autput.to_csv("the_autput.csv",index=False)