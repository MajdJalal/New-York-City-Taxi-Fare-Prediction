import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import math
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 
#from sklearn.externals import joblib


#train = pd.read_csv("C:\\Users\\PC\\Desktop\\Home\\ml\\env\\taxi_fare\\train.csv", chunksize = 100000)

train_iterator = pd.read_csv("C:\\Users\\PC\\Desktop\\Home\\ml\\env\\taxi_fare\\train.csv", chunksize = 1000000)

test_iterator = pd.read_csv("C:\\Users\\PC\\Desktop\\Home\\ml\\env\\taxi_fare\\test.csv", chunksize = 1000000)
#print(train_iterator[0].info)

for train in train_iterator:
    train = train
    train_set  = train.drop("fare_amount", axis = 1)
    train_labels = train["fare_amount"]
    #print(train.info())
    break

for test in test_iterator:
    test = test
    break
#print(train_set.info())
##preproccessing pipeline
#1)dropping all invalid data dor lonitude and latitude 
#class for doing that 
class coords (BaseEstimator, TransformerMixin) :
    def __init__(self):
        return
    def fit(self, x, y = None):
        return self
    
    def transform(self, x, y = None):
        condition1 = (x["pickup_longitude"] > 180) | (x["pickup_longitude"] < -180) | (x["pickup_latitude"] > 90) | (x["pickup_latitude"] < -90 )
        condition2 =( x["dropoff_latitude"] > 90) | (x["dropoff_latitude"] < -90) | (x["dropoff_longitude"] > 180) | (x["dropoff_longitude"] <  -180)
        return x.drop(x[(condition1 | condition2)].index)        

#2)drop all missing data 
##use sdropna
class deal_missing(BaseEstimator, TransformerMixin):

    def __init__(self, drop_missing = True, fillnull = False):
        self.drop_missing = drop_missing
        self.fillnull = fillnull

    def fit(self, x , y = None):
        if(self.fillnull):
            self.imputer = SimpleImputer(strategy = "median")
            self.imputer.fit(x)
            return self
        return self
            

    def transform(self, x, y = None):
        if(self.fillnull):
            return self.imputer.transform(x)
        
        elif (self.drop_missing) :
            return x.dropna()


#3)drop zero passenger count rows or delete the entire feature 
class passenger_cnt(BaseEstimator, TransformerMixin):
    def __init__(self, delete_feature = False):
        self.delete_feature = delete_feature
    def fit(self, x , y =None):
        return self
    def transform(self, x, y = None):
        if(self.delete_feature):
            return x.drop("passenger_count", axis = 1)
        return x.drop(x[x["passenger_count"] == 0].index)
    

#4) add attributes
class add_attributes(BaseEstimator, TransformerMixin):
    def __init__(self, distance = True):
        self.distance = distance
    def fit(self, x, y =None):
        return self
    def transform(self, x, y=None):
        if(self.distance):
            manip = x
            manip["distance"] = ((manip["pickup_longitude"] - manip["dropoff_longitude"])**2  + (manip["pickup_latitude"] - manip["dropoff_latitude"]) **2)**0.5
            return manip
#5)extract time and date
class extract(BaseEstimator, TransformerMixin):
    def __init__(self) :
        pass
    def fit(self, x, y =None):
        return self
    def transform(self, x, y = None):
        temp = x
        temp["pickup_datetime"] = pd.to_datetime(temp["pickup_datetime"])
        temp['hour'] = temp["pickup_datetime"].dt.hour
        temp['minute'] = temp["pickup_datetime"].dt.minute
        temp['second'] = temp["pickup_datetime"].dt.second
        temp['year'] = temp["pickup_datetime"].dt.year
        temp['month'] = temp["pickup_datetime"].dt.month
        temp['day'] = temp["pickup_datetime"].dt.day
        temp.drop("pickup_datetime",axis =1, inplace = True)
        temp.drop("key",axis =1, inplace = True)
        return temp

##pipeline 
pipe = Pipeline([
    ("coords", coords()),  
    ("passenger_cnt", passenger_cnt()),
    ("atrribs", add_attributes()),
    ("missing", deal_missing()),
    ("extract", extract())
])

train_transformed = pipe.fit_transform(train)

train_set_transformed = train_transformed.drop("fare_amount", axis = 1)
train_labels_transformed = train_transformed["fare_amount"]
##trainin model



forest_reg = RandomForestRegressor()
forest_reg.fit( train_set_transformed, train_labels_transformed)


##transform the test set
test_transforemd  = pipe.transform(test)
predicts = forest_reg.predict(test_transforemd)
predicts_series = pd.Series(predicts)
test_transforemd["fare_amount"] = predicts_series
test_transforemd["key"] = test["key"]
test_transforemd.to_csv("C:\\Users\\PC\\Desktop\\Home\\ml\\env\\taxi_fare\\ans.csv", index = False)

