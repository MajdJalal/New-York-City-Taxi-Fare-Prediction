# New-York-City-Taxi-Fare-Prediction
This is a Supervised learning project.
I got the data of (New York City Taxi Fare Prediction) from a kaggle competition.
the goal was to predict the "fare_amount" of the a test_set.

Here is what i did:
1)I created a pipeline to transform the data 
i wrote 4 classes to do that 
*coords: clean data from wrong longitude and latitude
*deal missing: dealing with the null values, either by dropping the rows , or filling it with a median value fit from the training set
*passenger_cnt: delas with illogical passenger_cnt
*add_attribs:combing existing attributes and comming up with a more logical one 
*extract: extracting useful columns from an already given one


2)then i tried three regression models and found out the best to be forest regression
3)saving the predicted test_set in a csv file


**NOTE: i couldnt upload the train_set as it is considered to be big in GITHUB, u can reference it from kaggle itself;
it is worth noting that i just trained my model on the first 1M instances of the dataset, and that gave me good results.
