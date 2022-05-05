# CSci-4200-Final-Project
#This is the project for UMSL CSci 4200

import re

import pandas as pd

import numpy as np

import seaborn as sns

from datetime import datetime

from sklearn import preprocessing

from sklearn import linear_model

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import SGDRegressor

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

df = pd.read_csv('./Perth Data.csv')

#interpolation

mean_build= df['BUILD_YEAR'].mean()

mean_build =round(mean_build, 0 )

median_rank= df['NEAREST_SCH_RANK'].median()

df['BUILD_YEAR'].fillna( value = mean_build, inplace= True)

df['NEAREST_SCH_RANK'].fillna( value= median_rank, inplace= True)

counts=df['SUBURB'].nunique()

new_count =df['NEAREST_STN'].nunique()

df=df.dropna()

#We are dropping vales that we cannot interpolate further

M=df.corr()

sns.heatmap(M)

![Python Heat Map](https://user-images.githubusercontent.com/104104103/166865243-b4e4fd5f-aefe-4678-8c0b-55478e585636.png)

#This Plot seems to indicate that the variables generally have a weak correlation with price, the strongest correlation being Floor Area

#This section details hot encoding for the suburb variable. Since there exist 321 different suburbs and since manipulation of the main dataframe would become


#increasingly difficult when adding a 321x 33,000+ matrix to the mix, I am encoding it as a dummy data frame for ease of calculation when analyzing this section.

enc = OneHotEncoder(handle_unknown='ignore')

#enc_df = pd.DataFrame(enc.fit_transform(df[['SUBURB']]).toarray())

#dum_df = df.join(enc_df)

#dum_df

#Hot encoding the Nearest Station variable in a dummy dataframe

stn_df = pd.DataFrame(enc.fit_transform(df[['NEAREST_STN']]).toarray())


nst_df= df.join(stn_df)

nst_df

#In This step we combine some data cleaning with the introduction of aggregate columns

#We're adding a price range, a price std, and a price median. Since prices can vary wildly ( as price range suggests) it makes more sense to add a median over mean

#By examining the ratio of the price STD

df['PRICE_STDEV']=df.groupby('BATHROOMS')['PRICE'].transform('std')

df['PRICE_MEDIAN']=df.groupby('BATHROOMS')['PRICE'].transform('median')

df['PRICE_MAX']=df.groupby('BATHROOMS')['PRICE'].transform('max')

df['PRICE_MIN']=df.groupby('BATHROOMS')['PRICE'].transform('min')

df['PRICE_RANGE']=df['PRICE_MAX']-df['PRICE_MIN']

df=df.drop(columns = ['PRICE_MAX', 'PRICE_MIN', 'ADDRESS'])

num=.9*len(df)

num

#Num is roughly 28061, this corresponds to 90% test sample

#80% corresponds to 22449, as such we'll use that for this estimate

#70% corresponds to 19642

#60% corresponds to 16836

#50% corresponds to 14030

#40% corresponds to 11224


#30% corresponds to 8418

#20% corresponds to 5612

#10% corresponds to 2806

df_test= df.sample(n=28061)



cond= df.isin(df_test)

df_train = pd.concat([cond, df], axis="rows")

df_train=df_test.drop_duplicates(keep=False)

df_train


#The above gives us the 10% test data, we concetenated the dataframe with the original dataframe, 

#we then dropped all rows which had duplicates, leaving us with 10% of data that wasn't the original sample

df_test=df_test.drop( columns= ['SUBURB', 'NEAREST_STN', 'NEAREST_SCH'])

df_train=df_train.drop( columns= ['SUBURB', 'NEAREST_STN', 'NEAREST_SCH'])

#Normalizing our price data, and regardless of normalization method the SGD Regressor still showcases systemic error

scaler = StandardScaler()

#df_train['PRICE']=MinMaxScaler().fit_transform(np.array(df_train['PRICE']).reshape(-1,1))

df_train['PRICE']=scaler.fit_transform(np.array(df_train['PRICE']).reshape(-1,1))

#df_train


#df_test['PRICE']=MinMaxScaler().fit_transform(np.array(df_test['PRICE']).reshape(-1,1))

df_test['PRICE']=scaler.fit_transform(np.array(df_test['PRICE']).reshape(-1,1))

#df_train=df_train.astype('float')

#df_train.dtypes





#Splitting data

x_train=df_train
x_train = x_train[['BEDROOMS', 'BATHROOMS', 'LAND_AREA', 'GARAGE', 'NEAREST_SCH_RANK', 'BUILD_YEAR', 'FLOOR_AREA', 'DATE_SOLD', 'POSTCODE', 'NEAREST_STN_DIST', 

'CBD_DIST']]

x_test=df_test

x_test=x_test[['BEDROOMS', 'BATHROOMS', 'LAND_AREA', 'GARAGE', 'NEAREST_SCH_RANK', 'BUILD_YEAR', 'FLOOR_AREA', 'DATE_SOLD', 'POSTCODE', 'NEAREST_STN_DIST', 
'CBD_DIST']]

x_test=x_test.dropna()

x_train=x_train.dropna()

y_train=df_train['PRICE']



y_train=y_train.dropna()

y_test=df_test['PRICE']

y_test=y_test.dropna()





reg = linear_model.LinearRegression()

reg.fit(x_train,y_train)
#reg.score(x_test, y_test)



#Score is about .58 when the ratio is 10-90 test train .57 when 20-80, .56 when 30-70 .57 when 40-60 .57 when the ratio is 50-50, .57 when the ratio is 60-40

#.58 when the ratio is 70-30, .58 when the ratio is 80-20, and .57 when 90-10

le = LabelEncoder()


y_train = le.fit_transform(y_train)

y_test = le.fit_transform(y_test)

n_reg=linear_model.SGDRegressor()

n_reg.fit(x_train,y_train)

#n_reg.score(x_test,y_test)

#The regression score is on the order of magntiude of -1* 10^30. This implies a systematic failure for multivariate logistic regression to accurately model this data



#d = { 'x' : [.58, .57, .56, .57, .57, .57, .57, .58, .57], 'y': [10,20,30,40, 50,60,70,80,90]}


#df_plot= pd.DataFrame( data=d)

#df_plot

#sns.lineplot( data= df_plot, x='x', y='y')

#There is not a signfiicant difference between sample sizes and accuracy of linear models. Suggesting that multicollinearity may be a problem.




y_predict= n_reg.predict(x_test)

sns.lineplot(y_test,y_predict)

![SGD Regression](https://user-images.githubusercontent.com/104104103/166870819-28581e29-6ed3-4cec-b0bf-86ca0da1f53d.png)

#In The case of Using MinMaxScaler we still see wildly disproportionate regression, suggesting a severe systematic 

![SGD Regression 2](https://user-images.githubusercontent.com/104104103/166871255-c44dd618-ff9c-41d9-aad9-37827a6d3c7e.png)


#This plot suggests systematic failure on behalf of the SGD Regression. When analyzing the SGD regression plot to analyze accuracy the method detailed:

#in this medium article https://medium.com/@nikhilparmar9/simple-sgd-implementation-in-python-for-linear-regression-on-boston-housing-data-f63fcaaecfb1

#suggested that the ideal ratio should be a rough linear pattern with a slope of about 1 with some variance. This Ratio in this model borders on 10^17. 



#y_predict= reg.predict(x_test)

#sns.lineplot(y_test,y_predict)

![Python Regression Residual Linear](https://user-images.githubusercontent.com/104104103/166863978-56772770-b634-40f8-82fb-97d2be796a7d.png)

#Both Plots show forms of systematic error regardless of form, though linear plots appear to showcase far less error than SGD plots. There appear to be problems with multicolinearity and the correlations between variables price appear weakly correlated
 
