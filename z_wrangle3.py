# I am using this separate wrangle file to feature engineer a more robust model beyond my MVP without breaking anything in that,

# Plan:
# 1. Use the same query - change only scalers
# Linear & Standard
# See if it improves model
# 2. Import all the fields from the database, clean them up, and use K-best for modeling
# - Rename columns
# - Drop Columns
# - Engineer columns
# - Encode columns
# - Model


#################### IMPORT LIBRARIES #################

# Basic libraries
import pandas as pd
import numpy as np 

#Vizualization Tools
import matplotlib.pyplot as plt
import seaborn as sns

#Modeling Tools
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import statsmodels.api as sm

from datetime import date

import sklearn.preprocessing
from sklearn.model_selection import train_test_split


import warnings
warnings.filterwarnings("ignore")

#Custim functions
from env import host, user, password #Database credentials
import zillo_wrangle





################ PULL DATA FROM DB ############## 

def get_db_url(db_name):
    '''
    Gets appropriate url to pull data from credentials stored in env file
    '''
    return f"mysql+pymysql://{user}:{password}@{host}/{db_name}"


def get_data_from_sql():
    query = """
    SELECT * FROM predictions_2017
     JOIN properties_2017 USING(parcelid)
    WHERE (transactiondate >= '2017-01-01' AND transactiondate <= '2017-12-31') 
        AND propertylandusetypeid = '261'
        AND bedroomcnt > 0
        AND bathrooms > 0
        AND square_feet > 0 
        AND taxes > 0
        AND home_value > 0
        AND fips > 0
    ORDER BY fips;
    """
    df = pd.read_sql(query, get_db_url('zillow'))
    return df


################ REMOVE OUTLIERS #################

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

######## Clean Data ###########

def clean_data(df):
    '''
    This funciton takes in the zillow df and drops  Null values reassigns some dtypes.
    '''
    df = df.dropna()
    df['fips'] = df['fips'].astype(int)
    df['regionidzip'] = df['regionidzip'].astype('category')
    df['square_feet'] = df['square_feet'].astype('int')
    df['year'] = df['year'].astype(int)

    return df

######### ADD COUNTY AND STATE COLUMNS #######

def assign_county(row):
    '''
    Assign countes based on fips
    '''
    if row['fips']==6037:
        return 'Los Angeles'
    if row['fips']==6059:
        return 'Orange'
    if row['fips']==6111:
        return 'Ventura'

######## Feature engineering ########

### Getting an error for county... Comment out for now, use fips in models instead - same results

def engineer(zillow):
    '''
    Adds columns for county, state, and year, one-hot encodes county data
    '''
    #zillow['county'] = zillow.apply(lambda row: assign_county(row), axis =1) #Add counties
    #zillow['state'] = 'CA' #Add state
    zillow['age'] = date.today().year-zillow.year # Add age
    #One-hot Encode County
    dummy_df = pd.get_dummies(zillow[['fips']], drop_first=True)
    zillow = pd.concat([zillow, dummy_df], axis=1)
    return zillow

###### ADD COUNTY AVERAGE #############

#def county_avg(zillow):
#    '''
#    Defines average values by county
#    '''
#    la = zillow[zillow.county=='Los Angeles']
#    oc = zillow[zillow.county=='Orange']
#    ven = zillow[zillow.county=='Ventura']

#    la_avg = la.home_value.mean()
#    oc_avg = oc.home_value.mean()
#    ven_avg = ven.home_value.mean()

#    def assign_county_avg(row):
#        '''
#        Adds county averages
#        '''
#        if row['fips']==6037:
#            return la_avg
#        if row['fips']==6059:
#            return oc_avg
#        if row['fips']==6111:
#            return ven_avg

#   zillow['county_avg'] = zillow.apply(lambda row: assign_county_avg(row), axis =1)

#   return zillow

########## TRAIN VALIDATE TEST SPLIT #########

def split_my_data(df, pct=0.10):
    '''
    This splits a dataframe into train, validate, and test sets. 
    df = dataframe to split
    pct = size of the test set, 1/2 of size of the validate set
    Returns three dataframes (train, validate, test)
    '''
    train_validate, test = train_test_split(df, test_size=pct, random_state = 123)
    train, validate = train_test_split(train_validate, test_size=pct*2, random_state = 123)
    return train, validate, test

########## ADD BASELINE #########

def add_baseline(train, validate, test):
    '''
    Assigns median home price as baseline prediction
    '''
    baseline = train.home_value.median()
    train['baseline'] = baseline
    validate['baseline'] = baseline
    test['baseline'] = baseline
    return train, validate, test

######## SPLIT IN TO X /y features / target ########

def split_xy(train, validate, test):
    '''
    Splits dataframe into train, validate, and test data frames
    '''
    X_train = train.drop(columns='home_value')
    y_train = train.home_value

    X_validate = validate.drop(columns='home_value')
    y_validate = validate.home_value

    X_test = test.drop(columns='home_value')
    y_test = test.home_value

    return X_train, y_train, X_validate, y_validate, X_test, y_test

############## Standard Scale ###############

def scale(X_train, X_validate, X_test, train, validate, test):
    '''
    Uses robust scaler to scale specified numeric columns
    '''
    scaler = sklearn.preprocessing.MinMaxScaler()

    columns = ['bedroomcnt', 'bathrooms', 'square_feet', 'age']
    
    scaler.fit(X_train[columns])

    new_column_names = [c + '_scaled' for c in columns]

    X_train = pd.concat([X_train, pd.DataFrame(scaler.transform(X_train[columns]), columns=new_column_names, index = train.index),], axis=1)

    X_validate = pd.concat([X_validate, pd.DataFrame(scaler.transform(X_validate[columns]), columns=new_column_names, index = validate.index),], axis=1)

    X_test = pd.concat([X_test, pd.DataFrame(scaler.transform(X_test[columns]), columns=new_column_names, index = test.index),], axis=1)
    
    return X_train, X_validate, X_test

######### CALL ALL FUNCTIONS TOGETHER #######

def wrangle():
    '''
    Calls appropriate functions to import and clean data for modeling
    '''
    zillow = get_data_from_sql()
    zillow = remove_outliers(zillow, 1.5, ['bedrooms', 'bathrooms', 'square_feet', 'taxes', 'home_value'])
    zillow = clean_data(zillow) #Drop NAs and change dtypes
    zillow = engineer(zillow)
#    zillow = county_avg(zillow)
    train, validate, test = split_my_data(zillow)
    train, validate, test = add_baseline(train, validate, test)
    X_train, y_train, X_validate, y_validate, X_test, y_test = split_xy(train, validate, test)
    X_train, X_validate, X_test = scale(X_train, X_validate, X_test, train, validate, test)
    return train, X_train, y_train, X_validate, y_validate, X_test, y_test






