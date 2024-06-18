#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pickle
import pandas as pd
import numpy as np
import os


# In[4]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[5]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[7]:


df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')


# In[8]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[10]:


std_dev = np.std(y_pred)


# In[11]:


std_dev


# In[18]:


# Step 4: Create the ride_id column with correct formatting

df['ride_id'] = f'{2023:04d}/{3:02d}_' + df.index.astype('str')

# Step 5: Create a dataframe with the results
df_result = pd.DataFrame({
    'ride_id': df['ride_id'],
    'predicted_duration': y_pred
})

# Step 6: Save the results to a Parquet file
output_file = 'predictions.parquet'
df_result.to_parquet(output_file, engine='pyarrow', compression=None, index=False)


# In[ ]:




