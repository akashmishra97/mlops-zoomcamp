import argparse
import pickle
import pandas as pd
import numpy as np

# Step 1: Parse CLI arguments for year and month
parser = argparse.ArgumentParser(description='Predict ride durations.')
parser.add_argument('--year', type=int, required=True, help='Year for ride data')
parser.add_argument('--month', type=int, required=True, help='Month for ride data')
args = parser.parse_args()

# Step 2: Load the model and DictVectorizer
with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

# Step 3: Define categorical columns
categorical = ['PULocationID', 'DOLocationID']

# Step 4: Define function to read data
def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

# Step 5: Read data for the specified year and month
filename = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{args.year:04d}-{args.month:02d}.parquet'
df = read_data(filename)

# Step 6: Transform data and make predictions
dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

# Step 7: Calculate mean predicted duration
mean_pred_duration = np.mean(y_pred)

# Step 8: Print mean predicted duration
print(f"Mean predicted duration for {args.year}-{args.month}: {mean_pred_duration:.2f} minutes")

# Step 9: Save results to Parquet file
df['ride_id'] = f'{args.year:04d}/{args.month:02d}_' + df.index.astype('str')
df_result = pd.DataFrame({
    'ride_id': df['ride_id'],
    'predicted_duration': y_pred
})
output_file = 'predictions.parquet'
df_result.to_parquet(output_file, engine='pyarrow', compression=None, index=False)

