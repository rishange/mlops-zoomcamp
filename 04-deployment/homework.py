# get_ipython().system('pip freeze | grep scikit-learn==1.5.0')
# get_ipython().system('python -V')

import sys
import pickle
import pandas as pd

# import mlflow

from prefect import task, flow

def load_model(filename):
    with open(filename, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model

def read_data(filename):   

    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    
    
    return df

def predict(model, dv, df, year, month, output_file):
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result = pd.DataFrame()

    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    print(f"Mean predicted duration is {df_result['predicted_duration'].mean():2f}")

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )


@task
def apply_model(input_file, output_file, year, month):
    dv, model = load_model('model.bin')
    df = read_data(input_file)
    predict(model=model, dv=dv, df=df, year=year, month=month, output_file=output_file)


@flow
def run():    
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    # year = 2023
    # month = 3

    output_file = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:4d}-{month:02d}.parquet'

    apply_model(input_file=input_file, output_file=output_file, year=year, month=month)

if __name__ == '__main__':
    run()



