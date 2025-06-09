import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

from prefect import flow, task
import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

import os
from pathlib import Path

@task
def read_data(month='03'):
    filename = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-{month}.parquet'
    df = pd.read_parquet(filename)
    print(f"We loaded {len(df)} records for month {month} of 2023")
    return filename

@task
def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds()/60
    df = df[(df['duration'] >= 1) & (df['duration'] <=60)]
    df[['PULocationID', 'DOLocationID']] = df[['PULocationID', 'DOLocationID']].astype(str)
    
    return df

@task
def transform_data(df, v=None):
    df_dict_list = df.loc[:,['PULocationID', 'DOLocationID']].to_dict('records')
    if (v == None):
        v = DictVectorizer()
        X = v.fit_transform(df_dict_list)
    else:
        X = v.transform(df_dict_list)
    y = df['duration']

    # models_folder = Path('models')
    # models_folder.mkdir(exist_ok=True)
    
    # # Save DictVectorizer and datasets
    # dump_pickle(v, os.path.join(models_folder, "dv.pkl"))
    # dump_pickle((X, y), os.path.join(models_folder, "train.pkl"))
    return X, y, v

@task
def train_data(X, y, v):

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("nyc-taxi-experiment")
    with mlflow.start_run():
        mlflow.set_tag("model", "LinearRegressor")

        reg = LinearRegression().fit(X, y)
        
        print(f"The intercept of the model is {reg.intercept_:.2f}")
        
        mlflow.log_param("intercept", reg.intercept_)
        mlflow.sklearn.log_model(reg, artifact_path="artifacts_local")
    return v, reg

@task
def register_model():

    client = MlflowClient()

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name("nyc-taxi-experiment")

    last_run = mlflow.last_active_run().info.run_id

    # Register the best model
    mlflow.register_model(
        model_uri=f"runs:/{last_run}/models",
        name="linear-regressor"
    )
    return f"runs:/{last_run}/models"

@flow
def run_train():
    filename = read_data()
    df = read_dataframe(filename)
    X_train, y_train, v= transform_data(df)
    v, reg = train_data(X_train, y_train, v)
    model_uri = register_model()

if __name__ == "__main__":
    run_train.serve(name="orchestration-homework")