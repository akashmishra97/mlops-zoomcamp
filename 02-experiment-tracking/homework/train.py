import os
import pickle
import click
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    # Set the tracking URI
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    # Set the artifact location
    mlflow.set_experiment(experiment_name="taxi-experiment")
    artifact_location = os.path.abspath("./mlruns")
    
    # Enable MLflow autologging
    mlflow.sklearn.autolog()

    with mlflow.start_run() as run:
        # Load the datasets
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        # Initialize and train the model
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = rf.predict(X_val)

        # Calculate RMSE manually
        mse = mean_squared_error(y_val, y_pred)
        rmse = mse ** 0.5

        # Log the RMSE metric explicitly
        mlflow.log_metric("rmse", rmse)

        print(f"RMSE: {rmse}")

        # Manually log parameters and model
        mlflow.sklearn.log_model(rf, "model")
        mlflow.log_param("max_depth", rf.max_depth)
        mlflow.log_param("random_state", rf.random_state)


if __name__ == '__main__':
    run_train()
