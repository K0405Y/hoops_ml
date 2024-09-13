import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
from src.preprocessing.prep import merge_data, preprocess_data
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
from scipy.stats import uniform, randint


# Define the input function
def input_data():
    data = merge_data()
    data = preprocess_data(data) 
    X = data.drop('PTS', axis=1)
    y = data['PTS']
    return X, y

# #train class
class ModelTrainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.model = XGBRegressor(random_state=42)
        self.best_params = None
        self.best_score = None

    def train_models(self):
        param_distributions = {
            'max_depth': randint(3, 11),
            'learning_rate': uniform(0.01, 0.29),
            'n_estimators': randint(100, 1001),
            'min_child_weight': randint(1, 11),
            'subsample': uniform(0.5, 0.5),
            'colsample_bytree': uniform(0.5, 0.5)
        }

        random_search = RandomizedSearchCV(
            self.model,
            param_distributions,
            n_iter=5,
            cv=5,
            n_jobs=-1,
            verbose=1,
            scoring='neg_mean_squared_error',
            random_state=42
        )

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        with mlflow.start_run(run_name="XGBoost_Model"):
            random_search.fit(X_train, y_train)
            self.best_params = random_search.best_params_
            self.best_score = -random_search.best_score_  # Convert back to MSE

            # Log best hyperparameter combination
            mlflow.log_params(self.best_params)

            # Log all trials
            for i, (params, score) in enumerate(zip(random_search.cv_results_['params'], random_search.cv_results_['mean_test_score'])):
                with mlflow.start_run(run_name=f"Trial_{i}", nested=True):
                    mlflow.log_params(params)
                    mlflow.log_metric("mse", -score)  # Convert back to MSE

            # Train final model with best parameters
            best_model = XGBRegressor(**self.best_params, random_state=42)
            best_model.fit(X_train, y_train)

            # Log training metrics
            y_train_pred = best_model.predict(X_train)
            train_mse = mean_squared_error(y_train, y_train_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            train_rmse = np.sqrt(train_mse)  # Calculate RMSE
            mlflow.log_metric("train_mse", train_mse)
            mlflow.log_metric("train_r2", train_r2)
            mlflow.log_metric("train_rmse", train_rmse)  # Log RMSE

            # Evaluate on test set and log test metrics
            y_pred = best_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)  # Calculate RMSE

            mlflow.log_metric("test_mse", mse)
            mlflow.log_metric("test_r2", r2)
            mlflow.log_metric("test_rmse", rmse)  # Log RMSE

            # Log the best model with signature
            signature = mlflow.models.infer_signature(X_train.iloc[[0]], best_model.predict(X_train.iloc[[0]]))
            mlflow.xgboost.log_model(best_model, "best_model", signature=signature, input_example = X_train.iloc[[0]],
                                      registered_model_name = "NBA_XGB", conda_env = "C:/hoops_ml/conda copy.yaml")                                             
        return None
    
def main():
    # Set the MLflow tracking URI to Databricks
    os.environ['MLFLOW_TRACKING_URI'] = 'databricks'
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment("/Users/kehinde.awomuti@pwc.com/NBA_XGB")
    X, y = input_data()
    X = X.astype("float64")
    y = y.astype("float64")
    trainer = ModelTrainer(X, y)
    trainer.train_models()
if __name__ == "__main__":
    main()