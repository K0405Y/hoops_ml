import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#function to import two files needed and to append them
def merge_data():
    file_path1 = "C:/basketball_ml_project/data/raw/Regular Season Player PerGame Stats/PlayerPerGameStats04-24.csv"
    file_path2 = "C:/basketball_ml_project/data/raw/Regular Season Player Advanced Stats/PlayerAdvancedStats04-24.csv"
    # Read data from CSV files
    data1 = pd.read_csv(file_path1, encoding="latin1")
    data2 = pd.read_csv(file_path2, encoding="latin1")

    # Drop unnecessary columns
    data1 = data1.drop(['Year', 'Team', 'Rk'], axis=1)
    data2 = data2.drop(['Age', 'MP', 'Player', 'Pos', 'Age', 'Tm', 'G', 'Rk'], axis=1)

    # Concatenate the dataframes
    data = pd.concat([data1, data2], axis=1)

    return data

#function to preprocess data
def preprocess_data(data, correlation_threshold=0.5):
    # List of numeric features
    numeric_features = list(data.select_dtypes(include=['number']).columns)

    # Remove the 'Year' column
    numeric_features.remove('Year')

    # Get correlation matrix
    cor = data[numeric_features].corr()

    # Convert 'Pos' column to category type
    data['Pos'] = data['Pos'].astype('category')

    # Initialize the label encoder
    label_encoder = LabelEncoder()

    # Apply label encoding to the 'Pos' column
    data['Pos'] = label_encoder.fit_transform(data['Pos'])

    # Fill missing values with median
    data[numeric_features] = data[numeric_features].fillna(data[numeric_features].median())

    # Filter columns with highly positive correlations with PTS
    positive_corr_cols = cor['PTS'][(cor['PTS'] > correlation_threshold) & (cor.index != 'Year')].index

    pkl.dump(positive_corr_cols, open("C:/basketball_ml_project/data/features/poscorcols.pkl", 'wb'))

    # Concatenate the selected columns with 'Pos'
    final_data = pd.concat([data[positive_corr_cols], data['Pos']], axis=1)

    final_data.to_csv("C:/basketball_ml_project/data/processed/final_input.csv")

    return final_data

#function to select best random states and test sizes 
def select_params(models, x, y, test_sizes, random_states):
    # initialize empty dictionary
    best_params_per_model = {}

    # initialize a for loop tha
    for model in models:
        print(f"\n Getting best params for model {model.__class__.__name__}")

        # set variable names
        best_test_size = None
        best_random_state = None
        best_r2_score = -float('inf')

        # initialize a loop for the test sizes and random states
        for test_size in test_sizes:
            for random_state in random_states:
                X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
                # fit models across each of the test sizes and random states to get the best
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                # assign best params for each model
                if r2 > best_r2_score:
                    best_r2_score = r2
                    best_test_size = test_size
                    best_random_state = random_state
        # create a dictionary across each mdodel that contains the best test size  and random state  
        best_params_per_model[model.__class__.__name__] = {
                'best_test_size': best_test_size,
                'best_random_state': best_random_state,
            }
    print("Done getting all params")   
    return best_params_per_model

#function to train models, log metrics and artifacts
def mlflow_logs(models, x, y, best_params_per_model):
    for model in models:
            # Get best parameters for the each model in the "models" list
            params = best_params_per_model[model.__class__.__name__]
            # mlflow run for each model
            with mlflow.start_run(run_name= model.__class__.__name__):
                # Log model parameters
                mlflow.log_param("model", model.__class__.__name__)
                mlflow.log_param("best_test_size", params['best_test_size'])
                mlflow.log_param("best_random_state", params['best_random_state'])

                # train, calculate and log model metrics
                X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=params['best_test_size'], random_state=params['best_random_state'])
                model = model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                r2 = r2_score(y_test, y_pred)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                mse = mean_squared_error(y_test, y_pred)

                mlflow.log_metric("r2_score", r2)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mse", mse)

                # Infer model signature
                signature = mlflow.models.infer_signature(X_test, model.predict(X_test))

                # Log artifacts
                mlflow.sklearn.log_model(model, "model", signature=signature, input_example=X_test.iloc[[0]])

                print(f"Model outputs for: {model.__class__.__name__}")
                print(f"Best test size: {params['best_test_size']}")
                print(f"Best random state: {params['best_random_state']}")
                print(f"R2 score: {r2}")
                print(f"RMSE: {rmse}")
                print(f"MSE: {mse}")

#function to get best run ID from all models and register it
def best_run_id(experiment_name):
    # get mlflow experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    #condition to check if experiment exists
    if experiment:
        #get experiment id
        experiment_id = experiment.experiment_id
        # search runs in experiment, in this instance, we're only using one experiment
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        # another condition to check if runs are not empty
        if not runs.empty:
            # get best run based on min rmse
            min_rmse_run = runs.loc[runs['metrics.rmse'].idxmin()]
            #get run id of best run
            min_rmse_run_id = min_rmse_run['run_id']

            # mlflow.register_model(f"runs:/{min_rmse_run_id}/model", "NBAModel")

            return min_rmse_run_id
        # alternative condition to search of runs
        else:
            print(f"No runs found in experiment '{experiment_name}'.")
            return None
    # alternative condition to not finding experiment
    else:
        print(f"Experiment '{experiment_name}' not found.")
        return None    


# function to send input data through manipulations to get player averages over the last 3 years
def player_averages(player_names, df):
    """
    This function calculates the averages of NBA player statistics for one or more players.

    Args:
        player_names (list or str): A list of player names or a single player name.
        df (DataFrame): A pandas DataFrame containing NBA player statistics.

    Returns:
        DataFrame: A DataFrame containing the calculated averages for each player, 
                   including position information. Handles player not found case.
    """

    # Ensure player_names is a list
    player_names = [player_names] if not isinstance(player_names, list) else player_names

    # Initialize empty list for results
    all_results = []

    # Loop through each player name
    for player_name in player_names:
        # Filter data for the current player
        player_data = df[df['Player'] == player_name]
        if player_data.empty:
            # Player not found, add 'Player not found' message to results
            all_results.append({'Player': player_name, 'Result': "Player not found"})
            continue

        # Keep only the entry with the maximum number of games played (G) for each year
        player_data_max_games = player_data.loc[player_data.groupby('Year')['G'].idxmax()]

        # Sort the data by year
        player_data_sorted = player_data_max_games.sort_values(by='Year', ascending=False)

        # Drop the 'Year' column
        player_data_sorted = player_data_sorted.drop('Year', axis=1)

        # Calculate averages
        player_averages_numeric = player_data_sorted.iloc[:3, 2:].mean().round(2).to_frame().T

        # Add player name and position information back to averages DataFrame
        player_averages_numeric['Player'] = player_name

        # Append results to all_results list
        all_results.append(player_averages_numeric.to_dict(orient='records')[0])

    # Convert results to DataFrame
    all_results_df = pd.DataFrame(all_results)

    # Convert other selected columns to integer
    all_results_df[['G', 'GS']] = all_results_df[['G', 'GS']].astype(int)

    cols = pkl.load(open("C:/NBA Project/Python-Folder/poscorcols.pkl", 'rb'))

    all_results_df = all_results_df[cols]  

    return all_results_df

def main(player_names):
    # Import and transform data
    data = merge_data()
    result_data = preprocess_data(data)

    # Specify models
    models = [KNeighborsRegressor(), GradientBoostingRegressor(), DecisionTreeRegressor()]

    # Extract features and target variable
    x = result_data.drop('PTS', axis=1)
    y = result_data['PTS']

    test_sizes = [0.2, 0.25, 0.3]
    random_states = [0, 1, 42, 43, 100, 313]

    # Set up MLflow tracking
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    experiment_name = "NBA_project"

    # Select best test size and random state for each model
    best_params_per_model = select_params(models, x, y, test_sizes, random_states)

    # Log metrics and artifacts for each model
    mlflow_logs(models, x, y, best_params_per_model)

    # Get the best run ID from all models
    run_id = best_run_id(experiment_name=experiment_name)

    print(run_id)

    if run_id:
        run_id = best_run_id(experiment_name=experiment_name)

        logged_model = f"runs:/{run_id}/model"

        # Load model as a PyFuncModel
        loaded_model = mlflow.pyfunc.load_model(logged_model)

        # Get predictions for player averages
        averages = player_averages(player_names, data)

        predictions =  loaded_model.predict(averages)

        result = pd.DataFrame({'Player': player_names, 'Predicted Points': predictions})

        return result 

# if __name__ == "__main__":
    
#     # Example: Call the main function with player names
#     player_names = ["Andrew Wiggins", "Stephen Curry"]
#     predictions = main(player_names)
#     print(predictions)