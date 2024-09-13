# Import all libraries and functions from the main script
from flask import Flask, request, jsonify
import mlflow
import pandas as pd
from preprocessing.NBA import player_averages, best_run_id, merge_data, preprocess_data, select_params, mlflow_logs
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

app = Flask(__name__)

#train api 
@app.route('/train', methods = ['POST'])
def train():
    #  #create file paths
    #  file_path1 = "C:/NBA Project/Regular Season Player PerGame Stats/PlayerPerGameStats04-24.csv"
    #  file_path2 = "C:/NBA Project/Regular Season Advanced/PlayerAdvancedStats04-24.csv"

    # create merged data from both files
     data = merge_data()
     data.to_csv("C:/NBA Project/Python-Folder/resultdata.csv", index=False)

    # preprocess the data using the preprocess function from main script
     result_data = preprocess_data(data)
    #  result_data.to_csv("C:/NBA Project/Python-Folder/prepdata.csv", index=False)

    # Specify models to train
     models = [KNeighborsRegressor(), GradientBoostingRegressor(), DecisionTreeRegressor()]

    # Extract features and target variable
     x = result_data.drop('PTS', axis=1)
     y = result_data['PTS']

    # set test sizes and random states
     test_sizes = [0.2, 0.25, 0.3]
     random_states = [0, 1, 42, 43]

    # Set up MLflow tracking and experiment
     mlflow.set_tracking_uri("http://127.0.0.1:8080")
     mlflow.set_experiment("NBA_project")

    # Select best test size and random state for each model by calling the function from the main script
     best_params_per_model = select_params(models, x, y, test_sizes, random_states)

    # Log metrics and artifacts for each model
     mlflow_logs(models, x, y, best_params_per_model)
     best_run_id(experiment_name="NBA_project")

     #return message 
     return jsonify({"status": "success", "message": "Training completed successfully"})

@app.route('/predict', methods=['POST'])
def predict(): 
    #set mlflow tracking
    mlflow.set_tracking_uri("http://127.0.0.1:8080")

    #extract best run id by calling the run id function from main script
    run_id = best_run_id("NBA_project")
    # get best model using run id
    logged_model_uri = f"runs:/{run_id}/model"  
    #get pyfunc model
    loaded_model = mlflow.pyfunc.load_model(logged_model_uri)

    # call merged data from the train module
    data = pd.read_csv("C:/NBA Project/Python-Folder/resultdata.csv")
    
    #get json of request
    name = request.get_json()
    player_names = name.get('player_names', [])

    # Get player averages using the function from the main script
    player_data = player_averages(player_names, data)

    # player_data.to_csv("C:/NBA Project/Python-Folder/playeraverages.csv", index=False)

    # Get predictions using the loaded MLflow model
    predictions =loaded_model.predict(player_data)

    #dataframe of predictions
    result = pd.DataFrame({'Player': player_names, 'Predicted PPG': [round(i, 2) for i in predictions]})

    # turn result dataframe to dic and return it as json
    return jsonify(result.to_dict(orient='records'))

 #set port 
 #    
if __name__ == '__main__':
    app.run(port=5000, debug=True)