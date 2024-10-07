import pandas as pd
import numpy as np
from mlflow.pyfunc import PythonModel
import mlflow
import pickle as pkl
import io
from databricks.sdk import WorkspaceClient
import logging
import os

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

#custom class to help with custom inference workloads
class PlayerPredictionModel(PythonModel):
    def __init__(self, model_name, model_version):
        """
        PlayerPredictionModel class initialization
        """
        self.model_name = model_name
        self.model_version = model_version
        self.base_model = None
        self.player_data = None
        self.cols = None

    def load_context(self, context):
        """
        This function loads the base XGBoost model from databricks model registry, with the features and inference data loaded
        from databricks file system.
        """
        workspace = WorkspaceClient()

        if "DATABRICKS_RUNTIME_VERSION" in os.environ:
            #to address Databricks serving environment issues
            mlflow.set_tracking_uri("databricks")
        else:
            #using local MLflow tracking URI
            mlflow.set_tracking_uri("databricks")

        # Load the base XGBoost model from Model Registry
        try:
            self.base_model = mlflow.pyfunc.load_model(
                model_uri=f"models:/{self.model_name}/{self.model_version}"
            )
            logger.info(f"Base model loaded successfully: {self.model_name}/{self.model_version}")
        except Exception as e:
            logger.error(f"Failed to load base model: {str(e)}")
            raise RuntimeError(f"Failed to load base model: {str(e)}")
        
        # Load features from DBFS
        try:
            pkl_content = workspace.dbfs.download("dbfs:/FileStore/features.pkl")
            pkl_str = pkl_content.read()
            self.cols = pkl.loads(pkl_str)
            logger.info(f"Columns loaded successfully: {self.cols}")
        except Exception as e:
            logger.error(f"Failed to load column information: {str(e)}")
            raise RuntimeError(f"Failed to load column information: {str(e)}")
        
        # Load inference data from DBFS
        try:    
            #download the data
            file_content = workspace.dbfs.download("dbfs:/FileStore/inference_data.csv")
            file_str = file_content.read()
            #convert to readable format before reading as csv
            file_content_stream = io.BytesIO(file_str)
            self.player_data = pd.read_csv(file_content_stream, encoding="latin1")
                      
            # Preprocess the inference data
            self.player_data = self.player_data.sort_values(['Player', 'Year', 'G'], ascending=[True, False, False])
            self.player_data = self.player_data.groupby(['Player', 'Year']).first().reset_index()           
            logger.info(f"Player data loaded and preprocessed. Shape: {self.player_data.shape}")
            logger.debug(f"Player data columns: {self.player_data.columns}")
            logger.debug(f"First few rows of preprocessed player data:\n{self.player_data.head()}")
        except Exception as e:
            logger.error(f"Failed to load or preprocess player data: {str(e)}")
            raise RuntimeError(f"Failed to load or preprocess player data: {str(e)}")

    def calculate_player_averages(self, player_names:list) -> pd.DataFrame:
        """
        This function calculates the averages of NBA player statistics for one or more players,
        including those with only one year of data.

        Returns: 
        player_averages: a dataframe of the player averages from the last 3 years, last 2 years for players with exactlt 2 years of available data 
        and last 1 year for players with one year of available data(rookies)
        """
        # Ensure player_names is a list
        player_names = [player_names] if isinstance(player_names, str) else player_names

        # Filter the preprocessed data for the requested players
        player_data = self.player_data[self.player_data['Player'].isin(player_names)]

        # Ensure all columns in self.cols are present in player_data
        missing_cols = set(self.cols) - set(player_data.columns)
        if missing_cols:
            print(f"Columns {missing_cols} not found in player data. They will be excluded.")
            cols_to_use = [col for col in self.cols if col in player_data.columns]
        else:
            cols_to_use = self.cols

        def calc_player_stats(group):
            # Sort by Year descending
            sorted_group = group.sort_values('Year', ascending=False)
            # Take up to 3 most recent years, or all available if less than 3
            recent_years = sorted_group.iloc[:3]
            # Calculate mean of available data
            return recent_years[cols_to_use].mean()

        # Apply the calculation to each player
        player_averages = player_data.groupby('Player').apply(calc_player_stats).reset_index()

        # Round numeric columns to 2 decimal places
        player_averages[cols_to_use] = player_averages[cols_to_use].round(2)

        # Check for truly missing players i.e unavailable players
        missing_players = set(player_names) - set(player_averages['Player'])
        if missing_players:
            print(f"No data found for players: {missing_players}")
            missing_df = pd.DataFrame({'Player': list(missing_players)})
            player_averages = pd.concat([player_averages, missing_df], ignore_index=True)

        print(f"Calculated averages for {len(player_averages)} players")
        return player_averages

    def predict(self, context, model_input:pd.DataFrame) -> pd.DataFrame:
        """
        This function defines the predict context of the custom model, routing it to the base XGBoost Model

        Returns:
        result_df: a dataframe of input player names and their corresponding PPG predictions
        """
        logger.info(f"Received model_input: {model_input}")        
        if isinstance(model_input, pd.DataFrame):
            if 'Player' not in model_input.columns:
                raise ValueError("Input DataFrame must have a 'Player' column")
            player_names = model_input['Player'].tolist()
        elif isinstance(model_input, list):
            player_names = model_input
        elif isinstance(model_input, str):
            player_names = [model_input]
        else:
            raise ValueError("Input must be a DataFrame, list, or string")      
        logger.info(f"Processing players: {player_names}")
       
        try:
            input_data = self.calculate_player_averages(player_names)
            logger.debug(f"Preprocessed input data shape: {input_data.shape}")
            logger.debug(f"Preprocessed input data columns: {input_data.columns}")
        except Exception as e:
            logger.error(f"Error calculating player statistics: {str(e)}")
            raise ValueError(f"Error calculating player statistics: {str(e)}")        
        result_df = pd.DataFrame({'Player': player_names})
        
        try:
            # Identify players with available data
            available_players = input_data[input_data[self.cols].notna().any(axis=1)]['Player']
            missing_players = set(player_names) - set(available_players)

            if not available_players.empty:
                # Predict for available players
                model_input_data = input_data[input_data['Player'].isin(available_players)][self.cols]
                predictions = self.base_model.predict(model_input_data)
                predictions = np.round(predictions, decimals=2)
                predictions = np.array(predictions, dtype=np.float32)
                
                # Create result DataFrame for available players
                available_result = pd.DataFrame({
                    'Player': available_players,
                    'Predicted PPG': predictions.flatten()
                })
                available_result['Predicted PPG'] = available_result['Predicted PPG'].apply(lambda x: f"{x:.2f}")
                
                # Merge with the result DataFrame
                result_df = result_df.merge(available_result, on='Player', how='left')
            
            # Handle missing players
            result_df.loc[result_df['Player'].isin(missing_players), 'Predicted PPG'] = "Player data not available"
            
            logger.info(f"Predictions made successfully. Shape: {result_df.shape}")
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise RuntimeError(f"Error making predictions: {str(e)}")

        logger.info(f"Final prediction DataFrame:\n{result_df}")
        
        return result_df
    
MODEL_NAME = "NBA_XGB"
MODEL_VERSION = "1" 

mlflow.set_tracking_uri("databricks")

experiment_id = mlflow.get_experiment_by_name("/Users/***.***@***.com/NBA_XGB").experiment_id
# Start an MLflow run
with mlflow.start_run(experiment_id= experiment_id, run_name="XGBoost_Model_Final") as run:
    # Create an instance of the custom model
    custom_model = PlayerPredictionModel(MODEL_NAME, MODEL_VERSION)

    # Save the custom model
    mlflow.pyfunc.save_model(
        path="Custom_Model",
        python_model=custom_model,
        input_example=pd.DataFrame({"Player": ["LeBron James", "Stephen Curry"]}),
        signature=mlflow.models.ModelSignature(
            inputs=mlflow.types.Schema([mlflow.types.ColSpec(type="string", name="Player")]),
            outputs=mlflow.types.Schema([mlflow.types.TensorSpec(type=np.dtype('float32'), shape=(-1,))])),
        conda_env = "C:/hoops_ml/conda.yaml"
    )    
    # Log the model
    mlflow.pyfunc.log_model(
        artifact_path="Custom_Model",
        python_model=custom_model,
        input_example=pd.DataFrame({"Player": ["LeBron James", "Stephen Curry"]}),
        signature=mlflow.models.ModelSignature(
            inputs=mlflow.types.Schema([mlflow.types.ColSpec(type="string", name="Player")]),
            outputs=mlflow.types.Schema([mlflow.types.TensorSpec(type=np.dtype('float32'), shape=(-1,))])),
        registered_model_name="NBA_XGB_Final",
        conda_env = "C:/hoops_ml/conda.yaml"
    )