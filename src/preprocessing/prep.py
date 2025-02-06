import pandas as pd
import pickle as pkl
from sklearn.preprocessing import LabelEncoder
from databricks.sdk import WorkspaceClient
import io
import numpy as np

#function to import two files needed and to append them
def merge_data() -> pd.DataFrame:
    """
    Imports and merges regular season per-game player data with advanced stats.
    The function:
    1. Reads player stats from two CSV files
    2. Cleans and merges the datasets
    3. Uploads the merged data to Databricks FileStore    
    Returns:
        pd.DataFrame: Merged and cleaned player statistics   
    Raises:
        FileNotFoundError: If source CSV files don't exist
        pd.errors.EmptyDataError: If CSV files are empty
        Exception: For other processing errors
    """
    try:
        # Define file paths
        per_game_stats_path = "C:/hoops_ml/data/raw/Regular Season Player PerGame Stats/PlayerPerGameStats04-24.csv"
        advanced_stats_path = "C:/hoops_ml/data/raw/Regular Season Player Advanced Stats/PlayerAdvancedStats04-24.csv"
        
        # Read CSV files
        per_game_stats = pd.read_csv(per_game_stats_path)
        advanced_stats = pd.read_csv(advanced_stats_path)
        
        # Log initial data shapes for validation
        print(f"Per game stats shape: {per_game_stats.shape}")
        print(f"Advanced stats shape: {advanced_stats.shape}")
        
        # Clean per-game stats
        per_game_stats = (per_game_stats
            .drop(['Year', 'Team', 'Rk', 'Awards'], axis=1)
            .query('Player != "League Average"'))
        
        # Clean advanced stats
        advanced_stats = advanced_stats.drop(
            ['Age', 'MP', 'Player', 'Pos', 'Age', 'Tm', 'G', 'Rk'], 
            axis=1
        )
        # Merge datasets
        merged_stats = pd.concat([per_game_stats, advanced_stats], axis=1)
        
        # Upload to Databricks FileStore
        workspace = WorkspaceClient()
        csv_buffer = io.BytesIO()
        merged_stats.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        workspace.dbfs.upload(
            src=csv_buffer,
            path="dbfs:/FileStore/inference_data.csv",
            overwrite=True
        )
        return merged_stats       
    except FileNotFoundError as e:
        print(f"Error: Source file not found - {e}")
        raise
    except pd.errors.EmptyDataError as e:
        print(f"Error: One or both CSV files are empty - {e}")
        raise
    except Exception as e:
        print(f"Error during data processing: {e}")
        raise

def preprocess_data(data:pd.DataFrame, correlation_threshold=0.6) -> pd.DataFrame:
    """ 
        This function calculates the selects features by calculating the correlation coefficients
        and selecting the features above 0.6

        Args: 
        data(pd.DataFrame) : merged and cleaned data
        correlation_threshold(float) : selection criterion

        Returns:
        final_data: a datafame of the features selected
    """
    try:
        # Create a copy of the DataFrame 
        data = data.copy()

        # Drop NaN values
        data.dropna(inplace=True)

        # List of numeric features
        numeric_features = data.select_dtypes(include=['number']).columns.tolist()

        # Remove 'Year' and 'PTS' columns
        exclude_cols = ['Year', 'PTS']
        numeric_features = [col for col in numeric_features if col not in exclude_cols]

        # Get correlation matrix
        cor = data[numeric_features + ['PTS']].corr()

        # Convert 'Pos' column to category type and apply label encoding
        if 'Pos' in data.columns:
            data['Pos'] = data['Pos'].astype('category')
            label_encoder = LabelEncoder()
            data.loc[:, 'Pos'] = label_encoder.fit_transform(data['Pos'])

        # Fill missing values with median
        data.loc[:, numeric_features] = data[numeric_features].fillna(data[numeric_features].median())

        # Filter columns with highly positive correlations with PTS, excluding PTS itself
        positive_corr_cols = cor['PTS'][(cor['PTS'] > correlation_threshold) & (cor.index != 'PTS')].index.tolist()

        # Serialize the selected columns
        pickle_bytes = io.BytesIO()
        pkl.dump(positive_corr_cols, pickle_bytes)
        pickle_bytes.seek(0)

        # Define the DBFS path
        dbfs_path = "/FileStore/features.pkl"

        # Upload the pickle bytes to DBFS
        workspace = WorkspaceClient()
        workspace.dbfs.upload(src=pickle_bytes, path=dbfs_path, overwrite=True)

        # Concatenate the selected columns with 'PTS' 
        final_data = data[positive_corr_cols + ['PTS']]
        # Save to CSV for random checks
        final_data.to_csv("C:/hoops_ml/data/features/data.csv", index=False)
        return final_data
    
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        print(f"Data shape: {data.shape}")
        print(f"Data types:\n{data.dtypes}")
        raise
