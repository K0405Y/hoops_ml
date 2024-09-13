import pandas as pd
import pickle as pkl
from sklearn.preprocessing import LabelEncoder
from databricks.sdk import WorkspaceClient
import io
import numpy as np

#function to import two files needed and to append them
def merge_data():
    """
        This function imports regular season per game player data and advanced stats,
         merges them into one with a some cleaning steps.
        """
    #set file paths
    file_path1 = "C:/hoops_ml/data/raw/Regular Season Player PerGame Stats/PlayerPerGameStats04-24.csv"
    file_path2 = "C:/hoops_ml/data/raw/Regular Season Player Advanced Stats/PlayerAdvancedStats04-24.csv"
    data1 = pd.read_csv(file_path1, encoding="latin1")
    data2 = pd.read_csv(file_path2, encoding="latin1")
    # Drop unnecessary columns
    data1 = data1.drop(['Year', 'Team', 'Rk', 'Awards'], axis=1)
    data1 = data1[data1['Player'] != 'League Average']
    data2 = data2.drop(['Age', 'MP', 'Player', 'Pos', 'Age', 'Tm', 'G', 'Rk'], axis=1)
    #validate data
    print(data1.shape)
    print(data2.shape)
    # Concatenate the dataframes
    data = pd.concat([data1, data2], axis=1)
    #initilaise databricks client to write to dbfs
    workspace = WorkspaceClient()
    csv_bytes = io.BytesIO()
    data.to_csv(csv_bytes, index=False, encoding='latin1')
    csv_bytes.seek(0)  # Rewind the BytesIO object to the beginning
    #upload file as inference data to dbfs
    workspace.dbfs.upload(src= csv_bytes, path="dbfs:/FileStore/inference_data.csv", overwrite= True)
    return data

def preprocess_data(data, correlation_threshold=0.6):
    """
        This function calculates the selects features by calculating the correlation coefficients
        and selecting the features above 0.6
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