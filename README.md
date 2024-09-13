# hoops_ml
 # NBA Points Per Game Prediction

This project aims to predict NBA players' Points Per Game (PPG) using historical data and advanced statistics. It utilizes web scraping, data preprocessing, machine learning with XGBoost, and MLflow for experiment tracking and model management.

## Project Structure

1. Data Collection
   - `RegSznAdv.py`: Scrapes and processes advanced NBA player statistics
   - `RegSznPerGame.py`: Scrapes and processes per-game NBA player statistics

2. Data Preprocessing
   - `prep.py`: Merges and preprocesses the collected data

3. Model Training
   - `train.py`: Trains an XGBoost model using the preprocessed data

4. Inference
   - `inference.py`: Implements custom inference logic for the trained model

5. Testing
   - `test.py`: Tests the deployed model with sample input

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/K0405Y/hoops_ml.git
   cd hoops_ml
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Databricks workspace and configure your environment variables for authentication.

## Usage

### Data Collection

Run the data collection scripts to gather NBA player statistics:

```
python RegSznAdv.py
python RegSznPerGame.py
```

### Data Preprocessing

Preprocess the collected data:

```
python prep.py
```

### Model Training

Train the XGBoost model:

```
python train.py
```

This script will use MLflow to track experiments and log the best model.

### Inference

To build the custom inference model:

```
python inference.py
```

This script sets up a custom MLflow model for inference, which can be used to predict PPG for given NBA players.

### Testing the Model

To test the deployed model with sample input:

```
python test.py
```

This script loads the deployed model from the Databricks Model Registry and makes predictions using sample player names.

## Model Details

The project uses an XGBoost regressor model to predict NBA player PPG. The model is trained on historical player data and advanced statistics. Hyperparameter tuning is performed using RandomizedSearchCV.

## MLflow Integration

This project heavily utilizes MLflow for experiment tracking, model versioning, and deployment. The trained models are logged to the Databricks Model Registry for easy management and deployment.

## Custom Inference

The `PlayerPredictionModel` class in `inference.py` provides custom inference logic, allowing for predictions based on player names. It handles data preprocessing, missing player data, and integrates with the base XGBoost model.