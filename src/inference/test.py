import mlflow
import mlflow.pyfunc
import pandas as pd

mlflow.set_tracking_uri("databricks")

# Define the path to the model you want to test
model_uri = "models:/NBA_XGB_Final/1"  # Assuming version 1 of the model

# Load the model
model = mlflow.pyfunc.load_model(model_uri)

# Create a sample input dataframe
sample_input = pd.DataFrame({
    "Player": ["LeBron James", "Stephen Curry"]  # Replace with the names you want to test
})

# Run the prediction
try:
    predictions = model.predict(sample_input)
    print("Predictions:")
    print(predictions)
except ValueError as e:
    print(f"Error during prediction: {e}")
