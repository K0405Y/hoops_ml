import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO
from unidecode import unidecode

# Function to get NBA player advanced stats for a given year
def get_nba_player_advanced_stats(year:int) -> pd.DataFrame:    
    """
    Function to extract player regular season advanced stats data from basketball reference
    
    Args:
    year (int): The year in focus
    
    Returns: 
    df: A dataframe of scraped data
    
    """
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
    response = requests.get(url)
    response.encoding = 'utf-8' #to address player names with special characters
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract the table
    table = soup.find('table', {'id': 'advanced_stats'})

    # lxml parser for better handling of special characters
    df = pd.read_html(StringIO(str(table)), encoding='utf-8', flavor='lxml')[0]

    # Remove rows where the 'Age' column contains 'Age'
    df = df[df['Age'] != 'Age']

    # Clean player names
    df['Player'] = df['Player'].apply(lambda x: unidecode(x) if pd.notnull(x) else x)

    return df

# Initialize a dictionary to store data for each year
all_advanced_stats = {}

# Loop through years 2004 to 2024
for year in range(2004, 2025):
    advanced_stats_for_year = get_nba_player_advanced_stats(year)
    advanced_stats_for_year['Year'] = year  # Add a 'Year' column
    all_advanced_stats[year] = advanced_stats_for_year
    print(f"Player advanced stats for year {year} extracted")

# Combine all data frames into a single data frame
PlayerAdvancedStats = pd.concat(all_advanced_stats.values(), ignore_index=True)

# Remove columns with all NA values
PlayerAdvancedStats.dropna(axis=1, how='all', inplace=True)

# Rename columns
PlayerAdvancedStats.rename(columns={
    "TS%": "TS.",
    "3PAr": "3PAr",
    "ORB%": "ORB.",
    "DRB%": "DRB.",
    "TRB%": "TRB.",
    "AST%": "AST.",
    "STL%": "STL.",
    "BLK%": "BLK.",
    "TOV%": "TOV."
}, inplace=True)

# Extract the first position before the hyphen to standardize positions
PlayerAdvancedStats['Pos'] = PlayerAdvancedStats['Pos'].str.split('-').str[0]

# Define the directory path
directory_path = "C:\\hoops_ml\\data\\raw\\Regular Season Player Advanced Stats"

# Check if the directory exists
if not os.path.exists(directory_path):
    try:
        os.makedirs(directory_path)
        print(f"Directory {directory_path} created successfully.")
    except Exception as e:
        print(f"Failed to create directory {directory_path}. Error: {e}")
else:
    print(f"Directory {directory_path} already exists.")

# Define the file path for the CSV file
csv_filename = os.path.join(directory_path, "PlayerAdvancedStats04-24.csv")

# Save the data frame to a CSV file
try:
    PlayerAdvancedStats.to_csv(csv_filename, index=False)
    print(f"Data successfully written to {csv_filename}.")
except Exception as e:
    print(f"Failed to write data to {csv_filename}. Error: {e}")