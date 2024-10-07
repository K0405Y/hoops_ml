import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO
from unidecode import unidecode

# Function to get NBA player stats for a given year
def get_nba_player_stats(year:int) -> pd.DataFrame:
    """
    Function to extract player regular season per game data from basketball reference
    
    Args:
    year (int): The year in focus
    
    Returns:
    df: A dataframe of scraped data
    
    """
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html"
    response = requests.get(url)
    response.encoding = 'utf-8' #to address player names with special characters
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract the table
    table = soup.find('table', {'id': 'per_game_stats'})

    # lxml parser for better handling of special characters
    df = pd.read_html(StringIO(str(table)), encoding='utf-8', flavor='lxml')[0]

    # Remove rows where the 'Age' column contains 'Age'
    df = df[df['Age'] != 'Age']

    # Clean player names
    df['Player'] = df['Player'].apply(lambda x: unidecode(x) if pd.notnull(x) else x)

    return df

# Initialize a dictionary to store data for each year
all_stats = {}

# Loop through years from 2004 to 2024
for year in range(2004, 2025):
    stats_for_year = get_nba_player_stats(year)
    stats_for_year['Year'] = year  # Add a 'Year' column
    all_stats[year] = stats_for_year
    print(f"Player pergame stats for year {year} extracted")

# Combine all data frames into a single data frame
PlayerPerGameStats = pd.concat(all_stats.values(), ignore_index=True)

# Rename columns for better structure
PlayerPerGameStats.rename(columns={
    "FG%": "FG.",
    "3PM": "3P",
    "3PA": "3PA",
    "3P%": "3P.",
    "2PM": "2P",
    "2PA": "2PA",
    "2P%": "2P.",
    "eFG%": "eFG.",
    "FT%": "FT."
}, inplace=True)

# Extract the first position before the hyphen to standardize positions
PlayerPerGameStats['Pos'] = PlayerPerGameStats['Pos'].str.split('-').str[0]

#set directory path
directory_path = "C:\\hoops_ml\\data\\raw\\Regular Season Player PerGame Stats"

if not os.path.exists(directory_path):
    try:
        os.makedirs(directory_path)
        print(f"Directory {directory_path} created successfully.")
    except Exception as e:
        print(f"Failed to create directory {directory_path}. Error: {e}")
else:
    print(f"Directory {directory_path} already exists.")

csv_filename = os.path.join(directory_path, "PlayerPerGameStats04-24.csv")

try:
    PlayerPerGameStats.to_csv(csv_filename, encoding='latin1', index=False)
    print(f"Data successfully written to {csv_filename}.")
except Exception as e:
    print(f"Failed to write data to {csv_filename}. Error: {e}")