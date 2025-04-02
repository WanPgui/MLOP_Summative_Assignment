import sqlite3
import pandas as pd

# Load CSV
df = pd.read_csv('/app/diabetic_data.csv')

# Connect to SQLite database (it will create the database if it doesn't exist)
conn = sqlite3.connect('/app/diabetic_data.db')

# Save dataframe to SQLite table
df.to_sql('diabetic_data', conn, if_exists='replace', index=False)

# Close the connection
conn.close()