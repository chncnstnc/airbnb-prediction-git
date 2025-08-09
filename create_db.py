import pandas as pd
import sqlite3

# Step 1: Load your CSV
df = pd.read_csv("bostonlistings.csv")

# Step 3: Manually match confirmed column names
df = df[[
    'id', 'name', 'host_id', 'host_name',
    'neighbourhood', 'room_type', 'price',
    'minimum_nights', 'number_of_reviews', 'availability_365'
]]

# Step 4: Clean price column
df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)

# Step 5: Normalize host table
hosts_df = df[['host_id', 'host_name']].drop_duplicates()
df = df.drop(columns=['host_name'])

# Step 6: Create SQLite DB
conn = sqlite3.connect("airbnb.db")
cursor = conn.cursor()

# Step 7: Load and execute SQL from external file
with open("airbnb.sql", "r") as f:
    schema_sql = f.read()

cursor.executescript(schema_sql)

# Step 8: Insert data
hosts_df.to_sql("Host", conn, if_exists='append', index=False)
df.to_sql("Listing", conn, if_exists='append', index=False)

# Step 9: Done
conn.commit()
conn.close()