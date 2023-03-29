# -*- coding: utf-8 -*-
import pandas as pd
import sqlite3
import os


# Load the iris dataset from scikit-learn
from sklearn.datasets import load_iris
iris = load_iris()

# Convert the iris dataset to a pandas dataframe
df = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
df['Species'] = iris['target']

# Rename the columns in df
df = df.rename(columns={'sepal length (cm)': 'sepal_length', 'sepal width (cm)': 'sepal_width',
                        'petal length (cm)': 'petal_length', 'petal width (cm)': 'petal_width'})


# Add a new column called 'item_id'
df['item_id'] = range(1, len(df) + 1)

database_folder = 'database'
if not os.path.exists(database_folder):
    os.makedirs(database_folder)

# Create a SQLite database and connect to it
conn = sqlite3.connect('database/iris.db')

# Create the ident_table
df_ident = df[['item_id', 'Species']]
df_ident.to_sql('ident_table', conn, index=False, if_exists='replace')

# Create the sepal_table
df_sepal = df[['item_id', 'sepal_width', 'sepal_length']]
df_sepal.to_sql('sepal_table', conn, index=False, if_exists='replace')

# Create the petal_table
df_petal = df[['item_id', 'petal_width', 'petal_length']]
df_petal.to_sql('petal_table', conn, index=False, if_exists='replace')

# Close the database connection
conn.close()
