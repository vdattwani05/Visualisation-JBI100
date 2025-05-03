from dash import dash, dcc, html, dash_table
import plotly.graph_objects as go
import pandas as pd

# ---------------------------------------------------------------
# This is a Dash app that displays a DataTable with data from an Airbnb dataset.
# The dataset is read from a CSV file, and some columns are dropped.
df = pd.read_csv('Airbnb_Open_Data.csv')
df = df.drop(columns=['house_rules', 'country code', 'country', 'license', 'Construction year'])
df.set_index('id', inplace=True)
# Remove the '$' sign and convert the price column to float
df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
# Removing missing values from lat and long
df = df.dropna(subset=['lat', 'long'])
df.rename(columns={df.columns[df.columns.str.strip() == 'service fee'][0]: 'service_fee'}, inplace=True)
df['service_fee'] = df['service_fee'].replace('[\$,]', '', regex=True).astype(float)
# First, let's ensure the columns are properly formatted
df['number of reviews'] = pd.to_numeric(df['number of reviews'], errors='coerce')
df['review rate number'] = pd.to_numeric(df['review rate number'], errors='coerce')
# Create a score column that combines review count and rating
max_reviews = df['number of reviews'].max()
df['score'] = (df['number of reviews'] / max_reviews * 0.5 + df['review rate number'] / 5 * 0.5) * 100
df['score'] = df['score'].round(2)
df['score'] = df['score'].fillna(0)
# ---------------------------------------------------------------

# Create a Dash app instance

app = dash.Dash(__name__)
app.title = "New York AirBnb"
server = app.server

if __name__ == '__main__':
    app.run_server(debug=True)