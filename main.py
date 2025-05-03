from dash import dash, dcc, html, dash_table, Input, Output
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Print total rows in the dataset
print("Total rows in the dataset: ", len(pd.read_csv('Airbnb_Open_Data.csv')))
# ---------------------------------------------------------------
# This is a Dash app that displays a DataTable with data from an Airbnb dataset.
# The dataset is read from a CSV file, and some columns are dropped.
df = pd.read_csv('Airbnb_Open_Data.csv', low_memory=False)
df = df.drop(columns=['house_rules', 'country code', 'country', 'license', 'Construction year'])
df.set_index('id', inplace=True)
# Remove the '$' sign and convert the price column to float
df['price'] = df['price'].replace(r'[\$,]', '', regex=True).astype(float)
# Removing missing values from lat and long
df = df.dropna(subset=['lat', 'long'])
# Print total rows in the dataset after dropping missing values
print("Total rows in the dataset after dropping missing values: ", len(df))
df.rename(columns={df.columns[df.columns.str.strip() == 'service fee'][0]: 'service_fee'}, inplace=True)
df['service_fee'] = df['service_fee'].replace(r'[\$,]', '', regex=True).astype(float)
# First, let's ensure the columns are properly formatted
df['number of reviews'] = pd.to_numeric(df['number of reviews'], errors='coerce')
df['review rate number'] = pd.to_numeric(df['review rate number'], errors='coerce')
# Create a score column that combines review count and rating
max_reviews = df['number of reviews'].max()
df['score'] = (df['number of reviews'] / max_reviews * 0.5 + df['review rate number'] / 5 * 0.5) * 100
df['score'] = df['score'].round(2)
df['score'] = df['score'].fillna(0)
df.rename(columns={'NAME': 'name'}, inplace=True)
df['hover_text'] = df.apply(
    lambda r: f"{r['name']}<br>Price: ${r['price']:.0f}<br>Score: {r['score']}",
    axis=1
)
# ---------------------------------------------------------------

cluster_steps  = [10,  50, 100]               # up to 10, up to 50, 50+
cluster_sizes  = [10,  20,  30]               # px diameter of each bubble
cluster_colors = ['lightblue', 'lightgreen','lightcoral']

point_color = 'red'  # color of individual points 
point_size = 8   # px diameter of individual points

# Build map figure using Plotly Express
fig_map = go.Figure(go.Scattermapbox(
    lat=df['lat'],
    lon=df['long'],
    # no per‐point text here—clusters will auto‐label themselves with counts
    mode='markers',

    # regular markers for individual points
    marker=go.scattermapbox.Marker(
        size=point_size,
        color=point_color,
        showscale=False
    ),

    # CLUSTER styling
    cluster=go.scattermapbox.Cluster(
        enabled=True,
        step=cluster_steps,
        size=cluster_sizes,
        color=cluster_colors,
        maxzoom=10,
        opacity=0.8
    )
))

fig_map.update_layout(
    mapbox_style='open-street-map',
    mapbox_center={'lat': df['lat'].mean(), 'lon': df['long'].mean()},
    mapbox_zoom=15,
    margin={'l':0,'r':0,'t':30,'b':0},
    title='NYC Airbnb Listings (clustered bubbles)'
)
# Create a Dash app instance
app = dash.Dash(__name__)
app.title = "New York AirBnb"
server = app.server

# --- App layout with sidebar + main area ---
app.layout = html.Div(style={'display': 'flex', 'height': '100vh'}, children=[

    # --- Sidebar (left) ---
    html.Div(style={
        'width': '250px',
        'padding': '20px',
        'backgroundColor': '#f8f9fa',
        'boxShadow': '2px 0 5px rgba(0,0,0,0.1)'
    }, children=[
        html.H3("Filters"),
        html.Label("Price range"),
        dcc.RangeSlider(
            id='price-slider',
            min=df['price'].min(),
            max=df['price'].max(),
            step=100,
            value=[df['price'].quantile(0.05), df['price'].quantile(0.95)],
            tooltip={'always_visible': False, 'placement': 'bottom'}
        ),
        html.Br(),
        html.Label("Neighbourhood group"),
       dcc.Dropdown(
            id='neigh-dropdown',
            options=[
                {'label': grp, 'value': grp}
                for grp in sorted(df['neighbourhood group'].dropna().unique())
                if isinstance(grp, str) and grp.strip() != ""
            ],
            multi=True,
            value=[],
            placeholder="All groups"
        ),
        html.Br(),
        html.Label("Room type"),
        dcc.Checklist(
            id='roomtype-checklist',
            options=[{'label': rt, 'value': rt} for rt in df['room type'].unique()],
            value=df['room type'].unique().tolist()
        )
    ]),

    # --- Main content (right) ---
    html.Div(style={'flex': '1', 'padding': '20px', 'overflowY': 'auto'}, children=[
        html.H1("NYC Airbnb Map and Listings", style={'textAlign':'center'}),
        dcc.Graph(id='map-graph'),
        html.H2("Listings Table", style={'marginTop': '2rem'}),
        dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i} for i in df.reset_index().columns],
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '5px'},
        )
    ])
])
# --- Callback to filter data and update both graph & table ---
@app.callback(
    [Output('map-graph', 'figure'),
     Output('table',      'data')],
    [Input('price-slider',      'value'),
     Input('neigh-dropdown',    'value'),
     Input('roomtype-checklist','value')]
)
def update_outputs(price_range, selected_groups, selected_rooms):
    low, high = price_range

    # Filter step by step
    dff = df[(df['price'] >= low) & (df['price'] <= high)]
    if selected_groups:
        dff = dff[dff['neighbourhood group'].isin(selected_groups)]
    if selected_rooms:
        dff = dff[dff['room type'].isin(selected_rooms)]

    # Map figure (clustered)
    fig = go.Figure(go.Scattermapbox(
        lat=dff['lat'],
        lon=dff['long'],
        mode='markers',
        text=dff.apply(lambda r: f"{r['name']}<br>${r['price']:.0f}<br>Score: {r['score']}", axis=1),
        hoverinfo='text',
        marker=go.scattermapbox.Marker(
            size=point_size,
            color=point_color,
            showscale=False
        ),
        cluster=go.scattermapbox.Cluster(
            enabled=True,
            step=cluster_steps,
            size=cluster_sizes,
            color=cluster_colors,
            maxzoom=15,
            opacity=0.8
        )
    ))
    fig.update_layout(
        mapbox_style='open-street-map',
        mapbox_center={'lat': df['lat'].mean(), 'lon': df['long'].mean()},
        mapbox_zoom=10,
        margin={'l':0,'r':0,'t':30,'b':0}
    )

    # Table data
    table_data = dff.reset_index().to_dict('records')
    return fig, table_data

if __name__ == '__main__':
    app.run(debug=True)