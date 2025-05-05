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
# right after you load df and before dropping lat/long
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

# Axes for the PCP toggle menu
AXIS_OPTS = [
  'price','service_fee','minimum nights','number of reviews',
  'reviews per month','review rate number','calculated host listings count',
  'availability 365','score','instant_bookable','cancellation_policy','room type'
]
NUM_AXES = 5

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
app.layout = html.Div(
    style={'display':'flex','height':'100vh'},
    children=[

        # — Sidebar (filters only) —
        html.Div(
            style={'width':'260px','padding':'20px','background':'#f8f9fa'},
            children=[
                html.H3("Filters"),
                html.Label("Price range"),
                dcc.RangeSlider(
                    id='price-slider',
                    min=df['price'].min(),
                    max=df['price'].max(),
                    step=100,
                    value=[df['price'].quantile(0.05), df['price'].quantile(0.95)]
                ),
                html.Br(),
                html.Label("Neighbourhood group"),
                dcc.Dropdown(
                    id='neigh-dropdown',
                    options=[{'label':g,'value':g} for g in sorted(df['neighbourhood group'].dropna().unique())],
                    value=[],
                    multi=True,
                    placeholder="All groups"
                ),
                html.Br(),
                html.Label("Room type"),
                dcc.Checklist(
                    id='roomtype-checklist',
                    options=[{'label':rt,'value':rt} for rt in df['room type'].unique()],
                    value=df['room type'].unique().tolist()
                ),
            ]
        ),

        # — Main content (map + axis selectors + PCP) —
        html.Div(
            style={'flex':'1','padding':'20px','overflowY':'auto'},
            children=[
                html.H1("NYC Airbnb Explorer", style={'textAlign':'center'}),

                # Map
                dcc.Graph(id='map-graph'),
                html.Hr(),

                # Axis-pickers in a single row
                html.Div(
                    style={'display':'flex','gap':'10px','marginBottom':'20px'},
                    children=[
                       html.Div(
                            style={'display':'flex','gap':'10px','marginBottom':'20px'},
                            children=[
                                # Axes 1–5
                                *[
                                    html.Div([
                                        html.Label(f"Axis {i}"),
                                        dcc.Dropdown(
                                            id=f'axis-{i}',
                                            options=[{'label': c, 'value': c} for c in AXIS_OPTS],
                                            value=AXIS_OPTS[i-1],
                                            clearable=False
                                        )
                                    ], style={'flex': '1', 'minWidth': '150px'})
                                    for i in range(1, 6)
                                ]
                            ]
                        ),
                    ]
                ),

                # PCP
                dcc.Graph(id='pcp-graph')
            ]
        )

    ]
) 

# --- Callback to filter data and update both graph & pcp ---
@app.callback(
  [Output('map-graph','figure'),
   Output('pcp-graph','figure')],
  [Input('price-slider','value'),
   Input('neigh-dropdown','value'),
   Input('roomtype-checklist','value')] +
  [Input(f'axis-{i+1}','value') for i in range(5)]
)
def update_outputs(price_range, selected_groups, selected_rooms,
                   axis1, axis2, axis3, axis4, axis5):
    
    axes = [axis1, axis2, axis3, axis4, axis5]

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

    # --- parallel coordinates ---
    # ensure all chosen axes exist and drop NaNs
    # 1) Only keep the first occurrence of each axis and ensure it's in the DF
    dims = []
    for ax in axes:
        if ax in dff.columns and ax not in dims:
            dims.append(ax)

    # 2) Pad with a safe default (e.g. 'price') so we always have 5
    while len(dims) < NUM_AXES:
        dims.append('price')

    pcp = px.parallel_coordinates(
    dff.dropna(subset=dims),
    dimensions=dims,
    color='score',
    color_continuous_scale=px.colors.sequential.Viridis
    )
    pcp.update_layout(margin={'t':20,'b':20,'l':20,'r':20})

    fig.update_layout(
        mapbox_style='open-street-map',
        mapbox_center={'lat': df['lat'].mean(), 'lon': df['long'].mean()},
        mapbox_zoom=10,
        margin={'l':0,'r':0,'t':30,'b':0}
    )

    return fig, pcp

if __name__ == '__main__':
    app.run(debug=True)