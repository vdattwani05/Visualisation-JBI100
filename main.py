from dash import Dash, dcc, html, Input, Output, dash_table, callback_context
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots


# ===============================================================================
# DATA LOADING AND PREPROCESSING
# ===============================================================================

def load_and_clean_data():
    """
    Loads Airbnb data from CSV and performs cleaning operations:
    - Drops unnecessary columns
    - Renames columns for consistency
    - Cleans price and service_fee columns
    - Converts columns to appropriate types
    - Creates score column combining review count and rating
    - Creates hover text for map visualization

    Returns:
        pandas.DataFrame: Cleaned and preprocessed Airbnb data
    """
    # Load the raw data file
    # Note: Ensure 'Airbnb_Open_Data.csv' is in the same directory as this script
    df = pd.read_csv('Airbnb_Open_Data.csv', low_memory=False)

    # Remove unnecessary columns that aren't relevant for visualization or analysis
    df = df.drop(columns=['house_rules', 'country code', 'country', 'license'])

    # Standardize column names for consistency
    df.rename(columns={'NAME': 'name'}, inplace=True)

    # Set the listing ID as the index for faster lookups
    df.set_index('id', inplace=True)

    # Clean price: Remove dollar signs and commas, convert to float
    df['price'] = df['price'].replace(r'[\$,]', '', regex=True).astype(float)

    # Fix the service fee column name (which might have extra spaces) and clean it
    df.rename(columns={df.columns[df.columns.str.strip() == 'service fee'][0]: 'service_fee'}, inplace=True)
    df['service_fee'] = df['service_fee'].replace(r'[\$,]', '', regex=True).astype(float)

    # Remove rows with missing essential data
    # Spatial coordinates, neighborhood group and room type are critical for visualization
    df = df.dropna(subset=['lat', 'long', 'neighbourhood group', 'room type'])

    # Convert various columns to numeric types, using coerce to handle non-numeric values
    # This ensures proper calculations and visualizations
    df['number of reviews'] = pd.to_numeric(df['number of reviews'], errors='coerce')
    df['review rate number'] = pd.to_numeric(df['review rate number'], errors='coerce')
    df['minimum nights'] = pd.to_numeric(df['minimum nights'], errors='coerce')
    df['reviews per month'] = pd.to_numeric(df['reviews per month'], errors='coerce')
    df['availability 365'] = pd.to_numeric(df['availability 365'], errors='coerce')
    df['calculated host listings count'] = pd.to_numeric(df['calculated host listings count'], errors='coerce')

    # Clip availability to valid range (0-365 days)
    df['availability 365'] = df['availability 365'].clip(0, 365)

    # Create composite score as a weighted combination of review count and rating
    # Formula: 50% of normalized review count + 50% of normalized rating, on a 0-100 scale
    max_reviews = df['number of reviews'].max()
    df['score'] = (df['number of reviews'] / max_reviews * 0.5 + df['review rate number'] / 5 * 0.5) * 100
    df['score'] = df['score'].round(2)  # Round to 2 decimal places for cleaner display
    df['score'] = df['score'].fillna(0)  # Handle listings with no reviews

    # Create hover text for map visualization that combines key listing information
    df['hover_text'] = df.apply(
        lambda r: f"{r['name']}<br>Price: ${r['price']:.0f}<br>Score: {r['score']}<br>Room type: {r['room type']}",
        axis=1
    )

    # Normalize borough names: capitalize and fix common typos
    df['neighbourhood group'] = df['neighbourhood group'].str.capitalize()
    df['neighbourhood group'] = df['neighbourhood group'].replace('Manhatan', 'Manhattan')

    return df


# Load and process the dataset
df = load_and_clean_data()

# ===============================================================================
# VISUALIZATION CONFIGURATION
# ===============================================================================

# Map cluster configuration for different zoom levels
# These parameters control the appearance of clustered points on the map
cluster_steps = [10, 50, 100]  # Thresholds for different cluster sizes
cluster_sizes = [10, 20, 30]  # Visual size of clusters based on threshold
cluster_colors = ['#66c2a5', '#8da0cb', '#fc8d62']  # Colors for different cluster sizes

# Configuration for individual point markers
point_color = '#e78ac3'  # Color for individual points when zoomed in
point_size = 8  # Size of individual points on the map

# Options for parallel coordinates plot axes
# These are the numeric attributes that can be selected for comparison
AXIS_OPTS = [
    'price', 'service_fee', 'minimum nights', 'number of reviews',
    'reviews per month', 'review rate number', 'calculated host listings count',
    'availability 365', 'score'
]

# ===============================================================================
# PRE-CALCULATIONS FOR PERFORMANCE OPTIMIZATION
# ===============================================================================

# Pre-calculate neighborhood statistics to avoid recalculation on every filter change
# This improves dashboard performance by preparing aggregated data in advance
neighborhood_stats = df.groupby('neighbourhood').agg({
    'price': 'mean',
    'score': 'mean',
    'number of reviews': 'mean',
    'review rate number': 'mean',
    'neighbourhood group': 'first',  # Keep track of which borough each neighborhood belongs to
})
neighborhood_stats['count'] = df.groupby('neighbourhood').size()  # Add count of listings per neighborhood

# Pre-calculate room type counts by borough for the stacked bar chart
borough_room_counts = df.groupby(['neighbourhood group', 'room type']).size().reset_index(name='count')

# ===============================================================================
# DASH APPLICATION SETUP
# ===============================================================================

# Initialize the Dash application
app = Dash(__name__)
app.title = "NYC Airbnb Market Research Dashboard"
server = app.server  # For deployment to production servers

# ===============================================================================
# APPLICATION LAYOUT
# ===============================================================================

app.layout = html.Div([
    # Dashboard header section
    html.Div([
        html.H1("NYC Airbnb Market Research Dashboard",
                style={'textAlign': 'center', 'color': '#506784', 'fontFamily': 'Open Sans'}),
        html.P("Explore NYC Airbnb data to make informed investment decisions",
               style={'textAlign': 'center', 'color': '#666666'})
    ], style={'backgroundColor': '#f9f9f9', 'padding': '10px', 'marginBottom': '20px', 'borderRadius': '5px',
              'boxShadow': '2px 2px 2px lightgrey'}),

    # Main content container with filters and visualizations
    html.Div([
        # Left sidebar with filters
        html.Div([
            html.H3("Filters", style={'marginBottom': '20px', 'color': '#506784'}),

            # Price range filter
            html.Label("Price Range ($)", style={'fontWeight': 'bold'}),
            dcc.RangeSlider(
                id='price-slider',
                min=int(df['price'].min()),
                max=int(min(df['price'].max(), 1000)),  # Cap max at 1000 for better UI
                step=50,
                marks={i: f"${i}" for i in range(0, 1001, 200)},
                value=[int(df['price'].quantile(0.05)), int(df['price'].quantile(0.75))],
                # Default: 5th to 75th percentile
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Br(),

            # Borough filter (neighborhood group)
            html.Label("Neighborhood Group", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='neigh-dropdown',
                options=[{'label': g, 'value': g} for g in sorted(df['neighbourhood group'].unique())],
                value=list(df['neighbourhood group'].unique()),  # Default: all boroughs selected
                multi=True,
                placeholder="Select boroughs"
            ),
            html.Br(),

            # Room type filter
            html.Label("Room Type", style={'fontWeight': 'bold'}),
            dcc.Checklist(
                id='roomtype-checklist',
                options=[{'label': ' ' + rt, 'value': rt} for rt in df['room type'].unique()],
                value=df['room type'].unique().tolist(),  # Default: all room types selected
                labelStyle={'display': 'block', 'marginBottom': '5px'}
            ),
            html.Br(),

            # Review rating filter
            html.Label("Minimum Review Rating", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='rating-slider',
                min=0,
                max=5,
                step=0.5,
                marks={i: str(i) for i in range(6)},
                value=0,  # Default: no minimum rating
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Br(),

            # Minimum nights filter
            html.Label("Minimum Nights", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='min-nights-slider',
                min=1,
                max=30,
                step=1,
                marks={i: str(i) for i in [1, 5, 10, 15, 20, 25, 30]},
                value=1,  # Default: minimum 1 night
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Br(),

            # Statistics for currently selected data
            html.Div([
                html.H4("Selected Data Statistics", style={'textAlign': 'center', 'color': '#506784'}),
                html.Div(id='stats-container')  # Content updated dynamically via callback
            ], style={'marginTop': '20px', 'padding': '10px', 'backgroundColor': '#f0f0f0', 'borderRadius': '5px'})

        ], style={'width': '25%', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px',
                  'boxShadow': '2px 2px 2px lightgrey'}),

        # Store component for preserving selection state across tabs
        dcc.Store(id='borough-selection', data={'room_sel': [], 'price_sel': []}),

        # Main visualization area with tabs
        html.Div([
            dcc.Tabs([
                # Tab 1: Map View
                dcc.Tab(label="Map View", children=[
                    html.Div([
                        html.P("Explore the geographic distribution of Airbnb listings across NYC boroughs.",
                               style={'marginBottom': '10px'}),
                        dcc.Graph(id='map-graph', style={'height': '70vh'})
                    ], style={'padding': '20px'})
                ]),

                # Tab 2: Borough Analysis
                dcc.Tab(label="Borough Analysis", children=[
                    html.Div([
                        # First row of charts: Room type distribution and price analysis
                        html.Div([
                            # Room type distribution chart
                            html.Div([
                                html.H4("Room Type Distribution by Borough",
                                        style={'textAlign': 'center', 'color': '#506784'}),
                                dcc.Graph(
                                    id='room-type-graph',
                                    config={
                                        'modeBarButtonsToAdd': ['select2d', 'lasso2d'],  # Enable selection tools
                                        'displayModeBar': True
                                    }
                                )
                            ], style={'width': '50%'}),

                            # Average price by borough and room type chart
                            html.Div([
                                html.H4("Average Price by Borough and Room Type",
                                        style={'textAlign': 'center', 'color': '#506784'}),
                                dcc.Graph(
                                    id='price-borough-graph',
                                    config={
                                        'modeBarButtonsToAdd': ['select2d', 'lasso2d'],  # Enable selection tools
                                        'displayModeBar': True
                                    }
                                )
                            ], style={'width': '50%'})
                        ], style={'display': 'flex'}),

                        # Second row of charts: Cancellation policy and minimum nights
                        html.Div([
                            # Cancellation policy distribution chart
                            html.Div([
                                html.H4("Cancellation Policy Distribution",
                                        style={'textAlign': 'center', 'color': '#506784'}),
                                dcc.Graph(id='cancellation-graph')
                            ], style={'width': '50%'}),

                            # Average minimum nights by room type chart
                            html.Div([
                                html.H4("Average Minimum Nights by Room Type",
                                        style={'textAlign': 'center', 'color': '#506784'}),
                                dcc.Graph(id='min-nights-graph')
                            ], style={'width': '50%'})
                        ], style={'display': 'flex', 'marginTop': '20px'})
                    ], style={'padding': '20px'})
                ]),

                # Tab 3: Neighborhood Comparison
                dcc.Tab(label="Neighborhood Comparison", children=[
                    html.Div([
                        # Metric selection dropdown
                        html.Div([
                            html.Label("Select Metric", style={'fontWeight': 'bold'}),
                            dcc.Dropdown(
                                id='neighborhood-metric-dropdown',
                                options=[
                                    {'label': 'Number of Listings', 'value': 'count'},
                                    {'label': 'Average Price', 'value': 'price'},
                                    {'label': 'Average Score', 'value': 'score'},
                                    {'label': 'Average Review Rating', 'value': 'review rate number'},
                                    {'label': 'Average Number of Reviews', 'value': 'number of reviews'}
                                ],
                                value='count',  # Default: number of listings
                                clearable=False
                            )
                        ], style={'width': '30%', 'marginBottom': '20px'}),

                        # Neighborhood comparison chart
                        dcc.Graph(id='neighborhood-comparison-graph', style={'height': '70vh'})
                    ], style={'padding': '20px'})
                ]),

                # Tab 4: Advanced Analysis with Parallel Coordinates Plot (PCP)
                dcc.Tab(label="Advanced Analysis", children=[
                    html.Div([
                        html.P("Use the parallel coordinates plot to explore relationships between multiple variables.",
                               style={'marginBottom': '10px'}),

                        # Axis selection for parallel coordinates plot
                        html.Div([
                            html.Div([
                                html.Label(f"Axis {i + 1}", style={'fontWeight': 'bold'}),
                                dcc.Dropdown(
                                    id=f'axis-{i + 1}',
                                    options=[{'label': c, 'value': c} for c in AXIS_OPTS],
                                    value=AXIS_OPTS[i] if i < len(AXIS_OPTS) else AXIS_OPTS[0],
                                    clearable=False
                                )
                            ], style={'flex': '1', 'minWidth': '150px', 'marginRight': '10px'})
                            for i in range(5)  # Create 5 axis selectors
                        ], style={'display': 'flex', 'marginBottom': '20px'}),

                        # Parallel coordinates plot
                        dcc.Graph(id='pcp-graph', style={'height': '60vh'})
                    ], style={'padding': '20px'})
                ])
            ], style={'marginBottom': '20px'})
        ], style={'width': '75%', 'paddingLeft': '20px'})
    ], style={'display': 'flex'})
])


# ===============================================================================
# CALLBACKS
# ===============================================================================

@app.callback(
    # Output components to update
    [Output('map-graph', 'figure'),
     Output('room-type-graph', 'figure'),
     Output('price-borough-graph', 'figure'),
     Output('cancellation-graph', 'figure'),
     Output('min-nights-graph', 'figure'),
     Output('neighborhood-comparison-graph', 'figure'),
     Output('pcp-graph', 'figure'),
     Output('stats-container', 'children')],
    # Input components that trigger updates
    [Input('price-slider', 'value'),
     Input('neigh-dropdown', 'value'),
     Input('roomtype-checklist', 'value'),
     Input('rating-slider', 'value'),
     Input('min-nights-slider', 'value'),
     Input('neighborhood-metric-dropdown', 'value'),
     Input('axis-1', 'value'),
     Input('axis-2', 'value'),
     Input('axis-3', 'value'),
     Input('axis-4', 'value'),
     Input('axis-5', 'value'),
     Input('room-type-graph', 'selectedData'),
     Input('price-borough-graph', 'selectedData')
     ]
)
def update_outputs(price_range, selected_groups, selected_rooms, min_rating,
                   min_nights, neighborhood_metric, axis1, axis2, axis3, axis4, axis5,
                   room_selected, price_selected):
    """
    Main callback function that updates all dashboard outputs based on user inputs:
    - Filters data based on price range, neighborhoods, room types, ratings, and min nights
    - Creates interactive map visualization with clustering
    - Generates borough analysis charts showing room type distribution and pricing
    - Provides cancellation policy and minimum nights visualizations
    - Creates neighborhood comparison chart based on selected metric
    - Builds parallel coordinates plot for advanced analysis
    - Calculates statistics for the filtered dataset

    Args:
        price_range (list): Min and max price values [min, max]
        selected_groups (list): Selected borough/neighborhood groups
        selected_rooms (list): Selected room types
        min_rating (float): Minimum review rating threshold
        min_nights (int): Minimum nights threshold
        neighborhood_metric (str): Metric for neighborhood comparison
        axis1-axis5 (str): Selected dimensions for parallel coordinates plot
        room_selected (dict): Selection data from room type chart
        price_selected (dict): Selection data from price chart

    Returns:
        list: List of output figures and components in the order specified by Output
    """
    # Extract price range values
    low, high = price_range

    # Apply filters to the dataset
    # Start with price filter as it's likely to reduce the dataset size significantly
    dff = df[(df['price'] >= low) & (df['price'] <= high)]

    # Apply borough filter if any are selected
    if selected_groups:
        dff = dff[dff['neighbourhood group'].isin(selected_groups)]

    # Apply room type filter if any are selected
    if selected_rooms:
        dff = dff[dff['room type'].isin(selected_rooms)]

    # Apply review rating filter if set above 0
    if min_rating > 0:
        dff = dff[dff['review rate number'] >= min_rating]

    # Apply minimum nights filter if set above 1
    if min_nights > 1:
        dff = dff[dff['minimum nights'] >= min_nights]

    # Handle case where no data matches the filters
    if len(dff) == 0:
        return [go.Figure() for _ in range(7)] + [html.P("No data matches the selected filters.")]

    # Process selections from interactive charts
    # Extract selected boroughs from room type chart
    sel_boroughs = set()
    if room_selected and 'points' in room_selected:
        sel_boroughs = {pt['x'] for pt in room_selected['points']}

    # Extract selected boroughs from price chart
    sel_price_boroughs = set()
    if price_selected and 'points' in price_selected:
        sel_price_boroughs = {pt['x'] for pt in price_selected['points']}

    # -----------------------------------------------------------------------
    # Create Map Visualization
    # -----------------------------------------------------------------------
    map_fig = go.Figure(go.Scattermapbox(
        lat=dff['lat'],
        lon=dff['long'],
        mode='markers',
        text=dff['hover_text'],
        hoverinfo='text',
        marker=go.scattermapbox.Marker(
            size=point_size,
            color=dff['score'],  # Color points by score
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Score'),
            cmin=0,
            cmax=100
        ),
        # Enable clustering for better performance with large datasets
        cluster=go.scattermapbox.Cluster(
            enabled=True,
            step=cluster_steps,
            size=cluster_sizes,
            color=cluster_colors,
            maxzoom=15,  # Zoom level at which clusters break apart
            opacity=0.8
        )
    ))

    # Configure map layout
    map_fig.update_layout(
        mapbox_style='open-street-map',  # Use free open-street-map tiles
        mapbox_center={'lat': dff['lat'].mean(), 'lon': dff['long'].mean()},  # Center on data
        mapbox_zoom=10,  # Initial zoom level
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},  # Maximize map area
        height=600
    )

    # -----------------------------------------------------------------------
    # Create Room Type Distribution Chart
    # -----------------------------------------------------------------------
    # Filter pre-calculated room type counts by selected boroughs
    rtc_data = borough_room_counts[
        borough_room_counts['neighbourhood group'].isin(selected_groups)
    ]
    # Further filter by selections from price chart if any
    if sel_price_boroughs:
        rtc_data = rtc_data[rtc_data['neighbourhood group'].isin(sel_price_boroughs)]

    # Create stacked bar chart for room type distribution
    room_type_fig = px.bar(
        rtc_data,
        x='neighbourhood group',
        y='count',
        color='room type',
        barmode='stack',  # Stack bars to show part-to-whole relationship
        color_discrete_sequence=px.colors.qualitative.Safe,  # Colorblind-safe palette
        labels={'count': 'Number of Listings', 'neighbourhood group': 'Borough', 'room type': 'Room Type'}
    )

    # Configure room type chart layout
    room_type_fig.update_layout(
        xaxis_title='Borough',
        yaxis_title='Number of Listings',
        legend_title='Room Type',
        height=400,
        dragmode='select',  # Enable selection mode for cross-filtering
        uirevision='borough_selection',  # Preserve selections on callbacks
        selectionrevision=1  # Force selection to update
    )

    # -----------------------------------------------------------------------
    # Create Price by Borough and Room Type Chart
    # -----------------------------------------------------------------------
    # Calculate average prices by borough and room type
    borough_price_data = dff.groupby(['neighbourhood group', 'room type'])['price'].mean().reset_index()

    # Apply filter from room type chart selections if any
    if sel_boroughs:
        borough_price_data = borough_price_data[borough_price_data['neighbourhood group'].isin(sel_boroughs)]

    # Create grouped bar chart for price comparison
    price_borough_fig = px.bar(
        borough_price_data,
        x='neighbourhood group',
        y='price',
        color='room type',
        barmode='group',  # Group bars to facilitate direct comparison
        color_discrete_sequence=px.colors.qualitative.Safe,  # Colorblind-safe palette
        labels={'price': 'Average Price ($)', 'neighbourhood group': 'Borough', 'room type': 'Room Type'}
    )

    # Configure price chart layout
    price_borough_fig.update_layout(
        xaxis_title='Borough',
        yaxis_title='Average Price ($)',
        legend_title='Room Type',
        height=400,
        dragmode='select',  # Enable selection mode for cross-filtering
        uirevision='borough_selection',  # Preserve selections on callbacks
        selectionrevision=1  # Force selection to update
    )

    # -----------------------------------------------------------------------
    # Create Cancellation Policy Distribution Chart
    # -----------------------------------------------------------------------
    # Count listings by cancellation policy
    cancellation_data = dff['cancellation_policy'].value_counts().reset_index()
    cancellation_data.columns = ['policy', 'count']

    # Create donut chart for cancellation policy distribution
    cancellation_fig = px.pie(
        cancellation_data,
        values='count',
        names='policy',
        color_discrete_sequence=px.colors.qualitative.Safe,  # Colorblind-safe palette
        hole=0.4  # Create donut chart with 40% hole
    )

    # Configure cancellation policy chart layout
    cancellation_fig.update_layout(
        legend_title='Cancellation Policy',
        height=400
    )

    # -----------------------------------------------------------------------
    # Create Minimum Nights by Room Type Chart
    # -----------------------------------------------------------------------
    # Calculate average minimum nights by room type
    min_nights_data = dff.groupby(['room type'])['minimum nights'].mean().reset_index()

    # Create bar chart for minimum nights comparison
    min_nights_fig = px.bar(
        min_nights_data,
        x='room type',
        y='minimum nights',
        color='room type',
        color_discrete_sequence=px.colors.qualitative.Safe,  # Colorblind-safe palette
        labels={'minimum nights': 'Average Minimum Nights', 'room type': 'Room Type'}
    )

    # Configure minimum nights chart layout
    min_nights_fig.update_layout(
        xaxis_title='Room Type',
        yaxis_title='Average Minimum Nights',
        showlegend=False,  # Hide legend as colors already indicate room type
        height=400
    )

    # -----------------------------------------------------------------------
    # Create Neighborhood Comparison Chart
    # -----------------------------------------------------------------------
    # Calculate metrics by neighborhood
    neighborhood_data = dff.groupby('neighbourhood').agg({
        'price': 'mean',
        'score': 'mean',
        'review rate number': 'mean',
        'number of reviews': 'mean',
        'neighbourhood group': 'first',  # Keep track of which borough each neighborhood belongs to
    }).reset_index()

    # Add count of listings per neighborhood
    count_data = dff.groupby('neighbourhood').size().reset_index(name='count')
    neighborhood_data = neighborhood_data.merge(count_data, on='neighbourhood')

    # Sort by selected metric and take top 20 for readability
    neighborhood_data = neighborhood_data.sort_values(neighborhood_metric, ascending=False).head(20)

    # Create horizontal bar chart for neighborhood comparison
    neighborhood_fig = px.bar(
        neighborhood_data,
        x='neighbourhood',
        y=neighborhood_metric,
        color='neighbourhood group',  # Color by borough
        color_discrete_sequence=px.colors.qualitative.Bold,  # Use bold colors to distinguish boroughs
        labels={
            'neighbourhood': 'Neighborhood',
            'count': 'Number of Listings',
            'price': 'Average Price ($)',
            'score': 'Average Score',
            'review rate number': 'Average Review Rating',
            'number of reviews': 'Average Number of Reviews',
            'neighbourhood group': 'Borough'
        }
    )

    # Define readable metric titles for the y-axis
    metric_title = {
        'count': 'Number of Listings',
        'price': 'Average Price ($)',
        'score': 'Average Score',
        'review rate number': 'Average Review Rating',
        'number of reviews': 'Average Number of Reviews'
    }

    # Configure neighborhood comparison chart layout
    neighborhood_fig.update_layout(
        xaxis_title='Neighborhood',
        yaxis_title=metric_title[neighborhood_metric],
        legend_title='Borough',
        height=600,
        xaxis={'categoryorder': 'total descending', 'tickangle': 45}  # Rotate labels for readability
    )

    # -----------------------------------------------------------------------
    # Create Parallel Coordinates Plot
    # -----------------------------------------------------------------------
    # Process selected axes, ensuring uniqueness
    axes = [axis1, axis2, axis3, axis4, axis5]

    dims = []
    for ax in axes:
        if ax in dff.columns and ax not in dims:
            dims.append(ax)

    # If fewer than 5 unique axes were selected, add additional ones
    while len(dims) < 5:
        for ax in AXIS_OPTS:
            if ax not in dims:
                dims.append(ax)
                break

    # Limit to first 5 dimensions
    dims = dims[:5]

    # Create parallel coordinates plot
    pcp_fig = px.parallel_coordinates(
        dff.dropna(subset=dims),  # Remove rows with NaN in selected dimensions
        dimensions=dims,
        color='score',  # Color lines by listing score
        color_continuous_scale=px.colors.sequential.Viridis,
        labels={
            'price': 'Price ($)',
            'service_fee': 'Service Fee ($)',
            'minimum nights': 'Minimum Nights',
            'number of reviews': 'Number of Reviews',
            'reviews per month': 'Reviews per Month',
            'review rate number': 'Review Rating',
            'calculated host listings count': 'Host Listings Count',
            'availability 365': 'Availability (days/year)',
            'score': 'Score'
        }
    )

    # Configure parallel coordinates plot layout
    pcp_fig.update_layout(
        margin={'t': 30, 'b': 30, 'l': 30, 'r': 30},  # Tight margins to maximize plot area
        height=500,
        coloraxis_colorbar=dict(title='Score')
    )

    # -----------------------------------------------------------------------
    # Create Statistics Summary
    # -----------------------------------------------------------------------
    # Create container with key statistics for the filtered dataset
    stats_container = html.Div([
        html.P(f"Total Listings: {len(dff)}"),
        html.P(f"Average Price: ${dff['price'].mean():.2f}"),
        html.P(f"Average Score: {dff['score'].mean():.2f}"),
        html.P(f"Average Review Rating: {dff['review rate number'].mean():.2f}"),
        html.P(f"Average Minimum Nights: {dff['minimum nights'].mean():.2f}")
    ])

    # Return all figures and components
    return map_fig, room_type_fig, price_borough_fig, cancellation_fig, min_nights_fig, neighborhood_fig, pcp_fig, stats_container


# ===============================================================================
# RUN APPLICATION
# ===============================================================================

if __name__ == '__main__':
    app.run(debug=True)  # Set debug=False for production deployment
