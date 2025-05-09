from dash import Dash, dcc, html, Input, Output, dash_table, callback_context
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots


def load_and_clean_data():
    """
    Loads Airbnb data from CSV and performs cleaning operations:
    - Drops unnecessary columns
    - Renames columns for consistency
    - Cleans price and service_fee columns
    - Converts columns to appropriate types
    - Creates score column combining review count and rating
    - Creates hover text for map visualization
    """
    df = pd.read_csv('Airbnb_Open_Data.csv', low_memory=False)

    df = df.drop(columns=['house_rules', 'country code', 'country', 'license'])
    df.rename(columns={'NAME': 'name'}, inplace=True)

    df.set_index('id', inplace=True)
    df['price'] = df['price'].replace(r'[\$,]', '', regex=True).astype(float)
    df.rename(columns={df.columns[df.columns.str.strip() == 'service fee'][0]: 'service_fee'}, inplace=True)
    df['service_fee'] = df['service_fee'].replace(r'[\$,]', '', regex=True).astype(float)

    df = df.dropna(subset=['lat', 'long', 'neighbourhood group', 'room type'])

    df['number of reviews'] = pd.to_numeric(df['number of reviews'], errors='coerce')
    df['review rate number'] = pd.to_numeric(df['review rate number'], errors='coerce')
    df['minimum nights'] = pd.to_numeric(df['minimum nights'], errors='coerce')
    df['reviews per month'] = pd.to_numeric(df['reviews per month'], errors='coerce')
    df['availability 365'] = pd.to_numeric(df['availability 365'], errors='coerce')
    df['calculated host listings count'] = pd.to_numeric(df['calculated host listings count'], errors='coerce')

    df['availability 365'] = df['availability 365'].clip(0, 365)

    max_reviews = df['number of reviews'].max()
    df['score'] = (df['number of reviews'] / max_reviews * 0.5 + df['review rate number'] / 5 * 0.5) * 100
    df['score'] = df['score'].round(2)
    df['score'] = df['score'].fillna(0)

    df['hover_text'] = df.apply(
        lambda r: f"{r['name']}<br>Price: ${r['price']:.0f}<br>Score: {r['score']}<br>Room type: {r['room type']}",
        axis=1
    )

    df['neighbourhood group'] = df['neighbourhood group'].str.capitalize()
    df['neighbourhood group'] = df['neighbourhood group'].replace('Manhatan', 'Manhattan')

    return df


df = load_and_clean_data()

cluster_steps = [10, 50, 100]
cluster_sizes = [10, 20, 30]
cluster_colors = ['#66c2a5', '#8da0cb', '#fc8d62']

point_color = '#e78ac3'
point_size = 8

AXIS_OPTS = [
    'price', 'service_fee', 'minimum nights', 'number of reviews',
    'reviews per month', 'review rate number', 'calculated host listings count',
    'availability 365', 'score'
]

neighborhood_stats = df.groupby('neighbourhood').agg({
    'price': 'mean',
    'score': 'mean',
    'number of reviews': 'mean',
    'review rate number': 'mean',
    'neighbourhood group': 'first',
})
neighborhood_stats['count'] = df.groupby('neighbourhood').size()

borough_room_counts = df.groupby(['neighbourhood group', 'room type']).size().reset_index(name='count')

app = Dash(__name__)
app.title = "NYC Airbnb Market Research Dashboard"
server = app.server

app.layout = html.Div([
    html.Div([
        html.H1("NYC Airbnb Market Research Dashboard",
                style={'textAlign': 'center', 'color': '#506784', 'fontFamily': 'Open Sans'}),
        html.P("Explore NYC Airbnb data to make informed investment decisions",
               style={'textAlign': 'center', 'color': '#666666'})
    ], style={'backgroundColor': '#f9f9f9', 'padding': '10px', 'marginBottom': '20px', 'borderRadius': '5px',
              'boxShadow': '2px 2px 2px lightgrey'}),

    html.Div([
        html.Div([
            html.H3("Filters", style={'marginBottom': '20px', 'color': '#506784'}),

            html.Label("Price Range ($)", style={'fontWeight': 'bold'}),
            dcc.RangeSlider(
                id='price-slider',
                min=int(df['price'].min()),
                max=int(min(df['price'].max(), 1000)),
                step=50,
                marks={i: f"${i}" for i in range(0, 1001, 200)},
                value=[int(df['price'].quantile(0.05)), int(df['price'].quantile(0.75))],
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Br(),

            html.Label("Neighborhood Group", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='neigh-dropdown',
                options=[{'label': g, 'value': g} for g in sorted(df['neighbourhood group'].unique())],
                value=list(df['neighbourhood group'].unique()),
                multi=True,
                placeholder="Select boroughs"
            ),
            html.Br(),

            html.Label("Room Type", style={'fontWeight': 'bold'}),
            dcc.Checklist(
                id='roomtype-checklist',
                options=[{'label': ' ' + rt, 'value': rt} for rt in df['room type'].unique()],
                value=df['room type'].unique().tolist(),
                labelStyle={'display': 'block', 'marginBottom': '5px'}
            ),
            html.Br(),

            html.Label("Minimum Review Rating", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='rating-slider',
                min=0,
                max=5,
                step=0.5,
                marks={i: str(i) for i in range(6)},
                value=0,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Br(),

            html.Label("Minimum Nights", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='min-nights-slider',
                min=1,
                max=30,
                step=1,
                marks={i: str(i) for i in [1, 5, 10, 15, 20, 25, 30]},
                value=1,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Br(),

            html.Div([
                html.H4("Selected Data Statistics", style={'textAlign': 'center', 'color': '#506784'}),
                html.Div(id='stats-container')
            ], style={'marginTop': '20px', 'padding': '10px', 'backgroundColor': '#f0f0f0', 'borderRadius': '5px'})

        ], style={'width': '25%', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px',
                  'boxShadow': '2px 2px 2px lightgrey'}),

        dcc.Store(id='borough-selection', data={'room_sel': [], 'price_sel': []}),
        html.Div([
            dcc.Tabs([
                dcc.Tab(label="Map View", children=[
                    html.Div([
                        html.P("Explore the geographic distribution of Airbnb listings across NYC boroughs.",
                               style={'marginBottom': '10px'}),
                        dcc.Graph(id='map-graph', style={'height': '70vh'})
                    ], style={'padding': '20px'})
                ]),

                dcc.Tab(label="Borough Analysis", children=[
                    html.Div([
                        html.Div([
                            html.Div([
                                html.H4("Room Type Distribution by Borough",
                                        style={'textAlign': 'center', 'color': '#506784'}),
                                dcc.Graph(
                                    id='room-type-graph',
                                    config={
                                        'modeBarButtonsToAdd': ['select2d', 'lasso2d'],
                                        'displayModeBar': True
                                    }
                                )
                            ], style={'width': '50%'}),

                            html.Div([
                                html.H4("Average Price by Borough and Room Type",
                                        style={'textAlign': 'center', 'color': '#506784'}),
                                dcc.Graph(
                                    id='price-borough-graph',
                                    config={
                                        'modeBarButtonsToAdd': ['select2d', 'lasso2d'],
                                        'displayModeBar': True
                                    }
                                )
                            ], style={'width': '50%'})
                        ], style={'display': 'flex'}),

                        html.Div([
                            html.Div([
                                html.H4("Cancellation Policy Distribution",
                                        style={'textAlign': 'center', 'color': '#506784'}),
                                dcc.Graph(id='cancellation-graph')
                            ], style={'width': '50%'}),

                            html.Div([
                                html.H4("Average Minimum Nights by Room Type",
                                        style={'textAlign': 'center', 'color': '#506784'}),
                                dcc.Graph(id='min-nights-graph')
                            ], style={'width': '50%'})
                        ], style={'display': 'flex', 'marginTop': '20px'})
                    ], style={'padding': '20px'})
                ]),

                dcc.Tab(label="Neighborhood Comparison", children=[
                    html.Div([
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
                                value='count',
                                clearable=False
                            )
                        ], style={'width': '30%', 'marginBottom': '20px'}),

                        dcc.Graph(id='neighborhood-comparison-graph', style={'height': '70vh'})
                    ], style={'padding': '20px'})
                ]),

                dcc.Tab(label="Advanced Analysis", children=[
                    html.Div([
                        html.P("Use the parallel coordinates plot to explore relationships between multiple variables.",
                               style={'marginBottom': '10px'}),

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
                            for i in range(5)
                        ], style={'display': 'flex', 'marginBottom': '20px'}),

                        dcc.Graph(id='pcp-graph', style={'height': '60vh'})
                    ], style={'padding': '20px'})
                ])
            ], style={'marginBottom': '20px'})
        ], style={'width': '75%', 'paddingLeft': '20px'})
    ], style={'display': 'flex'})
])


@app.callback(
    [Output('map-graph', 'figure'),
     Output('room-type-graph', 'figure'),
     Output('price-borough-graph', 'figure'),
     Output('cancellation-graph', 'figure'),
     Output('min-nights-graph', 'figure'),
     Output('neighborhood-comparison-graph', 'figure'),
     Output('pcp-graph', 'figure'),
     Output('stats-container', 'children')],
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
    """
    low, high = price_range

    dff = df[(df['price'] >= low) & (df['price'] <= high)]
    if selected_groups:
        dff = dff[dff['neighbourhood group'].isin(selected_groups)]
    if selected_rooms:
        dff = dff[dff['room type'].isin(selected_rooms)]
    if min_rating > 0:
        dff = dff[dff['review rate number'] >= min_rating]
    if min_nights > 1:
        dff = dff[dff['minimum nights'] >= min_nights]

    if len(dff) == 0:
        return [go.Figure() for _ in range(7)] + [html.P("No data matches the selected filters.")]

    sel_boroughs = set()
    if room_selected and 'points' in room_selected:
        sel_boroughs = {pt['x'] for pt in room_selected['points']}
    sel_price_boroughs = set()
    if price_selected and 'points' in price_selected:
        sel_price_boroughs = {pt['x'] for pt in price_selected['points']}

    map_fig = go.Figure(go.Scattermapbox(
        lat=dff['lat'],
        lon=dff['long'],
        mode='markers',
        text=dff['hover_text'],
        hoverinfo='text',
        marker=go.scattermapbox.Marker(
            size=point_size,
            color=dff['score'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Score'),
            cmin=0,
            cmax=100
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

    map_fig.update_layout(
        mapbox_style='open-street-map',
        mapbox_center={'lat': dff['lat'].mean(), 'lon': dff['long'].mean()},
        mapbox_zoom=10,
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
        height=600
    )

    rtc_data = borough_room_counts[
        borough_room_counts['neighbourhood group'].isin(selected_groups)
    ]
    if sel_price_boroughs:
        rtc_data = rtc_data[rtc_data['neighbourhood group'].isin(sel_price_boroughs)]

    room_type_fig = px.bar(
        rtc_data,
        x='neighbourhood group',
        y='count',
        color='room type',
        barmode='stack',
        color_discrete_sequence=px.colors.qualitative.Safe,
        labels={'count': 'Number of Listings', 'neighbourhood group': 'Borough', 'room type': 'Room Type'}
    )

    room_type_fig.update_layout(
        xaxis_title='Borough',
        yaxis_title='Number of Listings',
        legend_title='Room Type',
        height=400,
        dragmode='select',
        uirevision='borough_selection',
        selectionrevision=1
    )

    borough_price_data = dff.groupby(['neighbourhood group', 'room type'])['price'].mean().reset_index()
    if sel_boroughs:
        borough_price_data = borough_price_data[borough_price_data['neighbourhood group'].isin(sel_boroughs)]

    price_borough_fig = px.bar(
        borough_price_data,
        x='neighbourhood group',
        y='price',
        color='room type',
        barmode='group',
        color_discrete_sequence=px.colors.qualitative.Safe,
        labels={'price': 'Average Price ($)', 'neighbourhood group': 'Borough', 'room type': 'Room Type'}
    )
    price_borough_fig.update_layout(
        xaxis_title='Borough',
        yaxis_title='Average Price ($)',
        legend_title='Room Type',
        height=400,
        dragmode='select',
        uirevision='borough_selection',
        selectionrevision=1
    )

    cancellation_data = dff['cancellation_policy'].value_counts().reset_index()
    cancellation_data.columns = ['policy', 'count']

    cancellation_fig = px.pie(
        cancellation_data,
        values='count',
        names='policy',
        color_discrete_sequence=px.colors.qualitative.Safe,
        hole=0.4
    )

    cancellation_fig.update_layout(
        legend_title='Cancellation Policy',
        height=400
    )

    min_nights_data = dff.groupby(['room type'])['minimum nights'].mean().reset_index()
    min_nights_fig = px.bar(
        min_nights_data,
        x='room type',
        y='minimum nights',
        color='room type',
        color_discrete_sequence=px.colors.qualitative.Safe,
        labels={'minimum nights': 'Average Minimum Nights', 'room type': 'Room Type'}
    )

    min_nights_fig.update_layout(
        xaxis_title='Room Type',
        yaxis_title='Average Minimum Nights',
        showlegend=False,
        height=400
    )

    neighborhood_data = dff.groupby('neighbourhood').agg({
        'price': 'mean',
        'score': 'mean',
        'review rate number': 'mean',
        'number of reviews': 'mean',
        'neighbourhood group': 'first',
    }).reset_index()
    count_data = dff.groupby('neighbourhood').size().reset_index(name='count')
    neighborhood_data = neighborhood_data.merge(count_data, on='neighbourhood')

    neighborhood_data = neighborhood_data.sort_values(neighborhood_metric, ascending=False).head(20)

    neighborhood_fig = px.bar(
        neighborhood_data,
        x='neighbourhood',
        y=neighborhood_metric,
        color='neighbourhood group',
        color_discrete_sequence=px.colors.qualitative.Bold,
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

    metric_title = {
        'count': 'Number of Listings',
        'price': 'Average Price ($)',
        'score': 'Average Score',
        'review rate number': 'Average Review Rating',
        'number of reviews': 'Average Number of Reviews'
    }

    neighborhood_fig.update_layout(
        xaxis_title='Neighborhood',
        yaxis_title=metric_title[neighborhood_metric],
        legend_title='Borough',
        height=600,
        xaxis={'categoryorder': 'total descending', 'tickangle': 45}
    )

    axes = [axis1, axis2, axis3, axis4, axis5]

    dims = []
    for ax in axes:
        if ax in dff.columns and ax not in dims:
            dims.append(ax)

    while len(dims) < 5:
        for ax in AXIS_OPTS:
            if ax not in dims:
                dims.append(ax)
                break

    dims = dims[:5]

    pcp_fig = px.parallel_coordinates(
        dff.dropna(subset=dims),
        dimensions=dims,
        color='score',
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

    pcp_fig.update_layout(
        margin={'t': 30, 'b': 30, 'l': 30, 'r': 30},
        height=500,
        coloraxis_colorbar=dict(title='Score')
    )

    stats_container = html.Div([
        html.P(f"Total Listings: {len(dff)}"),
        html.P(f"Average Price: ${dff['price'].mean():.2f}"),
        html.P(f"Average Score: {dff['score'].mean():.2f}"),
        html.P(f"Average Review Rating: {dff['review rate number'].mean():.2f}"),
        html.P(f"Average Minimum Nights: {dff['minimum nights'].mean():.2f}")
    ])

    return map_fig, room_type_fig, price_borough_fig, cancellation_fig, min_nights_fig, neighborhood_fig, pcp_fig, stats_container


if __name__ == '__main__':
    app.run(debug=True)