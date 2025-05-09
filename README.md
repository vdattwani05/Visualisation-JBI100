# NYC Airbnb Market Research Dashboard

## Overview
This interactive dashboard analyzes the NYC Airbnb market to support data-driven investment decisions. The application transforms the 102,591-row Airbnb Open Data snapshot for New York City (August 2020) into coordinated interactive visualizations that help users identify profitable areas, understand market trends, and optimize listing configurations.

## Features
- **Interactive Map View**: Geospatial cluster map showing the distribution of Airbnb listings across NYC with dynamic clustering and color-coded scores
- **Borough Analysis**: 
  - Room type distribution across boroughs (stacked bar chart)
  - Average price by borough and room type (grouped bar chart)
  - Cancellation policy distribution (donut chart)
  - Average minimum nights by room type (bar chart)
- **Neighborhood Comparison**: Horizontal bar chart ranking neighborhoods by various performance metrics
- **Advanced Analysis**: Parallel coordinates plot for exploring multivariate relationships
- **Dynamic Filtering**: Filter sidebar with price range, borough selection, room type, review rating, and minimum nights

## Getting Started

### Prerequisites
- Python 3.7+
- pip

### Installation
1. Unzip the file
2. Install required dependencies:
3. Download the dataset: (data already included in the zip)
   - Place the 'Airbnb_Open_Data.csv' file in the project directory
   - The dataset can be obtained from [Airbnb Open Data](http://insideairbnb.com/get-the-data/)

### Running the Application
Access the dashboard at http://127.0.0.1:8050/ in your web browser.

## Implementation Details
- Built with Dash/Plotly for interactive visualization
- Leverages plotly.express and plotly.graph_objects for chart creation
- Implements cross-filtering through coordinated views


## Authors
- Vansh Dattwani (1953281)
- Lia Banuta (1949160)
- Vedarth Mittal (1781669)