import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import numpy as np
import os
import re
import openai  # For AI features
from datetime import datetime
import folium
from streamlit_folium import folium_static
# from geopy.geocoders import Nominatim # Not actively used, can be commented if not needed for future
# from sklearn.neighbors import NearestNeighbors # Not actively used
# from sklearn.preprocessing import StandardScaler # Not actively used
import math
import requests


# ========== PAGE CONFIG ==========
st.set_page_config(
    layout="wide",
    page_title="🏝️ Terra Caribbean Property Intelligence",
    page_icon="🏝️",
    initial_sidebar_state="expanded"
)

# ========== CONSTANTS & API KEYS ==========
# Load API keys securely using Streamlit Secrets
# Create a file named .streamlit/secrets.toml in your project root folder.
# Add your keys to this file like so:
#
# OPENAI_API_KEY = "sk-..."
# MAPBOX_ACCESS_TOKEN = "pk.eyJ..."
# OPENWEATHERMAP_API_KEY = "your_openweathermap_key"

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
MAPBOX_ACCESS_TOKEN = st.secrets.get("MAPBOX_ACCESS_TOKEN", "")
OPENWEATHERMAP_API_KEY = st.secrets.get("OPENWEATHERMAP_API_KEY", "")

# You can add checks here if you want to explicitly warn the user if a key is missing:
# if not OPENAI_API_KEY:
#     st.sidebar.warning("OpenAI API Key not found in secrets. AI features will be disabled.", icon="⚠️")
# if not MAPBOX_ACCESS_TOKEN:
#     st.sidebar.warning("Mapbox Token not found in secrets. Some map features might be limited.", icon="⚠️")
# if not OPENWEATHERMAP_API_KEY:
#     st.sidebar.warning("OpenWeatherMap API Key not found. Weather display disabled.", icon="⚠️")


# Define market-specific data sources and base configs for scalability
MARKET_DATA_SOURCES = {
    'Barbados': {
        'file': "Terra Caribbean NEW SAMPLE R.xlsx",
        'header_row': 3,
        'default_coords': (13.1939, -59.5432), # Center of Barbados
        'parish_coords': {
            'Christ Church': (13.0770, -59.5300),
            'St. Andrew': (13.2360, -59.5685),
            'St. George': (13.1500, -59.5500),
            'St. James': (13.1850, -59.6300),
            'St. John': (13.1720, -59.4900),
            'St. Joseph': (13.2020, -59.5250),
            'St. Lucy': (13.2850, -59.6100),
            'St. Michael': (13.1050, -59.6100),
            'St. Peter': (13.2450, -59.6300),
            'St. Philip': (13.1200, -59.4750),
            'St. Thomas': (13.1800, -59.5850)
        },
        'amenities': {
            'Beach': [
                {'name': 'Crane Beach', 'lat': 13.0986, 'lon': -59.4485},
                {'name': 'Miami Beach', 'lat': 13.0833, 'lon': -59.5333},
                {'name': 'Carlisle Bay', 'lat': 13.0778, 'lon': -59.6142}
            ],
            'School': [
                {'name': 'Harrison College', 'lat': 13.0978, 'lon': -59.6144},
                {'name': "Queen's College", 'lat': 13.0953, 'lon': -59.6169}
            ],
            'Restaurant': [
                {'name': 'The Cliff', 'lat': 13.1800, 'lon': -59.6389},
                {'name': 'Lone Star Restaurant', 'lat': 13.1975, 'lon': -59.6414}
            ]
        }
    }
    # Future Markets planned for expansion: Grenada, Trinidad, St. Lucia
    # To add a new market, create a new entry in this dictionary following the structure above.
    # E.g., 'Grenada': { 'file': 'GrenadaData.xlsx', ... }
}

# ========== THEME DEFINITIONS ==========
if 'current_theme' not in st.session_state:
    st.session_state.current_theme = 'Light'

# Plotly themes - Used directly in Plotly chart layout/colors
# Adjusted category_colors for better Rent vs. For Sale contrast
light_theme_plotly = {
    "primary": "#1a3e72", # Dark Blue
    "secondary": "#e9a229", # Orange/Yellow
    "accent": "#5f8fd9", # Medium Blue
    "neutral_grey": "#bdc3c7", # Light Grey
    "paper_bgcolor": "#ffffff", # White
    "plot_bgcolor": "#f5f7fa", # Very Light Grey/Blue
    "font_color": "#333333", # Dark Grey Text
    "grid_color": "#e0e0e0", # Light Grey Grid
    "map_bubble_color": "#00BCD4",
    "map_text_color": "#000000",
    # Category colors - Adjusted for clearer Rent vs. For Sale
    # Ensured keys match potential uppercase data values like "FOR SALE"
    "category_colors": {
        'FOR SALE': '#e74c3c', # Vivid Red
        'FOR RENT': '#3498db',  # Brighter Blue
        'Sold': '#27ae60',      # Green # Assuming 'Sold' might be title case
        'Leased': '#2980b9',    # Slightly different blue # Assuming 'Leased' might be title case
        'Unknown': '#bdc3c7',   # Neutral Grey
        # Adding original title case as fallbacks or if data is mixed,
        # though direct match to data is better.
        'For Sale': '#e74c3c',
        'Rent': '#3498db',
    },
    # Bar chart palette - adjust these for contrast in bar charts if needed
    "bar_palette": ["#1a3e72", "#e9a229", "#5f8fd9", "#2c5282", "#f6e05e", "#7f8fd9", "#4a5568", "#a0aec0", "#667eea", "#ed8936"]
}

# --- Also check the dark theme ---
dark_theme_plotly = {
    "primary": "#76a2dd", "secondary": "#f7ca4f", "accent": "#e9a229", "neutral_grey": "#95a5a6",
    "paper_bgcolor": "#2d394a", "plot_bgcolor": "#222b38", "font_color": "#ecf0f1", "grid_color": "#4a6378",
    "map_bubble_color": "#00ACC1", "map_text_color": "#ffffff",
    # Category colors - Adjusted for clearer Rent vs. For Sale
    # Ensured keys match potential uppercase data values like "FOR SALE"
    "category_colors": {
        'FOR SALE': '#e74c3c', # Vivid Red (works on dark background too)
        'FOR RENT': '#76a2dd',  # Retain lighter blue
        'Sold': '#32c878',      # Green # Assuming 'Sold' might be title case
        'Leased': '#5dade2',    # Slightly different blue # Assuming 'Leased' might be title case
        'Unknown': '#95a5a6',   # Neutral Grey
        # Adding original title case as fallbacks
        'For Sale': '#e74c3c',
        'Rent': '#76a2dd',
    },
    "bar_palette": ["#76a2dd", "#f7ca4f", "#A7C7E7", "#4e7ab5", "#fff176", "#a6bde7", "#718096", "#d3dce6", "#9ab5f9", "#f0a560"]
}

# Update the THEME_PLOTLY assignment line as well:
THEME_PLOTLY = light_theme_plotly if st.session_state.current_theme == 'Light' else dark_theme_plotly

# CSS variables for structural/custom components - Used directly in Markdown with unsafe_allow_html
# You can adjust these hex codes to better match the website image colors if desired
if st.session_state.current_theme == 'Light':
    theme_variables_css = """
    <style>
        :root {
            --primary: #1a3e72; /* Dark Blue */
            --secondary: #e9a229; /* Orange/Yellow */
            --accent: #5f8fd9; /* Medium Blue */
            --text-light: #ffffff; /* White */
            --text-dark: #333333; /* Dark Grey */
            --text-neutral: #7f8c8d; /* Neutral Grey */
            --background-main: #f5f7fa; /* Very Light Grey/Blue - could try #ffffff for pure white */
            --background-card: #ffffff; /* White */
            --background-sidebar: #1a3e72; /* Dark Blue */
            --border-color: #dddddd; /* Light Grey */
            --shadow-color: rgba(0,0,0,0.1);
            --search-highlight: #FFF59D;
            --data-quality-high: #4CAF50; /* Green */
            --data-quality-medium: #FFC107; /* Amber */
            --data-quality-low: #F44336; /* Red */
        }
    </style>
    """
else:
    theme_variables_css = """
    <style>
        :root {
            --primary: #76a2dd; --secondary: #f7ca4f; --accent: #e9a229;
            --text-light: #1f2c38; --text-dark: #ecf0f1; --text-neutral: #bdc3c7;
            --background-main: #222b38; --background-card: #2d394a; --background-sidebar: #1f2c38;
            --border-color: #4a6378; --shadow-color: rgba(0,0,0,0.3);
            --search-highlight: #FFD54F;
            --data-quality-high: #81C784;
            --data-quality-medium: #FFD54F;
            --data-quality-low: #E57373;
        }
    </style>
    """
st.markdown(theme_variables_css, unsafe_allow_html=True)
MAIN_STRUCTURAL_CSS = """
<style>
    /* General body and main container styling */
    body { color: var(--text-dark); background-color: var(--background-main); }
    .main { background-color: var(--background-main); padding: 1rem; }
    .st-emotion-cache-1y4p8pa { padding: 2rem 1rem; } /* Adjust padding of the main content area */

    /* Plotly chart container styling */
    .stPlotlyChart { border-radius: 8px; box-shadow: 0 4px 8px var(--shadow-color); margin-bottom: 1rem; }

    /* Metric cards styling */
    .metric-card {
        padding: 20px; border-radius: 10px; background: var(--background-card);
        box-shadow: 0 4px 8px var(--shadow-color); height: 100%;
        display: flex; flex-direction: column; justify-content: space-around;
        border-top: 5px solid var(--secondary);
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        color: var(--text-dark); /* Ensure text color is readable against card background */
    }
    .metric-card:hover { transform: translateY(-5px); box-shadow: 0 6px 12px var(--shadow-color); }
    .metric-card h3 { color: var(--primary); font-size: 1.1em; margin-bottom: 8px; }
    .metric-card p { font-size: 28px; font-weight: bold; color: var(--accent); margin-top: 5px; }

    /* Sidebar styling */
    .sidebar .sidebar-content { background-color: var(--background-sidebar); color: var(--text-light); }
    /* Make sidebar elements like titles/labels visible */
    .sidebar .st-emotion-cache-vk34n { color: var(--text-light); } /* Target element containing filter labels */
    .sidebar h2, .sidebar h3 { color: var(--secondary) !important; margin-bottom: 15px;} /* Sidebar section headers */
    .sidebar .stRadio > label { color: var(--text-light); } /* Radio button labels */


    /* Search highlight */
    .highlight { background-color: var(--search-highlight); padding: 0.1em 0.2em; border-radius: 0.2em; font-weight: bold; }

    /* DataFrame styling */
    .dataframe { width: 100%; border-collapse: collapse; border-spacing: 0; border-radius: 8px; overflow: hidden; } /* Added overflow hidden for border-radius */
    .dataframe th { background-color: var(--primary); color: var(--text-light); padding: 12px 8px; text-align: left; border-bottom: 1px solid var(--border-color); }
    .dataframe td { padding: 8px; border-bottom: 1px solid var(--border-color); color: var(--text-dark); } /* Ensure cell text is dark */
    .dataframe tr:nth-child(even) { background-color: var(--background-card); } /* Use card background for even rows */
    .dataframe tr:nth-child(odd) { background-color: var(--background-main); } /* Use main background for odd rows for subtle striping */
    .dataframe tbody tr:hover { background-color: rgba(233, 162, 41, 0.2) !important; } /* Hover effect using a translucent accent color */
    .stDataFrame { border-radius: 8px; box-shadow: 0 2px 4px var(--shadow-color); } /* Container shadow */
    .stDataFrame div[data-testid="stHorizontalBlock"] { gap: 0.5rem; }

    /* Data quality spans */
    .data-quality-high { color: var(--data-quality-high); font-weight: bold; }
    .data-quality-medium { color: var(--data-quality-medium); font-weight: bold; }
    .data-quality-low { color: var(--data-quality-low); font-weight: bold; }

    /* Map related */
    .amenity-marker { color: #FFFFFF; font-weight: bold; text-align: center; border-radius: 50%; }
    .price-heatmap { opacity: 0.7; } /* Assuming this was for a heatmap overlay */

    /* Custom legend styling */
    .map-legend-item { display: flex; align-items: center; margin-right: 15px; font-size: 0.9em; color: var(--text-dark); }
    .map-legend-color-box { width: 15px; height: 15px; margin-right: 5px; border-radius: 3px; border: 1px solid var(--border-color); }
    .map-legend-icon { margin-right: 5px; }

    /* Streamlit specific adjustments for selectboxes, sliders etc. */
    /* Adjusting padding/margins for better spacing */
    .stSelectbox, .stMultiSelect, .stSlider, .stCheckbox { margin-bottom: 10px; }
</style>
"""
st.markdown(MAIN_STRUCTURAL_CSS, unsafe_allow_html=True)

# ========== DATA LOADING & CLEANING FUNCTIONS ==========
@st.cache_data
def load_data(island_name):
    """Loads data for a specific island market."""
    market_config = MARKET_DATA_SOURCES.get(island_name)
    if not market_config:
        st.error(f"Error: Configuration for market '{island_name}' not found.")
        return pd.DataFrame()

    excel_file_path = market_config['file']
    header_row = market_config['header_row']

    if not os.path.exists(excel_file_path):
        st.error(f"Error: Data file '{excel_file_path}' not found for {island_name} market.")
        return pd.DataFrame()

    try:
        df = pd.read_excel(excel_file_path, header=header_row)
    except Exception as e:
        st.error(f"Error reading Excel file '{excel_file_path}': {e}")
        return pd.DataFrame()

    df = df.dropna(how='all')
    if df.empty:
        st.warning(f"Data loaded for {island_name} is empty after cleaning.")
        return df

    required_initial_cols = ['Price', 'Property Type', 'Type', 'Description', 'Parish', 'Category', 'Name']
    for col in required_initial_cols:
        if col not in df.columns:
            df[col] = None

    df['Price'] = pd.to_numeric(df['Price'].astype(str).str.replace('US$', '').str.replace(',', ''), errors='coerce')
    df['Price'] = df['Price'].fillna(0)

    df['Property Type'] = df['Property Type'].astype(str).replace({'nan': 'Unknown', 'N/A': 'Unknown', '': 'Unknown'})
    df['Type'] = df['Type'].astype(str).str.title().replace({'Nan': 'Unknown', 'N/A': 'Unknown', '': 'Unknown'})
    df['Category'] = df['Category'].astype(str).replace({'nan': 'Unknown', 'N/A': 'Unknown', '': 'Unknown'})
    df['Name'] = df['Name'].astype(str).str.strip().replace({'nan': 'Property', 'N/A': 'Property', '': 'Property'})
    df['Description'] = df['Description'].astype(str).fillna('')


    df['Bedrooms'] = pd.to_numeric(df['Description'].str.extract(r'(\d+)\s*Bed')[0], errors='coerce').fillna(0)
    df['Bathrooms'] = pd.to_numeric(df['Description'].str.extract(r'(\d+)\s*Bath')[0], errors='coerce').fillna(0)

    if 'Parish' in df.columns:
        if island_name == 'Barbados':
            df['Parish'] = df['Parish'].astype(str).str.strip()
            df['Parish'] = df['Parish'].replace({'165Christ Church': 'Christ Church', 'James': 'St. James', 'Church': 'Christ Church', 'Joseph': 'St. Joseph'})
            df['Parish'] = df['Parish'].str.replace(r'^Saint\s', 'St. ', regex=True)
            df['Parish'] = df['Parish'].str.replace(r'^ST\.?\s', 'St. ', regex=True)
            df['Parish'] = df['Parish'].fillna('Unknown')
            df.loc[df['Parish'] == '', 'Parish'] = 'Unknown'
        else:
            df['Parish'] = df['Parish'].astype(str).str.strip().fillna('Unknown').replace({'': 'Unknown'})
    else:
        df['Parish'] = 'Unknown'


    df['Country'] = island_name
    df['Data Quality Score'] = df.apply(calculate_data_quality_score, axis=1)
    return df

def calculate_data_quality_score(row):
    """Calculate a data quality score for each property (0-100)"""
    score = 100
    if row.get('Price', 0) == 0: score -= 30
    if row.get('Property Type', 'Unknown') == 'Unknown': score -= 20
    if row.get('Parish', 'Unknown') == 'Unknown': score -= 15
    if row.get('Bedrooms', 0) == 0: score -= 10
    if row.get('Bathrooms', 0) == 0: score -= 10
    if str(row.get('Description', '')).strip() == '': score -= 15
    if row.get('Category', 'Unknown') == 'Unknown': score -= 5
    if row.get('Type', 'Unknown') == 'Unknown': score -= 5
    return max(0, min(100, score))

def get_data_quality_class(score):
    """Return CSS class based on data quality score"""
    if score >= 80: return "data-quality-high"
    elif score >= 50: return "data-quality-medium"
    else: return "data-quality-low"

# ========== WEATHER FUNCTIONS ==========
@st.cache_data(ttl=900) # Cache for 15 minutes (900 seconds)
def get_live_weather(api_key, city_name="Bridgetown", country_code="BB"):
    """Fetches live weather data from OpenWeatherMap API."""
    if not api_key:
        return None
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": f"{city_name},{country_code}",
        "appid": api_key,
        "units": "metric"  # For Celsius
    }
    try:
        response = requests.get(base_url, params=params, timeout=5) # Added timeout
        response.raise_for_status()  # Raise an exception for HTTP errors (4XX, 5XX)
        data = response.json()

        weather_info = {
            "temp": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "description": data["weather"][0]["description"].title(),
            "icon_code": data["weather"][0]["icon"],
            "icon_url": f"http://openweathermap.org/img/wn/{data['weather'][0]['icon']}@2x.png",
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"],  # m/s
            "city": data["name"]
        }
        return weather_info
    except requests.exceptions.Timeout:
        print(f"Weather API request timed out for {city_name}.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Weather API request error for {city_name}: {e}")
        return None
    except KeyError as e:
        print(f"Unexpected weather API response format from OpenWeatherMap (Missing key: {e}) for {city_name}. Data: {data if 'data' in locals() else 'N/A'}")
        return None

def display_weather_sidebar(weather_api_key):
    """Gets and displays weather information in the sidebar."""
    if not weather_api_key:
        st.sidebar.caption("Weather API key not configured.")
        return

    weather_data = get_live_weather(weather_api_key) # Default is Bridgetown, BB

    if weather_data:
        st.sidebar.subheader(f"Weather in {weather_data['city']}")

        col1, col2 = st.sidebar.columns([0.4, 0.6], gap="small")
        with col1:
            st.image(weather_data['icon_url'], width=60)
        with col2:
            st.metric(label="Temp", value=f"{weather_data['temp']:.0f}°C")

        st.sidebar.caption(f"{weather_data['description']}")
        st.sidebar.caption(f"Feels like: {weather_data['feels_like']:.0f}°C")
        st.sidebar.caption(f"Humidity: {weather_data['humidity']}% | Wind: {weather_data['wind_speed']:.1f} m/s")
    else:
        st.sidebar.caption("Live weather data currently unavailable.")


# ========== AI FUNCTIONS ==========
def initialize_ai():
    """Initialize AI components"""
    if not OPENAI_API_KEY:
        return False
    try:
        openai.api_key = OPENAI_API_KEY
        return True
    except Exception as e:
        st.error(f"Failed to initialize OpenAI API: {e}")
        st.warning("AI features disabled.")
        return False

def generate_ai_insights(filtered_df, full_df, market_name):
    """Generate AI-powered insights about the current property selection"""
    if not OPENAI_API_KEY:
        return "AI features are not enabled due to missing API key."
    if len(filtered_df) < 5:
        return "Insufficient data in the current selection to generate meaningful AI insights (less than 5 properties)."
    try:
        summary = {
            "market": market_name,
            "total_properties_filtered": len(filtered_df),
            "total_properties_market": len(full_df),
            "price_stats_USD": {
                "min": filtered_df['Price'].min(),
                "max": filtered_df['Price'].max(),
                "median": filtered_df['Price'].median(),
                "mean": filtered_df['Price'].mean()
            } if 'Price' in filtered_df.columns and not filtered_df['Price'].isnull().all() and not filtered_df.empty else {"min":0,"max":0,"median":0,"mean":0},
            "property_types": filtered_df['Property Type'].value_counts().to_dict() if 'Property Type' in filtered_df.columns else {},
            "parish_distribution": filtered_df['Parish'].value_counts().to_dict() if 'Parish' in filtered_df.columns else {},
            "transaction_types": filtered_df['Category'].value_counts().to_dict() if 'Category' in filtered_df.columns else {}
        }
        prompt = f"""
        Analyze this real estate data summary for the {market_name} market. All prices provided are in US Dollars (USD).
        Provide 3-5 key insights in bullet points. Focus on:
        - Price trends and value propositions within the filtered selection (remember prices are USD).
        - Property type distribution within the filtered selection.
        - Geographic distribution (parishes) within the filtered selection.
        - Comparison to the overall market if possible (mention total market size).
        - Any notable patterns or outliers based on the summary.

        Data Summary (Filtered Selection, Prices in USD):
        {summary}

        Please provide concise, professional insights that would be valuable to real estate investors and buyers.
        Ensure you state that prices are in USD where appropriate in the insights.
        Format as a bulleted list.
        """
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        if "authentication_error" in str(e).lower():
            return "AI insight generation failed: Authentication error. Please check your OpenAI API key."
        elif "rate_limit_exceeded" in str(e).lower():
            return "AI insight generation failed: Rate limit exceeded. Please try again later."
        else:
            return f"AI insight generation failed: {str(e)}"

def apply_search_filter(df, search_term):
    """Applies a simple text search filter across multiple relevant columns."""
    if not search_term or df.empty: return df
    search_term_lower = search_term.lower()
    searchable_cols = ['Name', 'Description', 'Parish', 'Property Type', 'Category', 'Type']
    cols_to_search = [col for col in searchable_cols if col in df.columns]
    if not cols_to_search:
        st.warning("No searchable text columns found for simple search.")
        return df
    combined_mask = pd.Series(False, index=df.index)
    for col in cols_to_search:
        combined_mask |= df[col].astype(str).fillna('').str.lower().str.contains(search_term_lower, na=False)
    return df[combined_mask]

def natural_language_query(query, df):
    """Process natural language queries about properties"""
    if not OPENAI_API_KEY:
        return apply_search_filter(df, query)
    try:
        query_lower = query.lower()
        filtered = df.copy()
        applied_filter = False
        price_match = re.search(r'under\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)|less than\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', query_lower)
        if price_match:
            price_str = price_match.group(1) or price_match.group(2)
            if price_str:
                try:
                    price = float(price_str.replace(',', ''))
                    if 'Price' in filtered.columns:
                        filtered = filtered[filtered['Price'] <= price]
                        applied_filter = True
                except ValueError: pass
        bed_match = re.search(r'(\d+)\s*bed', query_lower)
        bath_match = re.search(r'(\d+)\s*bath', query_lower)
        beds = int(bed_match.group(1)) if bed_match and bed_match.group(1).isdigit() else None
        baths = int(bath_match.group(1)) if bath_match and bath_match.group(1).isdigit() else None
        if beds is not None and 'Bedrooms' in filtered.columns:
            filtered = filtered[filtered['Bedrooms'] >= beds]; applied_filter = True
        if baths is not None and 'Bathrooms' in filtered.columns:
            filtered = filtered[filtered['Bathrooms'] >= baths]; applied_filter = True
        if applied_filter: return filtered

        prompt = f"""
        You are a real estate data assistant. The user has asked:
        "{query}"
        The current dataframe `df` already represents properties that match the standard filters.
        Its columns are: {list(df.columns)}
        Columns likely to contain relevant text information: ['Name', 'Description', 'Parish', 'Property Type', 'Category', 'Type']
        Provide a Python pandas code snippet that would *further filter* the dataframe `df` to match the user's request.
        Focus on filtering based on keywords present in the query that were *not* covered by basic number/price checks, or more complex criteria.
        Use `.str.contains()` for text matching, ensuring case-insensitivity (`case=False, na=False`).
        Use numerical comparisons (`<=`, `>=`) for numbers found with keywords like 'price', 'bed', 'bath'.
        Combine conditions using `&` (AND) or `|` (OR).
        Only respond with the Python code snippet, no additional explanation.
        Ensure the code snippet is safe and directly applicable to the dataframe `df`. Do NOT include any print statements, imports, or other code.
        Examples:
        User: "properties in Christ Church"
        Response: df[df['Parish'].str.contains('Christ Church', case=False, na=False)]
        User: "villas for sale under 1 million"
        Response: df[(df['Category'].str.contains('For Sale', case=False, na=False)) & (df['Property Type'].str.contains('villa', case=False, na=False))]
        User: "3 bedroom house"
        Response: df[df['Property Type'].str.contains('house', case=False, na=False)]
        User: "Properties with a pool"
        Response: df[df['Description'].str.contains('pool', case=False, na=False)]
        User: "Properties in St. James or St. Peter"
        Response: df[df['Parish'].str.contains('St. James|St. Peter', case=False, na=False)]
        Provide the filter code:
        """
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=200
        )
        filter_code = response.choices[0].message.content.strip()
        if filter_code.startswith('df[') and filter_code.endswith(']'):
            try:
                safe_globals = {'df': df.copy(), 'pd': pd, 'np': np, 're': re}
                filtered_df_ai = eval(filter_code, safe_globals)
                if isinstance(filtered_df_ai, pd.DataFrame) and len(filtered_df_ai) <= len(df):
                    return filtered_df_ai
                else:
                    st.warning("AI response did not result in a valid dataframe filter. Falling back to text search.")
                    return apply_search_filter(df, query)
            except Exception as eval_e:
                st.error(f"Error evaluating AI query: {eval_e}. Falling back to text search.")
                return apply_search_filter(df, query)
        else:
            st.warning("AI did not provide a valid filter code. Falling back to text search.")
            return apply_search_filter(df, query)
    except Exception as api_e:
        st.error(f"AI query generation failed: {str(api_e)}. Falling back to text search.")
        return apply_search_filter(df, query)

# ========== GEOSPATIAL FUNCTIONS ==========
def geocode_properties(df, island_name):
    """Add latitude/longitude to properties based on parish, specific to island."""
    if df.empty: return df
    if 'Parish' not in df.columns:
        st.warning("Cannot geocode properties: 'Parish' column missing.")
        market_config = MARKET_DATA_SOURCES.get(island_name)
        default_lat, default_lon = market_config.get('default_coords', (0, 0)) if market_config else (0,0)
        df['lat'] = default_lat
        df['lon'] = default_lon
        return df

    market_config = MARKET_DATA_SOURCES.get(island_name)
    if not market_config or 'parish_coords' not in market_config:
        st.warning(f"No parish coordinates defined for {island_name}. Using default center for all properties.")
        default_lat, default_lon = market_config.get('default_coords', (0, 0)) if market_config else (0,0)
        df['lat'] = default_lat
        df['lon'] = default_lon
        return df

    parish_coords = market_config['parish_coords']
    default_lat, default_lon = market_config.get('default_coords', (0, 0))
    df['lat'] = default_lat
    df['lon'] = default_lon
    df['coords'] = df['Parish'].apply(lambda p: parish_coords.get(p, (default_lat, default_lon)))
    df['lat'] = df['coords'].apply(lambda c: c[0])
    df['lon'] = df['coords'].apply(lambda c: c[1])
    df = df.drop(columns=['coords'])

    assigned_parish_mask = df['Parish'].isin(parish_coords.keys())
    if assigned_parish_mask.any():
        valid_indices = df.index[assigned_parish_mask]
        if not valid_indices.empty:
            df.loc[valid_indices, 'lat'] = df.loc[valid_indices, 'lat'] + np.random.uniform(-0.005, 0.005, size=len(valid_indices))
            df.loc[valid_indices, 'lon'] = df.loc[valid_indices, 'lon'] + np.random.uniform(-0.005, 0.005, size=len(valid_indices))
    return df

def create_advanced_map(filtered_df, island_name, amenities_to_show=None, show_school_districts=False):
    """Create an interactive Folium map with advanced features for the selected island."""
    market_config = MARKET_DATA_SOURCES.get(island_name)
    if not market_config:
        st.error(f"Map config not found for {island_name}")
        return None

    default_lat, default_lon = market_config.get('default_coords', (0, 0))
    market_amenities = market_config.get('amenities', {})

    if filtered_df.empty:
        center_lat, center_lon = default_lat, default_lon
        zoom_level = 11
    else:
        if 'lat' in filtered_df.columns and 'lon' in filtered_df.columns:
            valid_coords = filtered_df[['lat', 'lon']].dropna().astype(float)
            if not valid_coords.empty:
                center_lat = valid_coords['lat'].mean()
                center_lon = valid_coords['lon'].mean()
                zoom_level = 13
            else:
                st.warning("Filtered data has no valid coordinates for map centering.")
                center_lat, center_lon = default_lat, default_lon
                zoom_level = 11
        else:
            st.warning("Latitude/Longitude columns missing for map centering.")
            center_lat, center_lon = default_lat, default_lon
            zoom_level = 11


    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_level,
        tiles='cartodbpositron' if st.session_state.current_theme == 'Light' else 'cartodbdark_matter',
        control_scale=True
    )

    property_group = folium.FeatureGroup(name=f'Properties ({island_name})').add_to(m)
    df_for_map = filtered_df.dropna(subset=['lat', 'lon']).copy()
    for col in ['Name', 'Property Type', 'Parish', 'Category', 'Type']:
        if col in df_for_map.columns:
            df_for_map[col] = df_for_map[col].astype(str)
    for col in ['Price', 'Bedrooms', 'Bathrooms', 'Data Quality Score']:
        if col in df_for_map.columns:
            df_for_map[col] = pd.to_numeric(df_for_map[col], errors='coerce').fillna(0)


    for idx, row in df_for_map.iterrows():
        # Determine icon color based on Category: Red for Sale, Green for Rent, Gray otherwise
        category_value = str(row.get('Category', 'Unknown')).strip().upper() # Normalize to uppercase for comparison

        if category_value == 'FOR SALE':
            icon_color = 'red'  # Red for properties listed as "For Sale"
        elif category_value == 'FOR RENT':
            icon_color = 'green' # Green for properties listed as "For Rent"
        else:
            icon_color = 'gray' # Default color for other categories (Sold, Leased, Unknown, etc.)

        # Determine icon shape
        icon_type_marker = 'home' if str(row.get('Type', '')).lower() == 'residential' else 'building'

        popup_html = f"""
        <b>{row.get('Name', 'N/A')}</b><br>
        <b>Type:</b> {row.get('Property Type', 'N/A')}<br>
        <b>Parish:</b> {row.get('Parish', 'N/A')}<br>
        <b>Category:</b> {row.get('Category', 'N/A')}<br>
        <b>Price:</b> ${row.get('Price', 0):,.0f} USD<br>
        <b>Beds:</b> {int(row.get('Bedrooms', 0))} | <b>Baths:</b> {int(row.get('Bathrooms', 0))}<br>
        <i>Data Quality: {row.get('Data Quality Score', 0):.0f}/100</i>
        """

        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(
                color=icon_color,
                icon=icon_type_marker,
                prefix='fa'
            )
        ).add_to(property_group)

    amenity_layers = {}
    # Use the amenities_to_show parameter received by this function
    if amenities_to_show:
        for amenity_type_key, amenity_locations in market_amenities.items():
            if amenity_type_key in amenities_to_show and amenity_locations:
                color = {'Beach': 'lightblue', 'School': 'orange', 'Restaurant': 'red'}.get(amenity_type_key, 'gray')
                icon_val = {'Beach': 'tint', 'School': 'info-sign', 'Restaurant': 'cutlery'}.get(amenity_type_key, 'info-sign')
                layer_name = f"{amenity_type_key} ({island_name})"
                amenity_group = folium.FeatureGroup(name=layer_name).add_to(m)
                amenity_layers[amenity_type_key] = amenity_group
                for loc in amenity_locations:
                    # Check for valid lat/lon before creating marker
                    if all(k in loc for k in ['lat', 'lon']) and pd.notna(loc['lat']) and pd.notna(loc['lon']):
                         try:
                             folium.Marker(
                                 location=[float(loc['lat']), float(loc['lon'])],
                                 popup=f"{amenity_type_key}: {loc.get('name', 'N/A')}",
                                 icon=folium.Icon(color=color, icon=icon_val, prefix='glyphicon')
                             ).add_to(amenity_group)
                         except ValueError:
                             print(f"Could not add amenity marker for {loc.get('name', 'N/A')}: invalid coordinates.")


    # Use the show_school_districts parameter received by this function
    if show_school_districts and 'School' in market_amenities and market_amenities['School']:
        school_zone_group = folium.FeatureGroup(name=f'School Zones ({island_name})').add_to(m)
        for loc in market_amenities['School']:
            # Check for valid lat/lon before creating circle
            if all(k in loc for k in ['lat', 'lon']) and pd.notna(loc['lat']) and pd.notna(loc['lon']):
                 try:
                     folium.Circle(
                         location=[float(loc['lat']), float(loc['lon'])],
                         radius=1000, color='orange', fill=True, fill_opacity=0.2,
                         popup=f"School Zone: {loc.get('name', 'N/A')}"
                     ).add_to(school_zone_group)
                 except ValueError:
                      print(f"Could not add school zone circle for {loc.get('name', 'N/A')}: invalid coordinates.")

    folium.LayerControl().add_to(m)
    return m

# ========== DATA QUALITY FUNCTIONS ==========
def show_data_quality_report(df):
    """Display a comprehensive data quality report"""
    if df.empty:
        st.caption("No data quality report available for the current selection.")
        return

    st.subheader("🔍 Data Quality Report")

    # Add an expander to explain the data quality score calculation
    with st.expander("How is the Data Quality Score calculated?"):
        st.markdown("""
            The Data Quality Score is a measure of data completeness for each property, starting from a base of 100 points. Points are deducted if key information is missing or marked as 'Unknown':
            <ul>
                <li><b>Price is missing or $0:</b> -30 points</li>
                <li><b>Property Type is 'Unknown':</b> -20 points</li>
                <li><b>Parish is 'Unknown':</b> -15 points</li>
                <li><b>Description is empty:</b> -15 points</li>
                <li><b>Bedrooms count is 0:</b> -10 points</li>
                <li><b>Bathrooms count is 0:</b> -10 points</li>
                <li><b>Category (e.g., For Sale, Rent) is 'Unknown':</b> -5 points</li>
                <li><b>Type (e.g., Residential, Commercial) is 'Unknown':</b> -5 points</li>
            </ul>
            The final score is capped between 0 and 100.
        """, unsafe_allow_html=True)

    if 'Data Quality Score' in df.columns and not df['Data Quality Score'].empty:
        # Ensure the column is numeric before calculating mean
        valid_scores = pd.to_numeric(df['Data Quality Score'], errors='coerce').dropna()
        if not valid_scores.empty:
             overall_score = valid_scores.mean()
             score_class = get_data_quality_class(overall_score)
             st.markdown(f"""
             <div style="background: var(--background-card); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                 <h3 style="margin-top: 0; color: var(--primary);">Overall Data Quality</h3>
                 <p style="font-size: 24px; margin-bottom: 0;">
                     <span class="{score_class}">{overall_score:.1f}/100</span>
                 </p>
                 <p style="color: var(--text-neutral); margin-top: 5px;">
                     Based on {len(df):,} property records in the current selection
                 </p>
             </div>
             """, unsafe_allow_html=True)
        else:
             st.caption("Overall Data Quality Score could not be calculated (no valid scores).")
    else:
        st.caption("Data Quality Score column is missing or empty.")

    st.markdown("**Column Completeness:**")
    completeness_data = []
    columns_to_check_post_clean = {
        'Price': lambda d: (pd.to_numeric(d.get('Price', 0), errors='coerce').fillna(0) == 0).sum(), # Ensure numeric check
        'Property Type': lambda d: (d.get('Property Type', 'Unknown') == 'Unknown').sum(),
        'Parish': lambda d: (d.get('Parish', 'Unknown') == 'Unknown').sum(),
        'Bedrooms': lambda d: (pd.to_numeric(d.get('Bedrooms', 0), errors='coerce').fillna(0) == 0).sum(), # Ensure numeric check
        'Bathrooms': lambda d: (pd.to_numeric(d.get('Bathrooms', 0), errors='coerce').fillna(0) == 0).sum(), # Ensure numeric check
        'Description': lambda d: (d.get('Description', '').astype(str).str.strip() == '').sum(),
        'Category': lambda d: (d.get('Category', 'Unknown') == 'Unknown').sum(),
        'Type': lambda d: (d.get('Type', 'Unknown') == 'Unknown').sum(),
    }
    total_rows = len(df)
    if total_rows > 0:
         for col_name, check_func in columns_to_check_post_clean.items():
             if col_name in df.columns:
                 try:
                     missing_count = check_func(df)
                     completeness_data.append({'Column': col_name, 'Missing Values': missing_count, 'Total': total_rows})
                 except Exception as e:
                     st.warning(f"Could not calculate completeness for column '{col_name}': {e}")


    if not completeness_data:
        st.caption("Could not generate column completeness report.")
    else:
        completeness = pd.DataFrame(completeness_data)
        completeness['% Complete'] = 100 * (1 - completeness['Missing Values'] / completeness['Total']) if total_rows > 0 else 0
        cols = st.columns(min(len(completeness), 4))
        for i, row in completeness.iterrows():
            if i < len(cols):
                with cols[i]:
                    metric_class = get_data_quality_class(row['% Complete'])
                    st.markdown(f"""
                    <div style="background: var(--background-card); padding: 15px; border-radius: 8px; margin-bottom: 10px;">
                        <p style="margin: 0 0 5px 0; font-weight: bold; color: var(--primary);">{row['Column']}</p>
                        <p style="margin: 0; font-size: 18px;"><span class="{metric_class}">{row['% Complete']:.1f}%</span></p>
                        <p style="margin: 5px 0 0 0; font-size: 12px; color: var(--text-neutral);">{row['Missing Values']} missing or unknown</p>
                    </div>""", unsafe_allow_html=True)


    st.markdown("**Data Quality Distribution:**")
    if 'Data Quality Score' in df.columns and not df['Data Quality Score'].empty:
        # Ensure the column is numeric for the histogram
        df['Data Quality Score'] = pd.to_numeric(df['Data Quality Score'], errors='coerce').fillna(0)
        fig = px.histogram(df, x='Data Quality Score', nbins=20, range_x=[0, 100], title='Distribution of Data Quality Scores', color_discrete_sequence=[THEME_PLOTLY['primary']])
        fig.update_layout(
            xaxis_title='Data Quality Score',
            yaxis_title='Number of Properties',
            paper_bgcolor=THEME_PLOTLY["paper_bgcolor"],
            plot_bgcolor=THEME_PLOTLY["plot_bgcolor"],
            font_color=THEME_PLOTLY["font_color"],
            yaxis=dict(gridcolor=THEME_PLOTLY["grid_color"]),
            xaxis=dict(gridcolor=THEME_PLOTLY["grid_color"])
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Insufficient data to show Data Quality Distribution.")
# ========== CHARTING FUNCTIONS ==========
def create_transaction_pie_charts(filtered_df, theme):
    if not filtered_df.empty and 'Category' in filtered_df.columns:
        trans_counts = filtered_df['Category'].value_counts()
        if not trans_counts.empty and trans_counts.sum() > 0: # Ensure there are counts before plotting
            pie_color_map_actual = {k: theme["category_colors"].get(k, theme["neutral_grey"]) for k in trans_counts.index}
            fig_trans = px.pie(trans_counts, names=trans_counts.index, values=trans_counts.values, color=trans_counts.index, color_discrete_map=pie_color_map_actual, hole=0.5, title="Overall Transaction Types")
            fig_trans.update_traces(textinfo='percent+label', hoverinfo='label+percent+value', marker=dict(line=dict(color=theme["paper_bgcolor"], width=2)))
            fig_trans.update_layout(title_x=0.5, paper_bgcolor=theme["paper_bgcolor"], plot_bgcolor=theme["plot_bgcolor"], font_color=theme["font_color"], legend_title_text='Category', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_trans, use_container_width=True)
        else:
            st.caption("No transaction data available for overall breakdown.")

        # The section for the 'For Sale vs. Rent Comparison' chart has been removed from here.

    else: # This else corresponds to the outer if `if not filtered_df.empty and 'Category' in filtered_df.columns:`
        st.caption("Insufficient data for transaction breakdown.")


def create_property_type_bar_chart(data_df, title, theme, palette_offset=0):
    if not data_df.empty and 'Property Type' in data_df.columns:
        counts = data_df['Property Type'].value_counts()
        if not counts.empty and counts.sum() > 0: # Ensure there are counts before plotting
            chart_palette = theme["bar_palette"]
            if palette_offset > 0 and len(chart_palette) > palette_offset:
                chart_palette = chart_palette[palette_offset:] + chart_palette[:palette_offset]
            # Repeat palette if not enough colors
            if len(counts) > len(chart_palette):
                chart_palette = chart_palette * (len(counts) // len(chart_palette) + 1)
            fig = px.bar(x=counts.index, y=counts.values, color=counts.index, color_discrete_sequence=chart_palette[:len(counts)], labels={'y': 'Count', 'x': 'Property Type'}, title=title) # Slice palette to match number of bars
            fig.update_layout(showlegend=False, title_x=0.5, xaxis_title=None, yaxis_title="Number of Properties", paper_bgcolor=theme["paper_bgcolor"], plot_bgcolor=theme["plot_bgcolor"], font_color=theme["font_color"], yaxis=dict(gridcolor=theme["grid_color"]), xaxis=dict(gridcolor=theme["grid_color"]))
            st.plotly_chart(fig, use_container_width=True)
        else: st.caption(f"No property type data for {title.lower().replace(' property types','')} properties.")
    else: st.caption(f"Insufficient data for {title.lower().replace(' property types','')} properties.")

def format_currency(value):
    if pd.isna(value) or value == 0: return "$0"
    if isinstance(value, (int, float)): return f"${value:,.0f}"
    return "N/A"

# ========== MAIN APPLICATION ==========
def main():
    ai_enabled = initialize_ai()

    with st.sidebar:
        logo_url = "https://www.terracaribbean.com/SiteAssets/terra_caribbean.png?"
        st.image(logo_url, width=200)

        # --- Display Weather ---
        # Check if the key was loaded AND weather data exists before trying to display
        if OPENWEATHERMAP_API_KEY:
            display_weather_sidebar(OPENWEATHERMAP_API_KEY)
            st.sidebar.markdown("---") # Optional separator
        else:
            st.sidebar.caption("Weather display disabled: API key missing.")
            st.sidebar.markdown("---") # Optional separator
        # --- End Weather Display -

        st.title("Filters & Tools")
        available_markets = list(MARKET_DATA_SOURCES.keys())
        selected_market = st.selectbox("Select Market", options=available_markets, index=available_markets.index('Barbados') if 'Barbados' in available_markets else 0, key="market_selector")
        if len(MARKET_DATA_SOURCES) == 1:
            st.info("More markets (Grenada, Trinidad, St. Lucia, etc.) planned!")


        # Ensure theme state is handled before reading config that depends on it
        current_theme_index = 0 if st.session_state.current_theme == 'Light' else 1
        theme_mode = st.radio("Select Theme", ['Light', 'Dark'], index=current_theme_index, key="theme_selector")
        if theme_mode != st.session_state.current_theme:
            st.session_state.current_theme = theme_mode
            st.rerun() # Rerun to apply theme changes immediately

        st.subheader("AI Search")
        if ai_enabled:
            nl_query = st.text_input(f"🔍 Ask about {selected_market} properties", placeholder="e.g., 'Beachfront villas under $1M'", help="Use natural language to search. Prices are in USD.", key="nl_query_input")
        else:
            nl_query = ""
            st.info("AI Search disabled (API key not set). Simple text search in data table.")

        st.subheader("Standard Filters")
        # Load data AFTER theme selector to potentially use theme in load_data (though not currently used)
        df_full = load_data(selected_market)
        if df_full.empty:
            st.warning(f"No data loaded for {selected_market}.")
            # Add the footer here as well if stopping early
            st.markdown(f"""---<div style="text-align: center; color: var(--text-neutral); font-size: 0.9em; padding-top:10px;"><p>Data source: <a href="https://www.terracaribbean.com" target="_blank" style="color: var(--accent);">Terra Caribbean</a></p><p>© {pd.Timestamp.now().year} Terra Caribbean Market Analytics • Prices in USD</p><p>Created by <b>Matthew Blackman</b> for Real Estate Data Analysis position</p><p>Assisted by <b>DeepSeek AI</b></p></div>""", unsafe_allow_html=True)
            st.stop()

        property_types_options = sorted(df_full['Property Type'].astype(str).unique()) if 'Property Type' in df_full.columns else []
        property_type = st.multiselect('Property Type', options=property_types_options, default=property_types_options, key="filter_property_type")
        categories_options = sorted(df_full['Category'].astype(str).unique()) if 'Category' in df_full.columns else []
        category = st.multiselect('Transaction Type', options=categories_options, default=categories_options, key="filter_category")
        parishes_available = sorted(df_full['Parish'].astype(str).unique()) if 'Parish' in df_full.columns else []
        default_parishes = [p for p in parishes_available if p != 'Unknown'] if 'Unknown' in parishes_available and len(parishes_available) > 1 else parishes_available
        parish = st.multiselect('Parish', options=parishes_available, default=default_parishes, key="filter_parish")

        price_min_val, price_max_val = 0.0, 1.0 # Initialize with a small valid range
        if 'Price' in df_full.columns:
            valid_prices = pd.to_numeric(df_full['Price'], errors='coerce').dropna()
            if not valid_prices.empty:
                # Use .agg for more robust calculation that handles NaNs after dropna
                price_min_val = float(valid_prices.agg('min'))
                price_max_val = float(valid_prices.agg('max'))
                # Ensure max is greater than min for the slider to work
                if price_min_val == price_max_val:
                    price_max_val += 1.0 # Add a small value if all prices are the same
                # Add a buffer to the max value if it's large, for better slider usability
                elif price_max_val > 100000:
                     price_max_val = price_max_val * 1.05 # Add 5% buffer for large values


        # Ensure default value tuple is within the adjusted min/max range
        default_price_range = (price_min_val, price_max_val)
        # Adjust step size dynamically based on range
        price_step = max(1.0, (price_max_val - price_min_val) / 200) # Ensure step is at least 1
        price_range = st.slider('Price Range (USD)', min_value=price_min_val, max_value=price_max_val, value=default_price_range, step=price_step, format="$%.0f", key="filter_price_range")


        st.subheader("Map Features")
        available_amenities = list(MARKET_DATA_SOURCES.get(selected_market, {}).get('amenities', {}).keys())
        # This variable is named 'show_amenities'
        show_amenities = st.multiselect("Show Amenities", options=available_amenities, default=[a for a in ['Beach', 'School', 'Restaurant'] if a in available_amenities], key="map_show_amenities")
        # Only show school zones option if 'School' amenity exists for the market
        show_school_zones = st.checkbox("Show School Zones (approx. 1km radius)", value=False, key="map_show_school_zones") if 'School' in available_amenities else False


        st.subheader("Data Quality")
        min_quality = st.slider("Minimum Data Quality Score", 0, 100, 50, 5, key="filter_min_quality") if 'Data Quality Score' in df_full.columns else 0
        if 'Data Quality Score' not in df_full.columns:
            st.info("'Data Quality Score' column missing from data.")


    # --- Apply Filters ---
    # Geocode the full dataset once after loading and caching
    df_full_geocoded = geocode_properties(df_full.copy(), selected_market) # Pass a copy

    # Ensure necessary columns exist in the geocoded dataframe before filtering
    filter_cols = ['Property Type', 'Category', 'Parish', 'Price', 'Data Quality Score', 'Type']
    for col in filter_cols:
        if col not in df_full_geocoded.columns:
            default_val = 'Unknown' if col in ['Property Type', 'Category', 'Parish', 'Type'] else 0 if col == 'Price' else 100
            df_full_geocoded[col] = default_val # Add missing column with default value


    # Apply standard filters
    prop_types_filter = df_full_geocoded['Property Type'].astype(str).isin(property_type) if property_type else pd.Series(True, index=df_full_geocoded.index)
    category_filter = df_full_geocoded['Category'].astype(str).isin(category) if category else pd.Series(True, index=df_full_geocoded.index)
    # Handle empty parish selection - should select all available parishes including 'Unknown' if present
    parish_filter = df_full_geocoded['Parish'].astype(str).isin(parish if parish else parishes_available) if (parish or parishes_available) else pd.Series(True, index=df_full_geocoded.index)


    # Ensure price and quality columns are numeric before comparison
    df_full_geocoded['Price'] = pd.to_numeric(df_full_geocoded['Price'], errors='coerce').fillna(0)
    df_full_geocoded['Data Quality Score'] = pd.to_numeric(df_full_geocoded['Data Quality Score'], errors='coerce').fillna(0)


    filtered_df = df_full_geocoded[
        prop_types_filter &
        category_filter &
        parish_filter &
        (df_full_geocoded['Price'] >= price_range[0]) &
        (df_full_geocoded['Price'] <= price_range[1]) &
        (df_full_geocoded['Data Quality Score'] >= min_quality)
    ].copy() # Use .copy() after filtering


    # Apply Natural Language Query/Search Filter
    if nl_query:
        if ai_enabled:
            # Pass a copy to NL query to prevent modifying the DataFrame in place if eval code does so
            filtered_df_nl = natural_language_query(query=nl_query, df=filtered_df.copy())
            if filtered_df_nl is not None and not filtered_df_nl.empty:
                filtered_df = filtered_df_nl.copy() # Use .copy() after AI filter
                st.success(f"Found {len(filtered_df):,} properties matching your query and standard filters.")
            elif len(filtered_df) > 0: # Standard filters yielded results, but NL query did not subset them
                 st.warning("NL query returned no results within current filters. Displaying standard filter results.")
            else: # Standard filters yielded no results, so NL query on top of that also yielded none
                 st.warning("No properties found matching your NL query or standard filters.")
        else: # AI not enabled, simple search
            # Pass a copy to text search
            filtered_df_search = apply_search_filter(df=filtered_df.copy(), search_term=nl_query)
            if filtered_df_search is not None and not filtered_df_search.empty:
                filtered_df = filtered_df_search.copy() # Use .copy() after text search
                st.info(f"Simple text search found {len(filtered_df):,} properties within current filters.")
            elif len(filtered_df) > 0: # Standard filters yielded results, but text search did not subset them
                st.warning("Text search returned no results within current filters. Displaying standard filter results.")
            else: # Standard filters yielded no results, so text search on top of that also yielded none
                st.warning("No properties found matching your text search or standard filters.")

    # --- Display Main Content ---
    st.title(f"🏝️ Terra Caribbean Property Intelligence")
    # Use filtered_df for the count displayed here
    st.markdown(f"""<span style="color:var(--text-neutral); font-size: 1.1em;">Professional market insights by Terra Caribbean | <b>{selected_market} Market</b> | <b>{len(filtered_df):,}</b> properties analyzed | All prices in USD</span>""", unsafe_allow_html=True)


    if ai_enabled:
        with st.expander(f"💡 AI-Powered Market Insights for {selected_market}", expanded=True):
            # Check if there are enough properties *in the filtered set* before offering to generate insights
            if len(filtered_df) < 5:
                 st.info(f"Need at least 5 properties in the current selection ({len(filtered_df)}) to generate meaningful AI insights.")
            elif st.button(f"Generate Insights for Current Selection ({len(filtered_df)} Properties)", key="generate_insights_button"):
                 with st.spinner(f"Analyzing {len(filtered_df)} properties..."):
                    # Pass filtered data to AI, and the full geocoded data for context if needed
                    insights = generate_ai_insights(filtered_df=filtered_df, full_df=df_full_geocoded, market_name=selected_market)
                    st.markdown(insights)
            else:
                 st.info("Click button above to generate AI insights for the current selection (Prices are in USD).")

    else:
        st.info("AI Insights disabled (OpenAI API key not set in secrets).")


    st.subheader(f'📊 {selected_market} Market Overview')
    total_props = len(filtered_df)
    # Ensure 'Type' column exists before filtering
    res_count = len(filtered_df[filtered_df.get('Type', '') == 'Residential']) if 'Type' in filtered_df.columns else 0
    com_count = len(filtered_df[filtered_df.get('Type', '') == 'Commercial']) if 'Type' in filtered_df.columns else 0
    # Ensure 'Price' column exists and is numeric before calculating max
    high_price = pd.to_numeric(filtered_df.get('Price', pd.Series()), errors='coerce').dropna().max() if 'Price' in filtered_df.columns and not filtered_df['Price'].empty else 0 # Use .get safely
    # Ensure 'Data Quality Score' column exists and is numeric before calculating mean
    avg_qual = pd.to_numeric(filtered_df.get('Data Quality Score', pd.Series()), errors='coerce').dropna().mean() if 'Data Quality Score' in filtered_df.columns and not filtered_df['Data Quality Score'].empty else 0 # Use .get safely

    metrics_data = [
        ("Total Properties", f"{total_props:,}"),
        ("Residential", f"{res_count:,}"),
        ("Commercial", f"{com_count:,}"),
        ("Highest Price (USD)", format_currency(high_price)),
        ("Avg Data Quality", f"{avg_qual:.1f}/100" if total_props > 0 and not np.isnan(avg_qual) else "N/A") # Handle division by zero and NaN
    ]
    cols_metrics = st.columns(len(metrics_data))
    for i, (label, value) in enumerate(metrics_data):
        with cols_metrics[i]:
            st.markdown(f'<div class="metric-card"><h3>{label}</h3><p>{value}</p></div>', unsafe_allow_html=True)


# ========== STORYTELLING SECTION ==========
    st.subheader('📈 Market Insights & Storytelling')

    # Create columns for layout
    col1, col2 = st.columns([0.6, 0.4])

    with col1:
        # Price Distribution Analysis
        # Check if filtered_df is not empty, Price column exists and has valid data for plotting
        if not filtered_df.empty and 'Price' in filtered_df.columns:
            valid_prices = pd.to_numeric(filtered_df['Price'], errors='coerce').dropna()
            if not valid_prices.empty and valid_prices.sum() > 0:
                st.markdown("### Price Distribution Analysis")

                # Calculate key metrics
                avg_price = valid_prices.mean()
                median_price = valid_prices.median()
                price_25th = valid_prices.quantile(0.25)
                price_75th = valid_prices.quantile(0.75)

                # Create histogram with box plot
                fig = px.histogram(filtered_df, x='Price',
                                   title='Property Price Distribution',
                                   labels={'Price': 'Price (USD)'},
                                   nbins=20,
                                   color_discrete_sequence=[THEME_PLOTLY['primary']])

                # Add reference lines only if metrics are meaningful (i.e., valid_prices not empty)
                fig.add_vline(x=avg_price, line_dash="dash", line_color="red",
                              annotation_text=f"Avg: ${avg_price:,.0f}",
                              annotation_position="top right") # Adjusted position
                fig.add_vline(x=median_price, line_dash="dash", line_color="green",
                              annotation_text=f"Median: ${median_price:,.0f}",
                              annotation_position="top left") # Adjusted position

                fig.update_layout(
                    paper_bgcolor=THEME_PLOTLY["paper_bgcolor"],
                    plot_bgcolor=THEME_PLOTLY["plot_bgcolor"],
                    font_color=THEME_PLOTLY["font_color"],
                    yaxis=dict(gridcolor=THEME_PLOTLY["grid_color"]),
                    xaxis=dict(gridcolor=THEME_PLOTLY["grid_color"])
                )

                st.plotly_chart(fig, use_container_width=True)

                # Price insights text
                st.markdown(f"""
                - The average property price is **${avg_price:,.0f} USD**, while the median is **${median_price:,.0f} USD**
                - 25% of properties are priced below **${price_25th:,.0f} USD**
                - 25% of properties are priced above **${price_75th:,.0f} USD**
                - The price range spans from **${valid_prices.min():,.0f} USD** to **${valid_prices.max():,.0f} USD**
                """)
            else:
                st.caption("Insufficient valid price data for distribution analysis.")
        else: st.caption("Insufficient data or no price data for distribution analysis.")


    with col2:
        # Top Parish Analysis
        if not filtered_df.empty and 'Parish' in filtered_df.columns:
            st.markdown("### Parish Distribution")

            parish_counts = filtered_df['Parish'].value_counts()

            if not parish_counts.empty and parish_counts.sum() > 0:
                # Calculate total properties for percentage calculation
                total_filtered = len(filtered_df)
                # Get top 5 parishes, excluding 'Unknown' if it's present but not dominant
                parish_counts_top = parish_counts[parish_counts.index != 'Unknown'].nlargest(5)
                if parish_counts_top.empty and 'Unknown' in parish_counts.index:
                     # If all properties are 'Unknown', show that
                     parish_counts_top = parish_counts[['Unknown']]
                elif parish_counts_top.empty:
                     st.caption("No parish data available for visualization.")


                if not parish_counts_top.empty and parish_counts_top.sum() > 0:
                     fig = px.pie(parish_counts_top,
                                   names=parish_counts_top.index,
                                   values=parish_counts_top.values,
                                   title='Top Parishes by Property Count', # Updated title
                                   color=parish_counts_top.index, # Use index for color mapping
                                   color_discrete_sequence=THEME_PLOTLY["bar_palette"])

                     # Adjust text position based on number of slices to avoid overlap
                     text_position = 'inside' if len(parish_counts_top) <= 5 else 'outside'
                     fig.update_traces(textposition=text_position, textinfo='percent+label')
                     fig.update_layout(
                         paper_bgcolor=THEME_PLOTLY["paper_bgcolor"],
                         plot_bgcolor=THEME_PLOTLY["plot_bgcolor"],
                         font_color=THEME_PLOTLY["font_color"],
                         showlegend=False,
                         title_x=0.5 # Center title
                         )


                     st.plotly_chart(fig, use_container_width=True)

                     # Parish insights text
                     if 'Unknown' in parish_counts_top.index:
                         st.markdown(f"""
                         - Note: {parish_counts_top['Unknown'] if 'Unknown' in parish_counts_top.index else 0} properties have an unknown parish.
                         """)
                     elif not parish_counts_top.empty:
                         st.markdown(f"""
                         - **{parish_counts_top.index[0]}** has the most properties ({parish_counts_top.values[0]}) among known parishes.
                         - The top {len(parish_counts_top)} parishes shown account for {parish_counts_top.sum()/total_filtered*100:.0f}% of all filtered properties.
                         """)
                else:
                     st.caption("No significant parish data available for visualization.")
            else: st.caption("No parish data available for visualization.")
        else: st.caption("Insufficient data for parish distribution.")


    # Property Type vs Price Analysis
    # Check if filtered_df is not empty, and required columns exist and have data
    if not filtered_df.empty and 'Property Type' in filtered_df.columns and 'Price' in filtered_df.columns:
        # Calculate average price by property type, only for properties with Price > 0
        avg_price_by_type = filtered_df[pd.to_numeric(filtered_df['Price'], errors='coerce').fillna(0) > 0].groupby('Property Type')['Price'].mean().sort_values(ascending=False) # Ensure numeric check before filter

        if len(avg_price_by_type) > 0:
            st.markdown("### Property Type vs Price")
            fig = px.bar(avg_price_by_type,
                         x=avg_price_by_type.values,
                         y=avg_price_by_type.index,
                         orientation='h',
                         title='Average Price by Property Type (USD)',
                         color=avg_price_by_type.index,
                         color_discrete_sequence=THEME_PLOTLY["bar_palette"])

            fig.update_layout(
                paper_bgcolor=THEME_PLOTLY["paper_bgcolor"],
                plot_bgcolor=THEME_PLOTLY["plot_bgcolor"],
                font_color=THEME_PLOTLY["font_color"],
                yaxis=dict(gridcolor=THEME_PLOTLY["grid_color"], title='Property Type'), # Add Y axis title
                xaxis=dict(gridcolor=THEME_PLOTLY["grid_color"], title='Average Price (USD)'), # Add X axis title
                showlegend=False,
                title_x=0.5 # Center title
            )
            # Improve hover format for price
            fig.update_traces(hovertemplate='<b>%{y}</b><br>Avg Price: $%{x:,.0f} USD')


            st.plotly_chart(fig, use_container_width=True)

            # Property type insights text
            if len(avg_price_by_type) >= 2:
                 st.markdown(f"""
                 - **{avg_price_by_type.index[0]}** commands the highest average price at **${avg_price_by_type.values[0]:,.0f} USD**
                 - The most affordable property type (among those with sales data) is **{avg_price_by_type.index[-1]}** at **${avg_price_by_type.values[-1]:,.0f} USD**
                 """)
            elif len(avg_price_by_type) == 1:
                 st.markdown(f"""
                 - Only one property type with price data found: **{avg_price_by_type.index[0]}** with an average price of **${avg_price_by_type.values[0]:,.0f} USD**.
                 """)
            else:
                 st.caption("No property types with valid price data found.")

        else: st.caption("No property types with valid price data found for analysis.")
    else: st.caption("Insufficient data for property type vs price analysis.")


    # Bedroom Analysis
    # Check if filtered_df is not empty and Bedrooms column exists
    if not filtered_df.empty and 'Bedrooms' in filtered_df.columns:
        st.markdown("### Bedroom Analysis")

        # Filter out properties with 0 bedrooms for this analysis unless it's the *only* data
        bedroom_counts = filtered_df[pd.to_numeric(filtered_df['Bedrooms'], errors='coerce').fillna(0) > 0]['Bedrooms'].astype(int).value_counts().sort_index() # Ensure numeric then int, then counts

        if len(bedroom_counts) > 0:
            fig = px.line(x=bedroom_counts.index,
                          y=bedroom_counts.values,
                          title='Property Count by Number of Bedrooms',
                          markers=True)

            fig.update_layout(
                xaxis_title="Number of Bedrooms",
                yaxis_title="Number of Properties",
                paper_bgcolor=THEME_PLOTLY["paper_bgcolor"],
                plot_bgcolor=THEME_PLOTLY["plot_bgcolor"],
                font_color=THEME_PLOTLY["font_color"],
                yaxis=dict(gridcolor=THEME_PLOTLY["grid_color"]),
                xaxis=dict(gridcolor=THEME_PLOTLY["grid_color"], tickmode='linear') # Force integer ticks
                )

            st.plotly_chart(fig, use_container_width=True)

            # Bedroom insights text
            most_common_bedrooms = bedroom_counts.idxmax()
            st.markdown(f"""
            - The most common configuration (among properties with >0 bedrooms) is **{int(most_common_bedrooms)}-bedroom** properties.
            - {bedroom_counts[most_common_bedrooms]} properties ({bedroom_counts[most_common_bedrooms]/len(filtered_df)*100:.0f}%) have this configuration.
            """)
        elif len(filtered_df) > 0:
             # Handle case where only 0-bedroom properties exist in the filtered set
             zero_bed_count = len(filtered_df[pd.to_numeric(filtered_df['Bedrooms'], errors='coerce').fillna(0) == 0]) # Ensure numeric check
             if zero_bed_count > 0:
                  st.caption(f"Only {zero_bed_count} properties with 0 bedrooms found in the current selection.")
             else:
                  st.caption("No bedroom data available for visualization.")
        else:
             st.caption("Insufficient data for bedroom analysis.")



    st.subheader(f'🌍 Interactive Property Map - {selected_market}')
    st.caption("📍 Please note: Property markers on the map indicate the approximate location within their respective parish, not the exact street address. The jitter added to markers is for visualization purposes to distinguish closely located properties.")

    # Only create and display map if there are properties to show OR amenities are selected
    if not filtered_df.empty or show_amenities or show_school_zones:
        # Pass 'show_amenities' to the map creation function where it's received as 'amenities_to_show'
        advanced_map_obj = create_advanced_map(filtered_df=filtered_df, island_name=selected_market, amenities_to_show=show_amenities, show_school_districts=show_school_zones)
        if advanced_map_obj:
             # Ensure Folium map is rendered
            folium_static(advanced_map_obj, width=1200, height=600)

            # --- MODIFIED LEGEND START ---
            # This section is in the 'main' function's scope,
            # so it must use the variable defined in this scope, which is 'show_amenities'.
            legend_items = []
            filtered_categories = filtered_df['Category'].astype(str).str.strip().str.upper().unique() if 'Category' in filtered_df.columns else []

            if 'FOR SALE' in filtered_categories or any(cat in filtered_categories for cat in ['FOR SALE', 'FOR SALE']): legend_items.append('<div class="map-legend-item"><div class="map-legend-color-box" style="background: red;"></div><i class="fa fa-home map-legend-icon" style="color: red;"></i> <span>For Sale</span></div>')
            if 'FOR RENT' in filtered_categories or any(cat in filtered_categories for cat in ['FOR RENT', 'FOR RENT']): legend_items.append('<div class="map-legend-item"><div class="map-legend-color-box" style="background: green;"></div><i class="fa fa-home map-legend-icon" style="color: green;"></i> <span>For Rent</span></div>')
            # Show Other Status if there are categories that are neither 'FOR SALE' nor 'FOR RENT'
            other_categories = [cat for cat in filtered_categories if cat not in ['FOR SALE', 'FOR RENT']]
            if other_categories: legend_items.append('<div class="map-legend-item"><div class="map-legend-color-box" style="background: gray;"></div><i class="fa fa-home map-legend-icon" style="color: gray;"></i> <span>Other Status</span></div>')


            # Use 'show_amenities' from main function's scope
            if show_school_zones and 'School' in MARKET_DATA_SOURCES.get(selected_market, {}).get('amenities', {}):
                legend_items.append('<div class="map-legend-item"><div class="map-legend-color-box" style="background: orange;"></div><i class="glyphicon glyphicon-info-sign map-legend-icon" style="color: orange;"></i> <span>Schools / Zones</span></div>')
            # Use 'show_amenities' from main function's scope
            if 'Restaurant' in (show_amenities or []) and 'Restaurant' in MARKET_DATA_SOURCES.get(selected_market, {}).get('amenities', {}):
                 legend_items.append('<div class="map-legend-item"><div class="map-legend-color-box" style="background: red;"></div> <i class="glyphicon glyphicon-cutlery map-legend-icon" style="color: red;"></i> <span>Restaurants</span></div>')
            # Use 'show_amenities' from main function's scope
            if 'Beach' in (show_amenities or []) and 'Beach' in MARKET_DATA_SOURCES.get(selected_market, {}).get('amenities', {}):
                 legend_items.append('<div class="map-legend-item"><div class="map-legend-color-box" style="background: lightblue;"></div><i class="glyphicon glyphicon-tint map-legend-icon" style="color: lightblue;"></i> <span>Beaches</span></div>')


            if legend_items:
                 st.markdown(f"""
                 <div style="display: flex; flex-wrap: wrap; gap: 15px; margin-top: 10px; justify-content: center; align-items: center;">
                 {''.join(legend_items)}
                 </div>
                 """, unsafe_allow_html=True)
            # --- MODIFIED LEGEND END ---
        else:
            st.caption("Could not generate map.") # Fallback if create_advanced_map returns None
    else:
        st.caption(f"No properties or selected amenities/zones to display on map for current filters in {selected_market}.")


    show_data_quality_report(df=filtered_df)

    st.subheader('📑 Transaction Type Distribution')
    # Check if there's enough data to justify columns, otherwise use a single column layout
    if not filtered_df.empty and 'Category' in filtered_df.columns and 'Type' in filtered_df.columns:
        # Check if there's more than one unique value in both 'Type' and 'Category' to justify a crosstab and two columns
        if filtered_df['Type'].nunique() > 1 and filtered_df['Category'].nunique() > 1:
             col_trans1, col_trans2 = st.columns(2)
             with col_trans1:
                  # create_transaction_pie_charts now only plots the first chart
                  create_transaction_pie_charts(filtered_df=filtered_df, theme=THEME_PLOTLY)
             with col_trans2:
                  type_trans = pd.crosstab(filtered_df['Type'], filtered_df['Category'])
                  if not type_trans.empty and type_trans.values.sum() > 0:
                      bar_color_map_actual = {k: THEME_PLOTLY["category_colors"].get(k, THEME_PLOTLY["neutral_grey"]) for k in type_trans.columns}
                      fig_type_trans = px.bar(type_trans, barmode='group', color_discrete_map=bar_color_map_actual, labels={'value': 'Count', 'variable': 'Transaction Type', 'Type': 'Property Category'}, title="Transactions by Property Category")
                      fig_type_trans.update_layout(title_x=0.5, paper_bgcolor=THEME_PLOTLY["paper_bgcolor"], plot_bgcolor=THEME_PLOTLY["plot_bgcolor"], font_color=THEME_PLOTLY["font_color"], legend_title_text='Transaction Type', xaxis_title=None, yaxis=dict(gridcolor=THEME_PLOTLY["grid_color"]), xaxis=dict(gridcolor=THEME_PLOTLY["grid_color"]))
                      st.plotly_chart(fig_type_trans, use_container_width=True)
                  else: st.caption("No breakdown data for current selection.")
        else:
             # Single column or no charts if not enough data variety
             # create_transaction_pie_charts now only plots the first chart
             create_transaction_pie_charts(filtered_df=filtered_df, theme=THEME_PLOTLY)
             if filtered_df['Type'].nunique() > 0 and filtered_df['Category'].nunique() > 0: # Check if crosstab is possible at all
                  type_trans = pd.crosstab(filtered_df['Type'], filtered_df['Category'])
                  if not type_trans.empty and type_trans.values.sum() > 0:
                       bar_color_map_actual = {k: THEME_PLOTLY["category_colors"].get(k, THEME_PLOTLY["neutral_grey"]) for k in type_trans.columns}
                       fig_type_trans = px.bar(type_trans, barmode='group', color_discrete_map=bar_color_map_actual, labels={'value': 'Count', 'variable': 'Transaction Type', 'Type': 'Property Category'}, title="Transactions by Property Category")
                       fig_type_trans.update_layout(title_x=0.5, paper_bgcolor=THEME_PLOTLY["paper_bgcolor"], plot_bgcolor=THEME_PLOTLY["plot_bgcolor"], font_color=THEME_PLOTLY["font_color"], legend_title_text='Transaction Type', xaxis_title=None, yaxis=dict(gridcolor=THEME_PLOTLY["grid_color"]), xaxis=dict(gridcolor=THEME_PLOTLY["grid_color"]))
                       st.plotly_chart(fig_type_trans, use_container_width=True)
                  else: st.caption("No breakdown data for current selection.")
             else:
                  st.caption("Insufficient data for breakdown.")

    else: st.caption("Insufficient data for transaction breakdown.")


    st.subheader('🏘️ Property Type Breakdown')
    # Check if there's enough data to justify columns for Residential/Commercial breakdown
    has_residential = not filtered_df[filtered_df.get('Type', '').astype(str).str.lower() == 'residential'].empty if 'Type' in filtered_df.columns else False
    has_commercial = not filtered_df[filtered_df.get('Type', '').astype(str).str.lower() == 'commercial'].empty if 'Type' in filtered_df.columns else False

    if has_residential and has_commercial:
        col_prop1, col_prop2 = st.columns(2)
        with col_prop1:
             st.markdown(f"<h4 style='text-align: center; color: {THEME_PLOTLY['primary']};'>Residential Properties</h4>", unsafe_allow_html=True)
             res_df = filtered_df[filtered_df.get('Type', '').astype(str).str.lower() == 'residential'].copy()
             create_property_type_bar_chart(data_df=res_df, title="Residential Property Types", theme=THEME_PLOTLY)
        with col_prop2:
             st.markdown(f"<h4 style='text-align: center; color: {THEME_PLOTLY['primary']};'>Commercial Properties</h4>", unsafe_allow_html=True)
             com_df = filtered_df[filtered_df.get('Type', '').astype(str).str.lower() == 'commercial'].copy()
             create_property_type_bar_chart(data_df=com_df, title="Commercial Property Types", theme=THEME_PLOTLY, palette_offset=1)
    elif has_residential:
         st.markdown(f"<h4 style='text-align: center; color: {THEME_PLOTLY['primary']};'>Residential Properties</h4>", unsafe_allow_html=True)
         res_df = filtered_df[filtered_df.get('Type', '').astype(str).str.lower() == 'residential'].copy()
         create_property_type_bar_chart(data_df=res_df, title="Residential Property Types", theme=THEME_PLOTLY)
         st.caption("No commercial properties in current selection for breakdown.")
    elif has_commercial:
         st.markdown(f"<h4 style='text-align: center; color: {THEME_PLOTLY['primary']};'>Commercial Properties</h4>", unsafe_allow_html=True)
         com_df = filtered_df[filtered_df.get('Type', '').astype(str).str.lower() == 'commercial'].copy()
         create_property_type_bar_chart(data_df=com_df, title="Commercial Property Types", theme=THEME_PLOTLY, palette_offset=1)
         st.caption("No residential properties in current selection for breakdown.")
    else:
        st.caption("No residential or commercial properties in current selection for breakdown.")


    st.subheader('📋 Property Data Preview')
    if not filtered_df.empty:
        display_df = filtered_df.copy()
        # Ensure columns exist before formatting
        if 'Price' in display_df.columns:
             # Format Price, handle NaN and 0 safely
             display_df['Price'] = pd.to_numeric(display_df['Price'], errors='coerce').apply(lambda x: f"${x:,.0f}" if pd.notna(x) and x > 0 else "$0")
        else:
             display_df['Price'] = "N/A"

        if 'Data Quality Score' in display_df.columns:
             # Format Data Quality, handle NaN safely
             display_df['Data Quality'] = pd.to_numeric(display_df['Data Quality Score'], errors='coerce').apply(lambda x: f"<span class='{get_data_quality_class(x)}'>{x:.0f}/100</span>" if pd.notna(x) else "N/A")
        else:
             display_df['Data Quality'] = "N/A (Column Missing)"

        # Ensure other potential display columns are strings to avoid errors in to_html
        for col in ['Name', 'Type', 'Category', 'Parish', 'Property Type', 'Description']:
            if col in display_df.columns:
                display_df[col] = display_df[col].astype(str).fillna('N/A')
            elif col in filtered_df.columns: # If column was in filtered_df but not display_df (e.g. due to earlier drop)
                 display_df[col] = filtered_df[col].astype(str).fillna('N/A')
            else:
                 display_df[col] = 'N/A' # Add if completely missing

        # Format Bed/Bath as integers, handle NaNs
        for col in ['Bedrooms', 'Bathrooms']:
             if col in display_df.columns:
                  display_df[col] = pd.to_numeric(display_df[col], errors='coerce').fillna(0).astype(int)
             elif col in filtered_df.columns:
                  display_df[col] = pd.to_numeric(filtered_df[col], errors='coerce').fillna(0).astype(int)
             else:
                  display_df[col] = 0


        display_cols_order = ['Name', 'Type', 'Category', 'Parish', 'Property Type', 'Price', 'Bedrooms', 'Bathrooms', 'Data Quality', 'Description']
        # Filter to only include columns actually present in the display_df copy
        display_columns = [col for col in display_cols_order if col in display_df.columns]

        # Check if there are columns left to display after selection
        if not display_columns:
             st.caption("No relevant columns available to display in the data preview.")
        else:
             # Convert to HTML table - use .copy() to avoid SettingWithCopyWarning if modifications were made earlier
             html_table = display_df[display_columns].to_html(escape=False, index=False, classes='dataframe table table-striped table-hover', border=0)
             html_table = f"""<div style="max-height: 500px; overflow-y: auto;">{html_table}</div>""" # Wrap in div for scrolling
             st.write(html_table, unsafe_allow_html=True)

             # Prepare DataFrame for download - start from filtered_df before display formatting
             cols_to_drop_dl = ['Data Quality', 'lat', 'lon', 'Data Quality Score', 'coords'] # Drop columns added for display/map
             # Use errors='ignore' to not fail if column wasn't in filtered_df anyway, and .copy()
             download_df = filtered_df.drop(columns=[col for col in cols_to_drop_dl if col in filtered_df.columns], errors='ignore').copy()
             # Price column might have been formatted as string for display, but we drop Data Quality column, so original Price is kept.
             # If there were other steps formatting price *before* the display_df copy, we might need re-conversion here, but as currently structured, filtered_df['Price'] should be numeric or NaN.
             # Add explicit re-conversion for safety if needed, but the current flow keeps it numeric.

             st.download_button(
                 label=f"📥 Download {selected_market} Data as CSV",
                 data=download_df.to_csv(index=False).encode('utf-8'),
                 file_name=f'{selected_market.lower().replace(" ", "_")}_terra_caribbean_properties.csv',
                 mime='text/csv'
             )
    else: st.caption(f"No property data for current filters in {selected_market} to display.")

    # Ensure this footer is always at the bottom unless stopped early
    st.markdown(f"""---<div style="text-align: center; color: var(--text-neutral); font-size: 0.9em; padding-top:10px;"><p>Data source: <a href="https://www.terracaribbean.com" target="_blank" style="color: var(--accent);">Terra Caribbean</a> • {len(filtered_df):,} properties analyzed</p><p>© {pd.Timestamp.now().year} Terra Caribbean Market Analytics • Prices in USD</p><p>Created by <b>Matthew Blackman</b> for Real Estate Data Analysis position</p><p>Assisted by <b>DeepSeek AI</b></p></div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
