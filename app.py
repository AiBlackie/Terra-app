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
# from geopy.geocoders import Nominatim # Not actively used
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
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
MAPBOX_ACCESS_TOKEN = st.secrets.get("MAPBOX_ACCESS_TOKEN", "")
OPENWEATHERMAP_API_KEY = st.secrets.get("OPENWEATHERMAP_API_KEY", "")

# Define market-specific data sources and base configs for scalability
MARKET_DATA_SOURCES = {
    'Barbados': {
        'file': "Terra Caribbean NEW SAMPLE R.xlsx", # EXPECTING AN EXCEL FILE NOW
        'header_row': 3,  # Header row for the Excel file
        'default_coords': (13.1939, -59.5432),
        'parish_coords': { # Canonical Names used as keys
            'St. James': (13.1850, -59.6300),
            'Christ Church': (13.0770, -59.5300),
            'St. Andrew': (13.2360, -59.5685),
            'St. George': (13.1500, -59.5500),
            'St. John': (13.1720, -59.4900),
            'St. Joseph': (13.2020, -59.5250),
            'St. Lucy': (13.2850, -59.6100),
            'St. Michael': (13.1050, -59.6100),
            'St. Peter': (13.2450, -59.6300),
            'St. Philip': (13.1200, -59.4750),
            'St. Thomas': (13.1800, -59.5850),
            'Unknown': (13.1939, -59.5432) # Default for unknown parishes
        },
        'amenities': {
            'Beach': [{'name': 'Crane Beach', 'lat': 13.0986, 'lon': -59.4485}, {'name': 'Miami Beach', 'lat': 13.0833, 'lon': -59.5333}, {'name': 'Carlisle Bay', 'lat': 13.0778, 'lon': -59.6142}],
            'School': [{'name': 'Harrison College', 'lat': 13.0978, 'lon': -59.6144}, {'name': "Queen's College", 'lat': 13.0953, 'lon': -59.6169}],
            'Restaurant': [{'name': 'The Cliff', 'lat': 13.1800, 'lon': -59.6389}, {'name': 'Lone Star Restaurant', 'lat': 13.1975, 'lon': -59.6414}]
        }
    }
}

# ========== THEME DEFINITIONS ==========
if 'current_theme' not in st.session_state:
    st.session_state.current_theme = 'Light'

light_theme_plotly = {
    "primary": "#1a3e72", "secondary": "#e9a229", "accent": "#5f8fd9", "neutral_grey": "#bdc3c7",
    "paper_bgcolor": "#ffffff", "plot_bgcolor": "#f5f7fa", "font_color": "#333333", "grid_color": "#e0e0e0",
    "map_bubble_color": "#00BCD4", "map_text_color": "#000000",
    "category_colors": {'FOR SALE': '#e74c3c', 'FOR RENT': '#3498db', 'SOLD': '#27ae60', 'LEASED': '#2980b9', 'Sold': '#27ae60', 'Leased': '#2980b9', 'For Sale': '#e74c3c', 'Rent': '#3498db', 'Unknown': '#bdc3c7'},
    "bar_palette": ["#1a3e72", "#e9a229", "#5f8fd9", "#2c5282", "#f6e05e", "#7f8fd9", "#4a5568", "#a0aec0", "#667eea", "#ed8936"]
}
dark_theme_plotly = {
    "primary": "#76a2dd", "secondary": "#f7ca4f", "accent": "#e9a229", "neutral_grey": "#95a5a6",
    "paper_bgcolor": "#2d394a", "plot_bgcolor": "#222b38", "font_color": "#ecf0f1", "grid_color": "#4a6378",
    "map_bubble_color": "#00ACC1", "map_text_color": "#ffffff",
    "category_colors": {'FOR SALE': '#e74c3c', 'FOR RENT': '#76a2dd', 'SOLD': '#32c878', 'LEASED': '#5dade2', 'Sold': '#32c878', 'Leased': '#5dade2', 'For Sale': '#e74c3c', 'Rent': '#76a2dd', 'Unknown': '#95a5a6'},
    "bar_palette": ["#76a2dd", "#f7ca4f", "#A7C7E7", "#4e7ab5", "#fff176", "#a6bde7", "#718096", "#d3dce6", "#9ab5f9", "#f0a560"]
}
# This global THEME_PLOTLY will be updated on rerun after session_state changes
THEME_PLOTLY = light_theme_plotly if st.session_state.current_theme == 'Light' else dark_theme_plotly

theme_variables_css_light = """<style>:root { --primary: #1a3e72; --secondary: #e9a229; --accent: #5f8fd9; --text-light: #ffffff; --text-dark: #333333; --text-neutral: #7f8c8d; --background-main: #f5f7fa; --background-card: #ffffff; --background-sidebar: #1a3e72; --border-color: #dddddd; --shadow-color: rgba(0,0,0,0.1); --search-highlight: #FFF59D; --data-quality-high: #4CAF50; --data-quality-medium: #FFC107; --data-quality-low: #F44336;}</style>"""
theme_variables_css_dark = """<style>:root { --primary: #76a2dd; --secondary: #f7ca4f; --accent: #e9a229; --text-light: #1f2c38; --text-dark: #ecf0f1; --text-neutral: #bdc3c7; --background-main: #222b38; --background-card: #2d394a; --background-sidebar: #1f2c38; --border-color: #4a6378; --shadow-color: rgba(0,0,0,0.3); --search-highlight: #FFD54F; --data-quality-high: #81C784; --data-quality-medium: #FFD54F; --data-quality-low: #E57373;}</style>"""
st.markdown(theme_variables_css_light if st.session_state.current_theme == 'Light' else theme_variables_css_dark, unsafe_allow_html=True)
MAIN_STRUCTURAL_CSS = """<style>body { color: var(--text-dark); background-color: var(--background-main); } .main { background-color: var(--background-main); padding: 1rem; } .stPlotlyChart { border-radius: 8px; box-shadow: 0 4px 8px var(--shadow-color); margin-bottom: 1rem; } .metric-card { padding: 20px; border-radius: 10px; background: var(--background-card); box-shadow: 0 4px 8px var(--shadow-color); height: 100%; display: flex; flex-direction: column; justify-content: space-around; border-top: 5px solid var(--secondary); color: var(--text-dark); } .metric-card:hover { transform: translateY(-5px); box-shadow: 0 6px 12px var(--shadow-color); } .metric-card h3 { color: var(--primary); font-size: 1.1em; margin-bottom: 8px; } .metric-card p { font-size: 28px; font-weight: bold; color: var(--accent); margin-top: 5px; } .sidebar .sidebar-content { background-color: var(--background-sidebar); color: var(--text-light); } .sidebar .st-emotion-cache-vk34n { color: var(--text-light); } .sidebar h2, .sidebar h3 { color: var(--secondary) !important; margin-bottom: 15px;} .sidebar .stRadio > label { color: var(--text-light); } .highlight { background-color: var(--search-highlight); padding: 0.1em 0.2em; border-radius: 0.2em; font-weight: bold; } .dataframe { width: 100%; border-collapse: collapse; border-spacing: 0; border-radius: 8px; overflow: hidden; } .dataframe th { background-color: var(--primary); color: var(--text-light); padding: 12px 8px; text-align: left; border-bottom: 1px solid var(--border-color); } .dataframe td { padding: 8px; border-bottom: 1px solid var(--border-color); color: var(--text-dark); } .dataframe tr:nth-child(even) { background-color: var(--background-card); } .dataframe tr:nth-child(odd) { background-color: var(--background-main); } .dataframe tbody tr:hover { background-color: rgba(233, 162, 41, 0.2) !important; } .stDataFrame { border-radius: 8px; box-shadow: 0 2px 4px var(--shadow-color); } .data-quality-high { color: var(--data-quality-high); font-weight: bold; } .data-quality-medium { color: var(--data-quality-medium); font-weight: bold; } .data-quality-low { color: var(--data-quality-low); font-weight: bold; } .map-legend-item { display: flex; align-items: center; margin-right: 15px; font-size: 0.9em; color: var(--text-dark); } .map-legend-color-box { width: 15px; height: 15px; margin-right: 5px; border-radius: 3px; border: 1px solid var(--border-color); } .map-legend-icon { margin-right: 5px; }</style>"""
st.markdown(MAIN_STRUCTURAL_CSS, unsafe_allow_html=True)

# ========== DATA LOADING & CLEANING FUNCTIONS ==========
@st.cache_data
def load_data(island_name):
    market_config = MARKET_DATA_SOURCES.get(island_name)
    if not market_config:
        st.error(f"Error: Configuration for market '{island_name}' not found.")
        return pd.DataFrame()

    excel_file_path = market_config['file']
    header_row = market_config['header_row']

    abs_file_path = "Could not determine absolute path"
    file_exists = False
    try:
        abs_file_path = os.path.abspath(excel_file_path)
        file_exists = os.path.exists(abs_file_path)
    except Exception:
        pass

    if not file_exists:
        try:
            file_exists = os.path.exists(excel_file_path)
        except Exception:
            pass

    if not file_exists:
        st.error(f"Error: Data file '{excel_file_path}' (resolves to '{abs_file_path}') not found for {island_name} market. Ensure the Excel file is in the correct location and the filename matches exactly (including case).")
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

    df.columns = [str(col).strip() for col in df.columns]

    required_initial_cols = ['Price', 'Property Type', 'Type', 'Description', 'Parish', 'Category', 'Name', 'Size']
    for col in required_initial_cols:
        if col not in df.columns:
            df[col] = None

    df['Price'] = pd.to_numeric(df['Price'].astype(str).str.replace('US$', '', regex=False).str.replace(',', '', regex=False), errors='coerce').fillna(0)

    if 'Size' in df.columns:
        df['Size'] = df['Size'].astype(str)
        df['Size_SqFt'] = df['Size'].apply(lambda x: float(re.sub(r'[^\d.]', '', x.split('Sq. Ft.')[0].split('Sq Ft')[0].split('sq. ft.')[0].split('sq ft')[0])) if pd.notna(x) and any(unit in x for unit in ['Sq. Ft.','Sq Ft','sq. ft.','sq ft']) else np.nan)
        df['Size_Acres'] = df['Size'].apply(lambda x: float(re.sub(r'[^\d.]', '', x.split('Acres')[0].split('Acre')[0].split('acres')[0].split('acre')[0])) if pd.notna(x) and any(unit in x for unit in ['Acres','Acre','acres','acre']) else np.nan)
        df['Size_SqFt'] = df['Size_SqFt'].fillna(df['Size_Acres'] * 43560)
        df = df.drop(columns=['Size_Acres', 'Size'], errors='ignore')
    else:
        df['Size_SqFt'] = 0
    df['Size_SqFt'] = pd.to_numeric(df['Size_SqFt'], errors='coerce').fillna(0)


    for col_name in ['Property Type', 'Type', 'Category', 'Name', 'Description']:
        default_val = 'Unknown' if col_name != 'Name' else 'Property'
        if col_name not in df.columns: df[col_name] = default_val
        df[col_name] = df[col_name].astype(str).fillna(default_val).replace({'nan': default_val, 'N/A': default_val, '': default_val})
        if col_name == 'Type': df[col_name] = df[col_name].str.title()
        elif col_name == 'Name': df[col_name] = df[col_name].str.strip()

    df['Bedrooms'] = pd.to_numeric(df['Description'].astype(str).str.extract(r'(\d+)\s*Bed(?:room)?s?', flags=re.IGNORECASE)[0], errors='coerce').fillna(0).astype(int)
    df['Bathrooms'] = pd.to_numeric(df['Description'].astype(str).str.extract(r'(\d+)\s*Bath(?:room)?s?', flags=re.IGNORECASE)[0], errors='coerce').fillna(0).astype(int)

    if 'Parish' in df.columns and island_name == 'Barbados':
        df['Parish'] = df['Parish'].astype(str).str.strip()
        canonical_parish_map_config = market_config.get('parish_coords', {})
        valid_parish_names_from_config = list(canonical_parish_map_config.keys())

        parish_variation_map = {
            'james': 'St. James', 'st james': 'St. James', 'st.james': 'St. James', 'saint james': 'St. James',
            'christchurch': 'Christ Church', '165christ church': 'Christ Church', 'church': 'Christ Church',
            'peter': 'St. Peter', 'st peter': 'St. Peter', 'st.peter': 'St. Peter', 'saint peter': 'St. Peter',
            'philip': 'St. Philip', 'st philip': 'St. Philip', 'st.philip': 'St. Philip', 'saint philip': 'St. Philip',
            'lucy': 'St. Lucy', 'st lucy': 'St. Lucy', 'st.lucy': 'St. Lucy', 'saint lucy': 'St. Lucy',
            'michael': 'St. Michael', 'st michael': 'St. Michael', 'st.michael': 'St. Michael', 'saint michael': 'St. Michael',
            'andrew': 'St. Andrew', 'st andrew': 'St. Andrew', 'st.andrew': 'St. Andrew', 'saint andrew': 'St. Andrew',
            'george': 'St. George', 'st george': 'St. George', 'st.george': 'St. George', 'saint george': 'St. George',
            'joseph': 'St. Joseph', 'st joseph': 'St. Joseph', 'st.joseph': 'St. Joseph', 'saint joseph': 'St. Joseph',
            'thomas': 'St. Thomas', 'st thomas': 'St. Thomas', 'st.thomas': 'St. Thomas', 'saint thomas': 'St. Thomas',
        }
        for valid_name in valid_parish_names_from_config:
            if valid_name != 'Unknown':
                parish_variation_map[valid_name.lower()] = valid_name

        df['Parish_lower'] = df['Parish'].str.lower()
        df['Parish'] = df['Parish_lower'].map(parish_variation_map)

        if valid_parish_names_from_config:
            df.loc[~df['Parish'].isin(valid_parish_names_from_config), 'Parish'] = 'Unknown'
        else:
            df['Parish'] = 'Unknown'
        df['Parish'] = df['Parish'].fillna('Unknown')
        if 'Parish_lower' in df.columns:
            df = df.drop(columns=['Parish_lower'])

    elif 'Parish' in df.columns:
        df['Parish'] = df['Parish'].astype(str).str.strip().fillna('Unknown').replace({'': 'Unknown', 'nan': 'Unknown'})
    else:
        df['Parish'] = 'Unknown'

    df['Country'] = island_name
    df['Data Quality Score'] = df.apply(calculate_data_quality_score, axis=1)
    return df

def calculate_data_quality_score(row):
    score = 100
    if row.get('Price', 0) == 0: score -= 30
    if row.get('Property Type', 'Unknown') == 'Unknown': score -= 20
    if row.get('Parish', 'Unknown') == 'Unknown': score -= 15
    if str(row.get('Description', '')).strip() == '': score -= 15
    if row.get('Bedrooms', 0) == 0: score -= 10
    if row.get('Bathrooms', 0) == 0: score -= 10
    if pd.isna(row.get('Size_SqFt')) or row.get('Size_SqFt', 0) == 0: score -= 10
    if row.get('Category', 'Unknown') == 'Unknown': score -= 5
    if row.get('Type', 'Unknown') == 'Unknown': score -= 5
    return max(0, min(100, score))

def get_data_quality_class(score):
    if score >= 80: return "data-quality-high"
    elif score >= 50: return "data-quality-medium"
    else: return "data-quality-low"

@st.cache_data(ttl=900)
def get_live_weather(api_key, city_name="Bridgetown", country_code="BB"):
    if not api_key: return None
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"q": f"{city_name},{country_code}", "appid": api_key, "units": "metric"}
    try:
        response = requests.get(base_url, params=params, timeout=10) # Increased timeout
        response.raise_for_status()
        data = response.json()
        return {"temp": data["main"]["temp"], "feels_like": data["main"]["feels_like"], "description": data["weather"][0]["description"].title(), "icon_url": f"http://openweathermap.org/img/wn/{data['weather'][0]['icon']}@2x.png", "humidity": data["main"]["humidity"], "wind_speed": data["wind"]["speed"], "city": data["name"]}
    except requests.exceptions.RequestException as e: # Catch specific exceptions
        st.sidebar.caption(f"Weather data error: {e}")
        return None
    except Exception: # Catch other potential errors like JSONDecodeError
        st.sidebar.caption("Weather data unavailable.")
        return None


def display_weather_sidebar(weather_api_key):
    if not weather_api_key: st.sidebar.caption("Weather API key not configured."); return
    weather_data = get_live_weather(weather_api_key)
    if weather_data:
        st.sidebar.subheader(f"Weather in {weather_data['city']}")
        c1,c2=st.sidebar.columns([.4,.6],gap="small"); c1.image(weather_data['icon_url'],width=60); c2.metric("Temp",f"{weather_data['temp']:.0f}°C")
        st.sidebar.caption(f"{weather_data['description']}\nFeels like: {weather_data['feels_like']:.0f}°C\nHumidity: {weather_data['humidity']}% | Wind: {weather_data['wind_speed']:.1f} m/s")
    else: st.sidebar.caption("Live weather data currently unavailable.")

def initialize_ai():
    if not OPENAI_API_KEY: return False
    try: openai.api_key = OPENAI_API_KEY; return True
    except Exception as e: st.error(f"OpenAI API init failed: {e}. AI features disabled."); return False

def generate_ai_insights(filtered_df, full_df, market_name):
    if not OPENAI_API_KEY: return "AI features disabled (API key not configured)."
    if len(filtered_df) < 5: return "Not enough data for AI insights. Please select at least 5 properties from the filters."
    
    summary_df = filtered_df.copy()
    for col in ['Property Type', 'Parish', 'Category', 'Price', 'Size_SqFt']:
        if col not in summary_df.columns:
            summary_df[col] = np.nan

    summary = {
        "market": market_name,
        "filtered_count": len(summary_df),
        "market_total_count": len(full_df),
        "price_USD": {},
        "size_sqft": {},
        "property_types_distribution": summary_df['Property Type'].value_counts().to_dict(),
        "parish_distribution": summary_df['Parish'].value_counts().to_dict(),
        "category_distribution": summary_df['Category'].value_counts().to_dict()
    }

    if pd.api.types.is_numeric_dtype(summary_df['Price']) and not summary_df['Price'].dropna().empty:
        prices = summary_df['Price'][summary_df['Price'] > 0].dropna()
        if not prices.empty:
            summary["price_USD"] = {"min": prices.min(), "max": prices.max(), "median": prices.median(), "mean": prices.mean()}

    if pd.api.types.is_numeric_dtype(summary_df['Size_SqFt']) and not summary_df['Size_SqFt'].dropna().empty:
        sizes = summary_df['Size_SqFt'][summary_df['Size_SqFt'] > 0].dropna()
        if not sizes.empty:
            summary["size_sqft"] = {"min": sizes.min(), "max": sizes.max(), "median": sizes.median(), "mean": sizes.mean()}

    prompt = f"""Analyze the real estate data for {market_name}. Prices are in USD, and sizes are in Sq.Ft.
Based on the following summary of filtered data:
{summary}

Please provide 3-5 key insights covering:
- Price trends (e.g., average, median, range for the selection).
- Dominant property types and sizes in the current selection.
- Geographical observations (e.g., popular parishes in the selection).
- Comparison of the filtered selection to the overall market, if discernible from counts.
- Any other notable patterns or characteristics.

Format your response using Markdown:
- Use bullet points for each key insight.
- Make the insights concise and actionable for potential investors or buyers.
- If specific numbers are relevant (e.g., average price), include them.
Example of a good insight:
* **Price Point:** The average price for selected properties is $XXX,XXX, with a median of $YYY,YYY.

Begin the insights directly.
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=450 # Increased slightly for potentially longer markdown
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI insight generation failed: {str(e)}"

def apply_search_filter(df, search_term):
    if not search_term or df.empty: return df
    term = search_term.lower()
    cols_to_search = [c for c in ['Name','Description','Parish','Property Type','Category','Type'] if c in df.columns]
    if not cols_to_search: return df
    return df[df[cols_to_search].astype(str).fillna('').apply(lambda r: r.str.lower().str.contains(term, na=False)).any(axis=1)]

def natural_language_query(query, df):
    if not OPENAI_API_KEY: return apply_search_filter(df, query)
    try:
        df_query = df.copy()
        text_cols_for_prompt = ['Name','Description','Parish','Property Type','Category','Type']
        for col in text_cols_for_prompt:
            if col in df_query.columns:
                df_query[col] = df_query[col].astype(str)

        prompt = f"User query: \"{query}\". DF cols: {list(df_query.columns)}. Text cols: {text_cols_for_prompt}. Python pandas code to *further filter* `df`. Case-insensitive `.str.contains(..., case=False, na=False)`. Combine conditions with `&` or `|`. ONLY Python code, like `df[...]`. Examples: 'Christ Church' -> df[df['Parish'].str.contains('Christ Church',case=False,na=False)]; 'villas under 1M' -> df[(df['Category'].str.contains('For Sale',case=False,na=False))&(df['Property Type'].str.contains('villa',case=False,na=False))&(df['Price'] < 1000000)]. Code:"
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}], temperature=0.2, max_tokens=250)
        code = response.choices[0].message.content.strip()

        if code.startswith('df[') and code.endswith(']') and 'df' in code:
            res_df = eval(code, {'df': df_query, 'pd': pd, 'np': np, 're': re})
            if isinstance(res_df, pd.DataFrame) and len(res_df) <= len(df_query):
                return res_df
            else:
                st.warning("AI query generated an invalid filter. Using text search.")
        else:
            st.warning("AI query did not generate a valid filter. Using text search.")

    except Exception as e:
        st.error(f"AI query failed: {e}. Using text search.")
    return apply_search_filter(df, query)

def geocode_properties(df, island_name):
    if df.empty: return df
    config = MARKET_DATA_SOURCES.get(island_name, {})
    def_lat, def_lon = config.get('default_coords', (13.1939, -59.5432))
    if 'Parish' not in df.columns:
        df['lat'] = def_lat
        df['lon'] = def_lon
        return df

    parish_coords_map = config.get('parish_coords', {})
    if 'Unknown' not in parish_coords_map:
        parish_coords_map['Unknown'] = (def_lat, def_lon)

    df['map_coords'] = df['Parish'].apply(lambda p: parish_coords_map.get(p, parish_coords_map['Unknown']))
    df['lat'], df['lon'] = zip(*df['map_coords'])
    df = df.drop(columns=['map_coords'], errors='ignore')

    mask = df['Parish'].isin(parish_coords_map.keys()) & \
           ~((df['lat'] == def_lat) & (df['lon'] == def_lon) & (df['Parish'] == 'Unknown')) & \
           ~((df['lat'] == parish_coords_map.get('Unknown', (None, None))[0]) & (df['lon'] == parish_coords_map.get('Unknown', (None, None))[1]))

    if mask.any():
        indices = df.index[mask]
        if not indices.empty:
            df.loc[indices,'lat'] += np.random.uniform(-0.005,0.005,size=len(indices))
            df.loc[indices,'lon'] += np.random.uniform(-0.005,0.005,size=len(indices))
    return df

def create_advanced_map(filtered_df, island_name, amenities_to_show=None, show_school_districts=False):
    config = MARKET_DATA_SOURCES.get(island_name)
    if not config: return None
    def_lat, def_lon = config.get('default_coords', (13.1939, -59.5432))
    center_lat, center_lon = def_lat, def_lon
    zoom = 11

    if not filtered_df.empty and 'lat' in filtered_df.columns and 'lon' in filtered_df.columns:
        valid_coords = filtered_df[['lat','lon']].dropna().astype(float)
        valid_coords_for_center = valid_coords[~((valid_coords['lat'] == def_lat) & (valid_coords['lon'] == def_lon))]
        if not valid_coords_for_center.empty:
            center_lat, center_lon = valid_coords_for_center['lat'].mean(), valid_coords_for_center['lon'].mean()
            zoom = 13

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles='cartodbpositron' if st.session_state.current_theme=='Light' else 'cartodbdark_matter', control_scale=True)

    if not filtered_df.empty and 'lat' in filtered_df.columns and 'lon' in filtered_df.columns:
        props_grp = folium.FeatureGroup(name=f'Properties ({island_name}) - {len(filtered_df)} shown').add_to(m)
        map_df = filtered_df.dropna(subset=['lat','lon']).copy()
        str_cols = ['Name','Property Type','Parish','Category','Type']
        num_cols = ['Price','Bedrooms','Bathrooms','Data Quality Score','Size_SqFt']

        for c in str_cols: map_df[c] = map_df[c].astype(str).fillna('N/A') if c in map_df.columns else 'N/A'
        for c in num_cols: map_df[c] = pd.to_numeric(map_df[c], errors='coerce').fillna(0) if c in map_df.columns else 0

        for _, r in map_df.iterrows():
            category_val = str(r.get('Category','Unknown')).upper()
            
            if category_val == 'FOR SALE':
                color = 'red'
            elif category_val == 'FOR RENT':
                color = 'green'
            elif category_val == 'SOLD':
                color = 'green'
            elif category_val == 'LEASED':
                color = 'purple'
            else:
                color = 'gray'      

            prop_type_val = str(r.get('Property Type','')).lower()
            icon_name = 'home'
            if 'land' in prop_type_val: icon_name = 'tree-conifer'
            elif 'commercial' in prop_type_val or 'office' in prop_type_val: icon_name = 'building'
            elif 'apartment' in prop_type_val or 'condo' in prop_type_val: icon_name = 'th-large'

            popup_html = f"""
            <b>{r.get('Name', 'N/A')}</b><br>
            Type: {r.get('Property Type', 'N/A')}<br>
            Parish: {r.get('Parish', 'N/A')}<br>
            Status: {r.get('Category', 'N/A')}<br>
            Price: {format_currency(r.get('Price',0))}<br>
            Beds: {int(r.get('Bedrooms',0))} | Baths: {int(r.get('Bathrooms',0))}<br>
            Size: {r.get('Size_SqFt',0):,.0f} Sq.Ft.<br>
            <i>DQ: {r.get('Data Quality Score',0):.0f}/100</i>
            """
            folium.Marker(
                location=[r['lat'],r['lon']],
                popup=folium.Popup(popup_html,max_width=300),
                icon=folium.Icon(color=color, icon=icon_name, prefix='fa' if icon_name in ['home', 'building'] else 'glyphicon')
            ).add_to(props_grp)

    market_amenities = config.get('amenities',{})
    if amenities_to_show and market_amenities:
        for amenity_type, locations in market_amenities.items():
            if amenity_type in amenities_to_show and locations:
                default_color, default_icon = ('gray', 'info-sign')
                amenity_style = {
                    'Beach':('blue','tint'),
                    'School':('orange','education'),
                    'Restaurant':('red','cutlery')
                }.get(amenity_type,(default_color, default_icon))

                amenity_grp = folium.FeatureGroup(name=f"{amenity_type}s").add_to(m)
                for loc_info in locations:
                    if all(k in loc_info for k in ['lat','lon','name']) and pd.notna(loc_info['lat']) and pd.notna(loc_info['lon']):
                        try:
                            folium.Marker(
                                location=[float(loc_info['lat']),float(loc_info['lon'])],
                                popup=f"{amenity_type}: {loc_info.get('name','N/A')}",
                                icon=folium.Icon(color=amenity_style[0],icon=amenity_style[1], prefix='glyphicon' if amenity_style[1] not in ['cutlery'] else 'fa')
                            ).add_to(amenity_grp)
                        except ValueError: pass

    if show_school_districts and 'School' in market_amenities and market_amenities['School']:
        school_zone_grp = folium.FeatureGroup(name='School Zones (1km Radius)').add_to(m)
        for school_info in market_amenities['School']:
            if all(k in school_info for k in ['lat','lon','name']) and pd.notna(school_info['lat']) and pd.notna(school_info['lon']):
                try:
                    folium.Circle(
                        location=[float(school_info['lat']),float(school_info['lon'])],
                        radius=1000,
                        color='orange',
                        fill=True,
                        fill_opacity=0.2,
                        tooltip=f"1km Zone: {school_info.get('name','N/A')}",
                        popup=f"School Zone: {school_info.get('name','N/A')}"
                    ).add_to(school_zone_grp)
                except ValueError: pass
    folium.LayerControl().add_to(m); return m


def show_data_quality_report(df):
    if df.empty: st.caption("No Data Quality report to display for the current selection."); return
    st.subheader("🔍 Data Quality Report")
    with st.expander("How is the Data Quality Score calculated?"):
        st.markdown("""
            The Data Quality Score (0-100) measures data completeness for each property. A higher score indicates more complete data.
            Starting at 100, points are deducted for missing or default ('Unknown', 0) key fields:
            <ul>
                <li><b>Price is $0 or missing:</b> -30 points</li>
                <li><b>Property Type is 'Unknown':</b> -20 points</li>
                <li><b>Parish is 'Unknown':</b> -15 points</li>
                <li><b>Description is empty:</b> -15 points</li>
                <li><b>Bedrooms count is 0:</b> -10 points</li>
                <li><b>Bathrooms count is 0:</b> -10 points</li>
                <li><b>Size (Sq.Ft.) is 0 or missing:</b> -10 points</li>
                <li><b>Category (e.g., For Sale) is 'Unknown':</b> -5 points</li>
                <li><b>Type (e.g., Residential) is 'Unknown':</b> -5 points</li>
            </ul>
        """, unsafe_allow_html=True)

    if 'Data Quality Score' in df.columns and not df['Data Quality Score'].empty:
        scores = pd.to_numeric(df['Data Quality Score'],errors='coerce').dropna()
        if not scores.empty:
            overall_mean_score = scores.mean()
            st.markdown(f"""<div style="background:var(--background-card);padding:15px;border-radius:8px;margin-bottom:15px;"><h3 style="margin-top:0;color:var(--primary);">Overall Average DQ Score</h3><p style="font-size:22px;margin-bottom:0;"><span class="{get_data_quality_class(overall_mean_score)}">{overall_mean_score:.1f}/100</span> (for {len(df):,} filtered properties)</p></div>""", unsafe_allow_html=True)
    else:
        st.info("Data Quality Score column not available or no data in current filter.")

    st.markdown("**Column Completeness (Percentage of non-missing/non-default values):**")
    completion_data, total_rows = [], len(df)
    cols_to_check_dq = {
        'Price':lambda data_frame:(pd.to_numeric(data_frame.get('Price', pd.Series(dtype=float)),errors='coerce').fillna(0)==0).sum(),
        'Property Type':lambda data_frame:(data_frame.get('Property Type', pd.Series(dtype=str)).astype(str).fillna('Unknown')=='Unknown').sum(),
        'Parish':lambda data_frame:(data_frame.get('Parish', pd.Series(dtype=str)).astype(str).fillna('Unknown')=='Unknown').sum(),
        'Bedrooms':lambda data_frame:(pd.to_numeric(data_frame.get('Bedrooms', pd.Series(dtype=float)),errors='coerce').fillna(0)==0).sum(),
        'Bathrooms':lambda data_frame:(pd.to_numeric(data_frame.get('Bathrooms', pd.Series(dtype=float)),errors='coerce').fillna(0)==0).sum(),
        'Size_SqFt':lambda data_frame:(pd.to_numeric(data_frame.get('Size_SqFt', pd.Series(dtype=float)),errors='coerce').fillna(0)==0).sum(),
        'Description':lambda data_frame:(data_frame.get('Description', pd.Series(dtype=str)).astype(str).str.strip().fillna('')=='').sum(),
        'Category':lambda data_frame:(data_frame.get('Category', pd.Series(dtype=str)).astype(str).fillna('Unknown')=='Unknown').sum(),
        'Type':lambda data_frame:(data_frame.get('Type', pd.Series(dtype=str)).astype(str).fillna('Unknown')=='Unknown').sum()
    }
    if total_rows > 0:
        for col_name, missing_func in cols_to_check_dq.items():
            if col_name in df.columns:
                try:
                    num_missing = missing_func(df)
                    completion_data.append({'Column':col_name,'Missing or Default Values':num_missing,'Total Rows':total_rows})
                except Exception as e:
                    st.caption(f"Could not process DQ for column {col_name}: {e}")
            else:
                completion_data.append({'Column':col_name,'Missing or Default Values':total_rows,'Total Rows':total_rows})

    if completion_data:
        completion_df = pd.DataFrame(completion_data)
        completion_df['% Complete'] = 100 * (1 - completion_df['Missing or Default Values'] / completion_df['Total Rows'])
        completion_df['% Complete'] = completion_df['% Complete'].clip(lower=0)

        num_cols_for_display = min(len(completion_df), 4 if len(completion_df) > 1 else 1)
        metric_cols = st.columns(num_cols_for_display)
        for i, row_data in completion_df.iterrows():
            with metric_cols[i % num_cols_for_display]:
                st.markdown(f"""<div style="background:var(--background-card);padding:10px;border-radius:5px;margin-bottom:8px;">
                                    <p style="margin:0 0 3px 0;font-weight:bold;font-size:0.9em;color:var(--primary);">{row_data['Column']}</p>
                                    <p style="margin:0;font-size:16px;"><span class="{get_data_quality_class(row_data['% Complete'])}">{row_data['% Complete']:.1f}% Complete</span></p>
                                    <p style="margin:3px 0 0 0;font-size:11px;color:var(--text-neutral);">{row_data['Missing or Default Values']} missing/default</p>
                                 </div>""", unsafe_allow_html=True)
    else:
        st.caption("No columns available for Data Quality completeness check.")

    st.markdown("**Distribution of Data Quality Scores:**")
    if 'Data Quality Score' in df.columns and not df['Data Quality Score'].empty:
        dq_scores_numeric = pd.to_numeric(df['Data Quality Score'],errors='coerce').fillna(0)
        if not dq_scores_numeric.empty:
            fig_dq_hist = px.histogram(x=dq_scores_numeric,nbins=20,range_x=[0,100],title='Property Data Quality Scores',color_discrete_sequence=[THEME_PLOTLY['primary']])
            fig_dq_hist.update_layout(xaxis_title='Data Quality Score (0-100)',yaxis_title='Number of Properties',paper_bgcolor=THEME_PLOTLY["paper_bgcolor"],plot_bgcolor=THEME_PLOTLY["plot_bgcolor"],font_color=THEME_PLOTLY["font_color"],yaxis_gridcolor=THEME_PLOTLY["grid_color"],xaxis_gridcolor=THEME_PLOTLY["grid_color"])
            st.plotly_chart(fig_dq_hist,use_container_width=True)
        else:
            st.caption("No valid Data Quality Scores to display distribution.")
    else:
        st.caption("Data Quality Score column not available for distribution chart.")


def create_transaction_pie_charts(df, theme):
    if df.empty or 'Category' not in df.columns: st.caption("No data available for transaction pie chart."); return
    counts = df['Category'].value_counts()
    if counts.empty or counts.sum()==0: st.caption("No transaction counts to display in pie chart."); return

    valid_categories = counts.index.astype(str).str.upper()
    color_map = {cat: theme["category_colors"].get(cat, theme["neutral_grey"]) for cat in valid_categories}

    fig = px.pie(counts, names=counts.index, values=counts.values,
                 color=counts.index,
                 color_discrete_map={idx: color_map[str(idx).upper()] for idx in counts.index},
                 hole=0.5, title="Overall Transaction Types")
    fig.update_traces(textinfo='percent+label', marker=dict(line=dict(color=theme.get("paper_bgcolor", "#FFFFFF"), width=2)))
    fig.update_layout(title_x=0.5, paper_bgcolor=theme.get("paper_bgcolor"), plot_bgcolor=theme.get("plot_bgcolor"), font_color=theme.get("font_color"),
                      legend_title_text='Category', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig,use_container_width=True)

def create_property_type_bar_chart(df, title, theme, offset=0):
    if df.empty or 'Property Type' not in df.columns: st.caption(f"No data available for {title} bar chart."); return
    counts = df['Property Type'].value_counts()
    if counts.empty or counts.sum()==0: st.caption(f"No '{df['Property Type'].name if 'Property Type' in df.columns else 'Property Type'}' counts for {title}."); return

    bar_palette = theme.get("bar_palette", px.colors.qualitative.Plotly)
    if offset > 0 and len(bar_palette) > offset:
        bar_palette = bar_palette[offset:] + bar_palette[:offset]
    if len(counts) > len(bar_palette):
        bar_palette = bar_palette * (len(counts) // len(bar_palette) + 1)

    fig = px.bar(x=counts.index, y=counts.values, color=counts.index,
                 color_discrete_sequence=bar_palette[:len(counts)],
                 labels={'y':'Number of Properties','x':'Property Type'}, title=title)
    fig.update_layout(showlegend=False, title_x=0.5, xaxis_title=None, yaxis_title="Number of Properties",
                      paper_bgcolor=theme.get("paper_bgcolor"), plot_bgcolor=theme.get("plot_bgcolor"),
                      font_color=theme.get("font_color"),
                      yaxis_gridcolor=theme.get("grid_color"), xaxis_gridcolor=theme.get("grid_color"))
    st.plotly_chart(fig,use_container_width=True)

def format_currency(value):
    if pd.isna(value): return "N/A"
    try:
        numeric_value = float(value)
        if numeric_value == 0: return "$0"
        return f"${numeric_value:,.0f}"
    except (ValueError, TypeError):
        return "N/A"


def safe_isin_filter(df, col_name, selected_values):
    if col_name not in df.columns or not selected_values: return pd.Series(True, index=df.index)
    if df[col_name].dtype == 'object' or pd.api.types.is_string_dtype(df[col_name]):
        selected_values_str = [str(v) for v in selected_values]
        return df[col_name].astype(str).isin(selected_values_str)
    return df[col_name].isin(selected_values)


def main():
    ai_enabled = initialize_ai()
    with st.sidebar:
        st.image("https://s3.us-east-2.amazonaws.com/terracaribbean.com/wp-content/uploads/2025/04/08080016/site-logo.png",width=200)
        if OPENWEATHERMAP_API_KEY: display_weather_sidebar(OPENWEATHERMAP_API_KEY); st.sidebar.markdown("---")
        st.title("Filters & Tools")
        markets_list = list(MARKET_DATA_SOURCES.keys())
        selected_market = st.selectbox("Select Market",options=markets_list,index=markets_list.index('Barbados') if 'Barbados' in markets_list else 0)

        sel_theme = st.radio("Theme",['Light','Dark'],index=0 if st.session_state.current_theme=='Light' else 1)
        if sel_theme != st.session_state.current_theme:
            st.session_state.current_theme = sel_theme
            st.rerun()

        df_full = load_data(selected_market)
        if df_full.empty: st.error(f"Stopping: No data loaded for {selected_market}. Check file and logs."); return

        st.subheader("AI Search")
        nl_query = ""
        if ai_enabled:
            nl_query = st.text_input(f"🔍 Ask about {selected_market} properties",placeholder="e.g., 'St. James over 100,000 Sq Ft'",help="Prices in USD. AI can filter by keywords, price, property type, etc.")
        else:
            st.info("AI Search disabled (OpenAI API key not configured or invalid).")


        st.subheader("Standard Filters")
        def get_unique_opts(df, col): return sorted(df[col].astype(str).unique()) if col in df.columns and not df[col].dropna().empty else []

        prop_type_opts = get_unique_opts(df_full,'Property Type')
        sel_prop_type = st.multiselect('Property Type',options=prop_type_opts,default=prop_type_opts)

        cat_opts = get_unique_opts(df_full,'Category')
        sel_category = st.multiselect('Transaction Type (Category)',options=cat_opts,default=cat_opts)

        par_opts = get_unique_opts(df_full,'Parish')
        def_parishes = [p for p in par_opts if p!='Unknown'] if 'Unknown' in par_opts and len(par_opts)>1 else par_opts
        sel_parish = st.multiselect('Parish',options=par_opts,default=def_parishes)

        p_min_data, p_max_data = 0.0, 1000000.0
        if 'Price' in df_full.columns:
            valid_prices = pd.to_numeric(df_full['Price'], errors='coerce').dropna()
            if not valid_prices.empty:
                p_min_data = float(valid_prices.min())
                p_max_data = float(valid_prices.max())
                if p_min_data == p_max_data:
                    p_max_data = p_min_data + max(10000, p_min_data * 0.1)
                elif p_max_data > 100000 :
                    p_max_data *=1.05
        
        p_step = 1.0
        if (p_max_data - p_min_data) > 0 :
             p_step = max(1.0, (p_max_data - p_min_data) / 200)
        
        if p_min_data > p_max_data: p_min_data = p_max_data

        sel_price_range = st.slider('Price Range (USD)',min_value=p_min_data,max_value=p_max_data,value=(p_min_data,p_max_data),step=p_step,format="$%.0f")

        st.subheader("Map Features")
        current_market_config = MARKET_DATA_SOURCES.get(selected_market,{})
        amenities_config_map = current_market_config.get('amenities',{})
        amen_keys_map = list(amenities_config_map.keys())
        sel_amenities = st.multiselect("Show Amenities on Map",options=amen_keys_map,default=[a for a in ['Beach','School','Restaurant'] if a in amen_keys_map])
        sel_school_zones = st.checkbox("Show School Zones (1km)",value=False) if 'School' in amen_keys_map else False

        st.subheader("Data Quality Filter")
        min_dq_score = 0
        if 'Data Quality Score' in df_full.columns:
            min_dq_score = st.slider("Minimum Data Quality Score",0,100,10,5)

    df_geocoded = geocode_properties(df_full.copy(),selected_market)

    conditions = pd.Series(True, index=df_geocoded.index)

    if sel_prop_type: conditions &= safe_isin_filter(df_geocoded, 'Property Type', sel_prop_type)
    if sel_category: conditions &= safe_isin_filter(df_geocoded, 'Category', sel_category)
    if sel_parish: conditions &= safe_isin_filter(df_geocoded, 'Parish', sel_parish)

    if 'Price' in df_geocoded.columns:
        prices_numeric_filter = pd.to_numeric(df_geocoded['Price'], errors='coerce').fillna(0)
        conditions &= (prices_numeric_filter >= sel_price_range[0]) & (prices_numeric_filter <= sel_price_range[1])

    if 'Data Quality Score' in df_geocoded.columns:
        quality_scores_filter = pd.to_numeric(df_geocoded['Data Quality Score'], errors='coerce').fillna(0)
        conditions &= (quality_scores_filter >= min_dq_score)

    filtered_df = df_geocoded[conditions].copy()

    if nl_query:
        temp_df_for_nl_search = filtered_df.copy()
        if ai_enabled:
            res_nl = natural_language_query(nl_query, temp_df_for_nl_search)
            if res_nl is not None and not res_nl.empty and len(res_nl) < len(temp_df_for_nl_search) :
                filtered_df = res_nl
                st.success(f"AI refined selection to {len(filtered_df)} properties.")
            elif res_nl is not None and len(res_nl) == len(temp_df_for_nl_search) and len(temp_df_for_nl_search) > 0:
                st.info("AI query did not further refine the current selection.")
            elif res_nl is not None and res_nl.empty:
                st.warning(f"AI query resulted in 0 properties. Showing results before AI query ({len(temp_df_for_nl_search)}).")
        else:
            res_search = apply_search_filter(temp_df_for_nl_search, nl_query)
            if res_search is not None and not res_search.empty and len(res_search) < len(temp_df_for_nl_search) :
                filtered_df = res_search
                st.info(f"Search refined selection to {len(filtered_df)} properties.")
            elif res_search is not None and len(res_search) == len(temp_df_for_nl_search) and len(temp_df_for_nl_search) > 0 :
                st.info("Search term did not further refine the current selection.")
            elif res_search is not None and res_search.empty:
                st.warning(f"Search term resulted in 0 properties. Showing results before search ({len(temp_df_for_nl_search)}).")


    st.title(f"🏝️ Terra Caribbean Property Intelligence")
    st.markdown(f"""<span style="color:var(--text-neutral);font-size:1.1em;">Insights | <b>{selected_market}</b> | <b>{len(filtered_df):,}</b> properties analyzed | Prices USD</span>""",unsafe_allow_html=True)

    if ai_enabled:
        with st.expander(f"💡 AI-Powered Market Insights for {selected_market} (Based on Filtered Data)",expanded=len(filtered_df)>=5):
            if len(filtered_df)<5: st.info(f"Select at least 5 properties (currently {len(filtered_df)}) to generate AI insights.")
            elif st.button(f"Generate Insights for {len(filtered_df)} Properties",key="gen_ins_main_button"):
                with st.spinner("🤖 AI is analyzing the data... please wait."):
                    insights_text = generate_ai_insights(filtered_df, df_full, selected_market)
                    st.markdown(insights_text, unsafe_allow_html=True) # allow HTML if AI returns complex markdown
            else: st.caption("Click the button to get AI-generated summary insights about the currently filtered properties.")

    st.subheader(f'📊 {selected_market} Market Overview (Filtered Selection)')
    tot_f_disp = len(filtered_df)
    res_f_disp = 0
    com_f_disp = 0
    if 'Type' in filtered_df.columns:
        res_f_disp = len(filtered_df[filtered_df['Type'].astype(str).str.lower()=='residential'])
        com_f_disp = len(filtered_df[filtered_df['Type'].astype(str).str.lower()=='commercial'])

    hp_f_disp = 0
    if 'Price' in filtered_df.columns and not filtered_df.empty:
        prices_disp = pd.to_numeric(filtered_df['Price'],errors='coerce').dropna()
        if not prices_disp.empty: hp_f_disp = prices_disp.max()

    aq_f_disp = np.nan
    if 'Data Quality Score' in filtered_df.columns and not filtered_df.empty:
        dq_scores_disp = pd.to_numeric(filtered_df['Data Quality Score'],errors='coerce').dropna()
        if not dq_scores_disp.empty: aq_f_disp = dq_scores_disp.mean()

    as_f_disp = np.nan
    if 'Size_SqFt' in filtered_df.columns and not filtered_df.empty:
        sizes_disp = pd.to_numeric(filtered_df['Size_SqFt'],errors='coerce').replace(0, np.nan).dropna()
        if not sizes_disp.empty: as_f_disp = sizes_disp.mean()

    metrics_main_display = [
        ("Total Properties",f"{tot_f_disp:,}"),
        ("Residential Props",f"{res_f_disp:,}"),
        ("Commercial Props",f"{com_f_disp:,}"),
        ("Highest Price",format_currency(hp_f_disp)),
        ("Avg. DQ Score",f"{aq_f_disp:.1f}/100" if pd.notna(aq_f_disp) else "N/A"),
        ("Avg. Size (Sq Ft)",f"{as_f_disp:,.0f}" if pd.notna(as_f_disp) else "N/A")
    ]
    cols_main_metrics_display = st.columns(len(metrics_main_display))
    for i,(label, val) in enumerate(metrics_main_display):
        with cols_main_metrics_display[i]: st.markdown(f'<div class="metric-card"><h3>{label}</h3><p>{val}</p></div>',unsafe_allow_html=True)

    st.subheader('📈 Detailed Market Visualizations (Filtered Selection)')
    s_c1_viz, s_c2_viz = st.columns([0.6,0.4])
    with s_c1_viz:
        if not filtered_df.empty and 'Price' in filtered_df.columns:
            prices_gt0_viz = pd.to_numeric(filtered_df['Price'],errors='coerce').dropna()
            prices_gt0_viz = prices_gt0_viz[prices_gt0_viz > 0]
            if not prices_gt0_viz.empty:
                st.markdown("##### Price Distribution (USD)")
                avg_price_viz, median_price_viz = prices_gt0_viz.mean(), prices_gt0_viz.median()
                fig_price_hist = px.histogram(prices_gt0_viz,nbins=30,title='Distribution of Property Prices',labels={'value':'Price (USD)'},color_discrete_sequence=[THEME_PLOTLY['primary']])
                fig_price_hist.add_vline(x=avg_price_viz,line_dash="dash",line_color="red",annotation_text=f"Avg: {format_currency(avg_price_viz)}",annotation_position="top right")
                fig_price_hist.add_vline(x=median_price_viz,line_dash="dash",line_color="green",annotation_text=f"Median: {format_currency(median_price_viz)}",annotation_position="bottom right" if median_price_viz < avg_price_viz else "top left")
                fig_price_hist.update_layout(title_x=0.5, paper_bgcolor=THEME_PLOTLY["paper_bgcolor"],plot_bgcolor=THEME_PLOTLY["plot_bgcolor"],font_color=THEME_PLOTLY["font_color"],yaxis_gridcolor=THEME_PLOTLY["grid_color"],xaxis_gridcolor=THEME_PLOTLY["grid_color"])
                st.plotly_chart(fig_price_hist,use_container_width=True)
                st.markdown(f"Key Price Stats: Avg: **{format_currency(avg_price_viz)}** | Median: **{format_currency(median_price_viz)}** | Range: **{format_currency(prices_gt0_viz.min())}** – **{format_currency(prices_gt0_viz.max())}**")
            else:
                st.caption("No properties with prices > $0 in the current selection to display price distribution.")
        else:
            st.caption("Price data not available for distribution chart in the current selection.")

    with s_c2_viz:
        if not filtered_df.empty and 'Parish' in filtered_df.columns:
            st.markdown("##### Parish Distribution")
            parish_counts_viz = filtered_df['Parish'].value_counts()
            parish_counts_known_viz = parish_counts_viz[parish_counts_viz.index!='Unknown']
            if not parish_counts_known_viz.empty and parish_counts_known_viz.sum()>0:
                parish_top_n_viz = parish_counts_known_viz.nlargest(5)
                fig_parish_pie = px.pie(parish_top_n_viz, names=parish_top_n_viz.index, values=parish_top_n_viz.values, title='Top Parishes (Excluding "Unknown")', color=parish_top_n_viz.index, color_discrete_sequence=THEME_PLOTLY["bar_palette"])
                fig_parish_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_parish_pie.update_layout(title_x=0.5, paper_bgcolor=THEME_PLOTLY["paper_bgcolor"],plot_bgcolor=THEME_PLOTLY["plot_bgcolor"],font_color=THEME_PLOTLY["font_color"],showlegend=False)
                st.plotly_chart(fig_parish_pie,use_container_width=True)
                if len(parish_top_n_viz.index) > 0:
                    st.markdown(f"Most active known parish: **{parish_top_n_viz.index[0]}** ({parish_top_n_viz.values[0]} properties).")
            else:
                st.caption("No parish data (excluding 'Unknown') to display distribution in the current selection.")
        else:
            st.caption("Parish data not available for distribution chart in the current selection.")

    if not filtered_df.empty and 'Property Type' in filtered_df.columns and 'Price' in filtered_df.columns:
        prices_gt0_for_avg = pd.to_numeric(filtered_df['Price'],errors='coerce').fillna(0)
        avg_price_df = filtered_df[prices_gt0_for_avg > 0].copy()
        if not avg_price_df.empty and 'Property Type' in avg_price_df.columns:
            avg_prices_by_type = avg_price_df.groupby('Property Type')['Price'].mean().sort_values(ascending=False)
            if not avg_prices_by_type.empty:
                st.markdown("##### Average Price by Property Type (USD)")
                fig_avg_price_bar = px.bar(avg_prices_by_type, x=avg_prices_by_type.values, y=avg_prices_by_type.index, orientation='h', title='Average Price by Property Type', color=avg_prices_by_type.index, color_discrete_sequence=THEME_PLOTLY["bar_palette"])
                fig_avg_price_bar.update_layout(title_x=0.5, paper_bgcolor=THEME_PLOTLY["paper_bgcolor"],plot_bgcolor=THEME_PLOTLY["plot_bgcolor"],font_color=THEME_PLOTLY["font_color"],yaxis_title='Property Type',xaxis_title='Average Price (USD)',showlegend=False)
                fig_avg_price_bar.update_traces(hovertemplate='<b>%{y}</b><br>Avg Price: $%{x:,.0f} USD'); st.plotly_chart(fig_avg_price_bar,use_container_width=True)

    if not filtered_df.empty and 'Bedrooms' in filtered_df.columns:
        st.markdown("##### Bedroom Analysis")
        bedrooms_gt0_df = filtered_df[pd.to_numeric(filtered_df['Bedrooms'],errors='coerce').fillna(0)>0]
        if not bedrooms_gt0_df.empty:
            bedroom_counts_viz = bedrooms_gt0_df['Bedrooms'].astype(int).value_counts().sort_index()
            if not bedroom_counts_viz.empty:
                fig_bedrooms_line = px.line(x=bedroom_counts_viz.index, y=bedroom_counts_viz.values, title='Property Count by Number of Bedrooms', markers=True, color_discrete_sequence=[THEME_PLOTLY['accent']])
                fig_bedrooms_line.update_layout(title_x=0.5, xaxis_title="Number of Bedrooms",yaxis_title="Number of Properties",paper_bgcolor=THEME_PLOTLY["paper_bgcolor"],plot_bgcolor=THEME_PLOTLY["plot_bgcolor"],font_color=THEME_PLOTLY["font_color"],yaxis_gridcolor=THEME_PLOTLY["grid_color"],xaxis_gridcolor=THEME_PLOTLY["grid_color"],xaxis=dict(tickmode='linear', dtick=1))
                st.plotly_chart(fig_bedrooms_line,use_container_width=True)
                st.markdown(f"Most common bedroom count (for properties with >0 bedrooms): **{int(bedroom_counts_viz.idxmax())}-bedroom** ({bedroom_counts_viz.max()} listings).")


    st.subheader(f'🌍 Interactive Property Map - {selected_market}')
    st.caption("📍 Markers indicate approximate parish locations, not exact addresses. Jitter is added for visibility. Use map layers to toggle amenities.")

    map_object = None # Initialize map_object
    if not filtered_df.empty or sel_amenities or sel_school_zones:
        map_object = create_advanced_map(filtered_df, selected_market, sel_amenities, sel_school_zones)
    
    if map_object:
        folium_static(map_object, width=None, height=600)
        legend_html_parts = []
        if 'Category' in filtered_df.columns and not filtered_df.empty:
            map_categories_unique = filtered_df['Category'].astype(str).str.upper().unique()
            if 'FOR SALE' in map_categories_unique: legend_html_parts.append('<div class="map-legend-item"><div class="map-legend-color-box" style="background:red;"></div>For Sale</div>')
            if 'FOR RENT' in map_categories_unique: legend_html_parts.append('<div class="map-legend-item"><div class="map-legend-color-box" style="background:green;"></div>For Rent</div>')
            if 'SOLD' in map_categories_unique: legend_html_parts.append('<div class="map-legend-item"><div class="map-legend-color-box" style="background:green;"></div>Sold</div>')
            if 'LEASED' in map_categories_unique: legend_html_parts.append('<div class="map-legend-item"><div class="map-legend-color-box" style="background:purple;"></div>Leased</div>')
            other_cats = [c for c in map_categories_unique if c not in ['FOR SALE', 'FOR RENT', 'SOLD', 'LEASED', 'UNKNOWN', ''] and pd.notna(c)]
            if other_cats: legend_html_parts.append('<div class="map-legend-item"><div class="map-legend-color-box" style="background:gray;"></div>Other Status</div>')

        if sel_school_zones and 'School' in amen_keys_map: legend_html_parts.append('<div class="map-legend-item"><div class="map-legend-color-box" style="background:orange;opacity:0.5;"></div>School Zone (1km)</div>')
        if legend_html_parts: st.markdown(f"""<div style="display:flex;flex-wrap:wrap;gap:10px;margin-top:5px;justify-content:center;">{''.join(legend_html_parts)}</div>""", unsafe_allow_html=True)
    elif filtered_df.empty and not (sel_amenities or sel_school_zones):
        st.info("No properties in the current filter to display on the map. Select amenities or adjust filters.")
    elif not (not filtered_df.empty or sel_amenities or sel_school_zones): # This condition means the initial if was false
        st.info("No properties selected and no map features (amenities/zones) enabled. Map not shown.")
    # If map_object is None due to other reasons like config error, create_advanced_map returns None, and this block won't execute without an explicit check for map_object being None when the first condition was true.
    # The current structure implies if the first condition is true but map_object is None, nothing is shown or messaged. This might be an edge case to consider if create_advanced_map can return None when it should ideally show a map.


    show_data_quality_report(filtered_df)
    
    # ========== TRANSACTION TYPE DISTRIBUTION ==========
    st.subheader('📑 Transaction Type Distribution (Filtered Selection)')
    if not filtered_df.empty and 'Category' in filtered_df.columns:
        use_two_cols_for_transactions = False
        if 'Type' in filtered_df.columns:
            if filtered_df['Type'].nunique(dropna=False) > 1 and \
               filtered_df['Category'].nunique(dropna=False) > 1:
                use_two_cols_for_transactions = True

        if use_two_cols_for_transactions:
            trans_col1, trans_col2 = st.columns(2)
            with trans_col1:
                create_transaction_pie_charts(filtered_df, THEME_PLOTLY)
            with trans_col2:
                crosstab_type_category = pd.crosstab(filtered_df['Type'], filtered_df['Category'])
                if not crosstab_type_category.empty and crosstab_type_category.values.sum() > 0:
                    crosstab_cats_upper = crosstab_type_category.columns.astype(str).str.upper()
                    bar_color_map_trans = {cat: THEME_PLOTLY["category_colors"].get(cat, THEME_PLOTLY["neutral_grey"]) for cat in crosstab_cats_upper}
                    fig_trans_bar = px.bar(crosstab_type_category, barmode='group', color_discrete_map=bar_color_map_trans, title="Transactions: Property Type vs. Category")
                    fig_trans_bar.update_layout(title_x=0.5, paper_bgcolor=THEME_PLOTLY["paper_bgcolor"],plot_bgcolor=THEME_PLOTLY["plot_bgcolor"],font_color=THEME_PLOTLY["font_color"],
                                                legend_title_text='Category', xaxis_title="Property Type", yaxis_title="Count",
                                                yaxis_gridcolor=THEME_PLOTLY["grid_color"],xaxis_gridcolor=THEME_PLOTLY["grid_color"])
                    st.plotly_chart(fig_trans_bar, use_container_width=True)
                else:
                    st.caption("No data for 'Type vs Category' breakdown in this selection.")
        else: 
            create_transaction_pie_charts(filtered_df, THEME_PLOTLY)
            if 'Type' not in filtered_df.columns:
                st.caption("Detailed breakdown by Property Type (second chart) requires 'Type' data.")
            elif not (filtered_df.get('Type', pd.Series(dtype=str)).nunique(dropna=False) > 1 and \
                      filtered_df['Category'].nunique(dropna=False) > 1):
                st.caption("Showing overall transaction distribution. More diversity in 'Property Type' and 'Category' data is needed for a side-by-side breakdown.")
    else: 
        create_transaction_pie_charts(filtered_df, THEME_PLOTLY)

    # ========== PROPERTY TYPE BREAKDOWN ==========
    st.subheader('🏘️ Property Type Breakdown (Filtered Selection)')
    has_type_col_breakdown = 'Type' in filtered_df.columns
    res_df_breakdown = pd.DataFrame()
    com_df_breakdown = pd.DataFrame()

    if has_type_col_breakdown and not filtered_df.empty:
        filtered_df_copy_for_type = filtered_df.copy()
        filtered_df_copy_for_type['Type_lower'] = filtered_df_copy_for_type['Type'].astype(str).str.lower()
        res_df_breakdown = filtered_df_copy_for_type[filtered_df_copy_for_type['Type_lower'] == 'residential']
        com_df_breakdown = filtered_df_copy_for_type[filtered_df_copy_for_type['Type_lower'] == 'commercial']

    has_residential_data = not res_df_breakdown.empty
    has_commercial_data = not com_df_breakdown.empty

    if has_residential_data and has_commercial_data:
        pt_col1, pt_col2 = st.columns(2)
        with pt_col1:
            st.markdown(f"<h5 style='text-align:center;color:{THEME_PLOTLY['primary']};'>Residential Property Types</h5>",unsafe_allow_html=True)
            create_property_type_bar_chart(res_df_breakdown,"Residential",THEME_PLOTLY)
        with pt_col2:
            st.markdown(f"<h5 style='text-align:center;color:{THEME_PLOTLY['primary']};'>Commercial Property Types</h5>",unsafe_allow_html=True)
            create_property_type_bar_chart(com_df_breakdown,"Commercial",THEME_PLOTLY,1)
    elif has_residential_data:
        st.markdown(f"<h5 style='text-align:center;color:{THEME_PLOTLY['primary']};'>Residential Property Types</h5>",unsafe_allow_html=True)
        create_property_type_bar_chart(res_df_breakdown,"Residential",THEME_PLOTLY)
    elif has_commercial_data:
        st.markdown(f"<h5 style='text-align:center;color:{THEME_PLOTLY['primary']};'>Commercial Property Types</h5>",unsafe_allow_html=True)
        create_property_type_bar_chart(com_df_breakdown,"Commercial",THEME_PLOTLY,1)
    else:
        if not has_type_col_breakdown and not filtered_df.empty:
            st.caption("Property 'Type' column is missing. Cannot provide Residential/Commercial breakdown.")
        elif filtered_df.empty:
            st.caption("No data in the current selection to provide Residential/Commercial breakdown.")
        else:
            st.caption("No Residential or Commercial properties found in the current selection for this breakdown.")

    # ========== PROPERTY DATA PREVIEW ==========
    st.subheader('📋 Property Data Preview (Filtered Selection)')
    if not filtered_df.empty:
        disp_df_preview = filtered_df.copy()
        default_preview_cols = {
            'Name':'N/A', 'Type':'N/A', 'Category':'N/A', 'Parish':'N/A',
            'Property Type':'N/A', 'Description':'N/A',
            'Price': 0, 'Size_SqFt': 0, 'Bedrooms': 0, 'Bathrooms': 0,
            'Data Quality Score':0
        }
        for col, default_val in default_preview_cols.items():
            if col not in disp_df_preview.columns:
                disp_df_preview[col] = default_val
            if col in ['Price', 'Size_SqFt', 'Bedrooms', 'Bathrooms', 'Data Quality Score']:
                disp_df_preview[col] = pd.to_numeric(disp_df_preview[col], errors='coerce').fillna(default_val if pd.isna(default_val) else float(default_val) )


        disp_df_preview['Price_f'] = disp_df_preview['Price'].apply(format_currency)
        disp_df_preview['Size_f'] = disp_df_preview['Size_SqFt'].apply(lambda x:f"{x:,.0f} Sq. Ft." if pd.notna(x)and x>0 else("0 Sq. Ft." if x==0 else "N/A"))
        disp_df_preview['DQ_f'] = disp_df_preview['Data Quality Score'].apply(lambda x:f"<span class='{get_data_quality_class(x)}'>{x:.0f}/100</span>" if pd.notna(x) else "N/A")
        disp_df_preview['Bedrooms'] = disp_df_preview['Bedrooms'].astype(int)
        disp_df_preview['Bathrooms'] = disp_df_preview['Bathrooms'].astype(int)

        display_order_preview = ['Name','Type','Category','Parish','Property Type','Price_f','Size_f','Bedrooms','Bathrooms','DQ_f','Description']
        final_cols_for_display = [c for c in display_order_preview if c in disp_df_preview.columns]

        rename_map_preview = {'Price_f':'Price (USD)','Size_f':'Size','DQ_f':'Data Quality'}
        disp_df_renamed_preview = disp_df_preview[final_cols_for_display].rename(columns=rename_map_preview)
        final_cols_renamed_preview = [rename_map_preview.get(c,c) for c in final_cols_for_display]

        if not disp_df_renamed_preview.empty:
            st.write(f'<div style="max-height:500px;overflow-y:auto;width:100%;">{disp_df_renamed_preview[final_cols_renamed_preview].to_html(escape=False,index=False,classes="dataframe",border=0)}</div>',unsafe_allow_html=True)

            cols_to_drop_for_dl = [c for c in ['lat','lon','coords', 'map_coords', 'Parish_lower', 'Type_lower'] if c in filtered_df.columns]
            download_df = filtered_df.drop(columns=cols_to_drop_for_dl, errors='ignore').copy()
            st.download_button(label="📥 Download Filtered Data as CSV",data=download_df.to_csv(index=False).encode('utf-8'),file_name=f'{selected_market.lower().replace(" ","_")}_filtered_properties.csv',mime='text/csv', key="download_csv_button")
    else:
        st.info(f"No property data to display or download for the current filters in {selected_market}.")

    st.markdown(f"""---<div style="text-align:center;color:var(--text-neutral);font-size:0.9em;padding-top:10px;"><p>Data Source: <a href="https://www.terracaribbean.com" target="_blank" style="color:var(--accent);">Terra Caribbean</a> • Displaying {len(filtered_df):,} properties based on filters.</p><p>© {pd.Timestamp.now().year} Terra Caribbean Market Analytics Platform • All Prices in USD</p><p>App Created by <b>Matthew Blackman</b>. Assisted by <b>AI</b>.</p></div>""",unsafe_allow_html=True)

if __name__ == "__main__":
    main()