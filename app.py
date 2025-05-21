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

    abs_file_path = "Could not determine absolute path" # Default
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
        # Updated regex to be more robust for numeric extraction before Sq. Ft./Acres
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
                 parish_variation_map[valid_name.lower()] = valid_name # Ensure canonical names map to themselves (with correct casing)
        
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
        response = requests.get(base_url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        return {"temp": data["main"]["temp"], "feels_like": data["main"]["feels_like"], "description": data["weather"][0]["description"].title(), "icon_url": f"http://openweathermap.org/img/wn/{data['weather'][0]['icon']}@2x.png", "humidity": data["main"]["humidity"], "wind_speed": data["wind"]["speed"], "city": data["name"]}
    except: return None

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
    if not OPENAI_API_KEY: return "AI features disabled (API key)."
    if len(filtered_df) < 5: return "Need min 5 properties for AI insights."
    summary = {
        "market": market_name, "filtered_count": len(filtered_df), "market_total": len(full_df),
        "price_USD": {}, "size_sqft": {},
        "prop_types": filtered_df['Property Type'].value_counts().to_dict() if 'Property Type' in filtered_df else {},
        "parishes": filtered_df['Parish'].value_counts().to_dict() if 'Parish' in filtered_df else {},
        "categories": filtered_df['Category'].value_counts().to_dict() if 'Category' in filtered_df else {}
    }
    if 'Price' in filtered_df and pd.api.types.is_numeric_dtype(filtered_df['Price']) and not filtered_df['Price'].empty:
        prices = filtered_df['Price'][filtered_df['Price'] > 0]; summary["price_USD"] = {"min": prices.min(), "max": prices.max(), "median": prices.median(), "mean": prices.mean()} if not prices.empty else {}
    if 'Size_SqFt' in filtered_df and pd.api.types.is_numeric_dtype(filtered_df['Size_SqFt']) and not filtered_df['Size_SqFt'].empty:
        sizes = filtered_df['Size_SqFt'][filtered_df['Size_SqFt'] > 0]; summary["size_sqft"] = {"min": sizes.min(), "max": sizes.max(), "median": sizes.median(), "mean": sizes.mean()} if not sizes.empty else {}
    prompt = f"Real estate data analysis for {market_name} (Prices USD, Size Sq.Ft.). Provide 3-5 key insights on price, property/size types, geography, market comparison, patterns. Data (Filtered): {summary}. Concise for investors/buyers."
    try:
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}], temperature=0.7, max_tokens=380)
        return response.choices[0].message.content
    except Exception as e: return f"AI insight failed: {str(e)}"

def apply_search_filter(df, search_term):
    if not search_term or df.empty: return df
    term = search_term.lower()
    cols = [c for c in ['Name','Description','Parish','Property Type','Category','Type'] if c in df.columns]
    if not cols: return df
    return df[df[cols].astype(str).fillna('').apply(lambda r: r.str.lower().str.contains(term, na=False)).any(axis=1)]

def natural_language_query(query, df):
    if not OPENAI_API_KEY: return apply_search_filter(df, query)
    try:
        prompt = f"User query: \"{query}\". DF cols: {list(df.columns)}. Text cols: ['Name','Description','Parish','Property Type','Category','Type']. Python pandas code to *further filter* `df`. Case-insensitive `.str.contains()`. Combine `&` or `|`. ONLY Python code. Examples: 'Christ Church' -> df[df['Parish'].str.contains('Christ Church',case=False,na=False)]; 'villas under 1M' -> df[(df['Category'].str.contains('For Sale',case=False,na=False))&(df['Property Type'].str.contains('villa',case=False,na=False))]. Code:"
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}], temperature=0.2, max_tokens=200)
        code = response.choices[0].message.content.strip()
        if code.startswith('df[') and code.endswith(']'):
            res_df = eval(code, {'df': df.copy(), 'pd': pd, 'np': np, 're': re})
            if isinstance(res_df, pd.DataFrame) and len(res_df) <= len(df): return res_df
        st.warning("AI query invalid filter. Using text search.")
    except Exception as e: st.error(f"AI query failed: {e}. Using text search.")
    return apply_search_filter(df, query)

def geocode_properties(df, island_name):
    if df.empty: return df
    config = MARKET_DATA_SOURCES.get(island_name, {})
    def_lat, def_lon = config.get('default_coords', (0,0))
    if 'Parish' not in df.columns: df['lat'], df['lon'] = def_lat, def_lon; return df
    coords = config.get('parish_coords', {})
    df['map_coords'] = df['Parish'].apply(lambda p: coords.get(p, (def_lat, def_lon)))
    df['lat'], df['lon'] = zip(*df['map_coords'])
    df = df.drop(columns=['map_coords'])
    mask = df['Parish'].isin(coords.keys()) & ~((df['lat'] == def_lat) & (df['lon'] == def_lon))
    if mask.any():
        indices = df.index[mask]
        if not indices.empty:
            df.loc[indices,'lat'] += np.random.uniform(-0.005,0.005,size=len(indices))
            df.loc[indices,'lon'] += np.random.uniform(-0.005,0.005,size=len(indices))
    return df

def create_advanced_map(filtered_df, island_name, amenities_to_show=None, show_school_districts=False):
    config = MARKET_DATA_SOURCES.get(island_name);
    if not config: return None
    def_lat, def_lon = config.get('default_coords', (0,0))
    center, zoom = ([def_lat, def_lon], 11)
    if not filtered_df.empty and 'lat' in filtered_df.columns and 'lon' in filtered_df.columns:
        v_coords = filtered_df[['lat','lon']].dropna().astype(float)
        if not v_coords.empty: center, zoom = ([v_coords['lat'].mean(), v_coords['lon'].mean()], 13)
    m = folium.Map(location=center, zoom_start=zoom, tiles='cartodbpositron' if st.session_state.current_theme=='Light' else 'cartodbdark_matter', control_scale=True)
    props_grp = folium.FeatureGroup(name=f'Properties ({island_name})').add_to(m)
    map_df = filtered_df.dropna(subset=['lat','lon']).copy()
    str_cols, num_cols = ['Name','Property Type','Parish','Category','Type'], ['Price','Bedrooms','Bathrooms','Data Quality Score','Size_SqFt']
    for c in str_cols: map_df[c] = map_df[c].astype(str).fillna('N/A') if c in map_df else 'N/A'
    for c in num_cols: map_df[c] = pd.to_numeric(map_df[c], errors='coerce').fillna(0) if c in map_df else 0 # Corrected: errors='coerce'
    for _, r in map_df.iterrows():
        cat = str(r.get('Category','U')).upper(); color = 'red' if cat=='FOR SALE' else 'green' if cat=='FOR RENT' else 'gray'
        icon = 'home' if str(r.get('Type','')).lower()=='residential' else 'building'
        popup = f"<b>{r.get('Name')}</b><br>Type: {r.get('Property Type')}<br>Parish: {r.get('Parish')}<br>Cat: {r.get('Category')}<br>Price: ${r.get('Price',0):,.0f}<br>Beds: {int(r.get('Bedrooms',0))} | Baths: {int(r.get('Bathrooms',0))}<br>Size: {r.get('Size_SqFt',0):,.0f} Sq.Ft.<br><i>DQ: {r.get('Data Quality Score',0):.0f}</i>"
        folium.Marker(location=[r['lat'],r['lon']], popup=folium.Popup(popup,max_width=300), icon=folium.Icon(color=color,icon=icon,prefix='fa')).add_to(props_grp)
    amenities = config.get('amenities',{})
    if amenities_to_show:
        for type, locs in amenities.items():
            if type in amenities_to_show and locs:
                clr,icn = ({'Beach':('lightblue','tint'),'School':('orange','info-sign'),'Restaurant':('red','cutlery')}).get(type,('gray','info-sign'))
                grp = folium.FeatureGroup(name=f"{type}s").add_to(m)
                for l in locs:
                    if all(k in l for k in ['lat','lon']) and pd.notna(l['lat']) and pd.notna(l['lon']):
                        try: folium.Marker(location=[float(l['lat']),float(l['lon'])],popup=f"{type}: {l.get('name','N/A')}",icon=folium.Icon(color=clr,icon=icn,prefix='glyphicon')).add_to(grp)
                        except ValueError: pass
    if show_school_districts and 'School' in amenities and amenities['School']:
        szg = folium.FeatureGroup(name='School Zones').add_to(m)
        for l in amenities['School']:
            if all(k in l for k in ['lat','lon']) and pd.notna(l['lat']) and pd.notna(l['lon']):
                try: folium.Circle(location=[float(l['lat']),float(l['lon'])],radius=1000,color='orange',fill=True,fill_opacity=0.2,popup=f"School Zone: {l.get('name','N/A')}").add_to(szg)
                except ValueError: pass
    folium.LayerControl().add_to(m); return m

def show_data_quality_report(df):
    if df.empty: st.caption("No DQ report."); return
    st.subheader("🔍 Data Quality Report")
    with st.expander("DQ Score Calc"): st.markdown("Base 100. Deductions: Price(-30), PropType(-20), Parish(-15), Desc(-15), Beds(-10), Baths(-10), Size(-10), Cat(-5), Type(-5).")
    if 'Data Quality Score' in df and not df['Data Quality Score'].empty:
        scores = pd.to_numeric(df['Data Quality Score'],errors='coerce').dropna() # Corrected: errors='coerce'
        if not scores.empty:
            overall = scores.mean()
            st.markdown(f"""<div style="background:var(--background-card);padding:15px;border-radius:8px;margin-bottom:15px;"><h3 style="margin-top:0;color:var(--primary);">Overall DQ</h3><p style="font-size:22px;margin-bottom:0;"><span class="{get_data_quality_class(overall)}">{overall:.1f}/100</span> ({len(df):,} props)</p></div>""", unsafe_allow_html=True)
    st.markdown("**Column Completeness (Missing/Default):**")
    comp_data, tot_rows = [], len(df)
    cols_dq_check = {
        'Price':lambda d:(pd.to_numeric(d.get('Price',0),errors='coerce').fillna(0)==0).sum(), # Corrected
        'Property Type':lambda d:(d.get('Property Type','U')=='U').sum(),
        'Parish':lambda d:(d.get('Parish','U')=='U').sum(),
        'Bedrooms':lambda d:(pd.to_numeric(d.get('Bedrooms',0),errors='coerce').fillna(0)==0).sum(), # Corrected
        'Bathrooms':lambda d:(pd.to_numeric(d.get('Bathrooms',0),errors='coerce').fillna(0)==0).sum(), # Corrected
        'Size_SqFt':lambda d:(pd.isna(d.get('Size_SqFt'))|(pd.to_numeric(d.get('Size_SqFt',0),errors='coerce').fillna(0)==0)).sum(), # Corrected
        'Description':lambda d:(d.get('Description','').astype(str).str.strip()=='').sum(),
        'Category':lambda d:(d.get('Category','U')=='U').sum(),
        'Type':lambda d:(d.get('Type','U')=='U').sum()
    }
    if tot_rows > 0:
        for name, func in cols_dq_check.items():
            if name in df.columns:
                try: comp_data.append({'Column':name,'Missing Values':func(df),'Total':tot_rows})
                except: pass
    if comp_data:
        comp = pd.DataFrame(comp_data); comp['% Complete'] = 100*(1-comp['Missing Values']/comp['Total'])
        cols_d = st.columns(min(len(comp),4 if len(comp)>1 else 1))
        for i, r_data in comp.iterrows():
            with cols_d[i%len(cols_d)]: st.markdown(f"""<div style="background:var(--background-card);padding:10px;border-radius:5px;margin-bottom:8px;"><p style="margin:0 0 3px 0;font-weight:bold;font-size:0.9em;color:var(--primary);">{r_data['Column']}</p><p style="margin:0;font-size:16px;"><span class="{get_data_quality_class(r_data['% Complete'])}">{r_data['% Complete']:.1f}%</span></p><p style="margin:3px 0 0 0;font-size:11px;color:var(--text-neutral);">{r_data['Missing Values']} missing/default</p></div>""", unsafe_allow_html=True)
    st.markdown("**DQ Score Distribution:**")
    if 'Data Quality Score' in df and not df['Data Quality Score'].empty:
        dq_scores_num = pd.to_numeric(df['Data Quality Score'],errors='coerce').fillna(0) # Corrected
        if not dq_scores_num.empty:
            fig = px.histogram(x=dq_scores_num,nbins=20,range_x=[0,100],title='DQ Scores',color_discrete_sequence=[THEME_PLOTLY['primary']])
            fig.update_layout(xaxis_title='Score',yaxis_title='Properties',paper_bgcolor=THEME_PLOTLY["paper_bgcolor"],plot_bgcolor=THEME_PLOTLY["plot_bgcolor"],font_color=THEME_PLOTLY["font_color"],yaxis_gridcolor=THEME_PLOTLY["grid_color"],xaxis_gridcolor=THEME_PLOTLY["grid_color"])
            st.plotly_chart(fig,use_container_width=True)

def create_transaction_pie_charts(df, theme): 
    if df.empty or 'Category' not in df: st.caption("No data for transaction pie."); return
    counts = df['Category'].value_counts()
    if counts.empty or counts.sum()==0: st.caption("No transaction counts."); return
    map_c = {str(k).upper(): theme["category_colors"].get(str(k).upper(),theme["neutral_grey"]) for k in counts.index}
    colors_c = [map_c.get(str(k).upper(),theme["neutral_grey"]) for k in counts.index]
    fig = px.pie(counts,names=counts.index,values=counts.values,color=counts.index,color_discrete_sequence=colors_c,hole=0.5,title="Overall Transaction Types")
    fig.update_traces(textinfo='percent+label',marker=dict(line=dict(color=theme["paper_bgcolor"],width=2)))
    fig.update_layout(title_x=0.5,paper_bgcolor=theme["paper_bgcolor"],plot_bgcolor=theme["plot_bgcolor"],font_color=theme["font_color"],legend_title_text='Category',legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
    st.plotly_chart(fig,use_container_width=True)

def create_property_type_bar_chart(df, title, theme, offset=0): 
    if df.empty or 'Property Type' not in df: st.caption(f"No data for {title}."); return
    counts = df['Property Type'].value_counts()
    if counts.empty or counts.sum()==0: st.caption(f"No counts for {title}."); return
    pal = theme["bar_palette"];
    if offset>0 and len(pal)>offset: pal=pal[offset:]+pal[:offset]
    if len(counts)>len(pal): pal=pal*(len(counts)//len(pal)+1)
    fig = px.bar(x=counts.index,y=counts.values,color=counts.index,color_discrete_sequence=pal[:len(counts)],labels={'y':'Count','x':'Property Type'},title=title)
    fig.update_layout(showlegend=False,title_x=0.5,xaxis_title=None,yaxis_title="Properties",paper_bgcolor=theme["paper_bgcolor"],plot_bgcolor=theme["plot_bgcolor"],font_color=theme["font_color"],yaxis_gridcolor=theme["grid_color"],xaxis_gridcolor=theme["grid_color"])
    st.plotly_chart(fig,use_container_width=True)

def format_currency(value):
    return f"${value:,.0f}" if isinstance(value,(int,float)) and pd.notna(value) and value!=0 else ("$0" if value==0 else "N/A")

def safe_isin_filter(df, col_name, selected_values):
    if col_name not in df.columns or not selected_values: return pd.Series(True, index=df.index)
    return df[col_name].astype(str).isin(selected_values)

def main():
    ai_enabled = initialize_ai()
    with st.sidebar:
        st.image("https://www.terracaribbean.com/SiteAssets/terra_caribbean.png?",width=200)
        if OPENWEATHERMAP_API_KEY: display_weather_sidebar(OPENWEATHERMAP_API_KEY); st.sidebar.markdown("---")
        st.title("Filters & Tools")
        markets_list = list(MARKET_DATA_SOURCES.keys())
        selected_market = st.selectbox("Select Market",options=markets_list,index=markets_list.index('Barbados') if 'Barbados' in markets_list else 0)
        sel_theme = st.radio("Theme",['Light','Dark'],index=0 if st.session_state.current_theme=='Light' else 1)
        if sel_theme != st.session_state.current_theme: st.session_state.current_theme = sel_theme; st.rerun()
        
        df_full = load_data(selected_market)
        if df_full.empty: st.error(f"Stopping: No data loaded for {selected_market}. Check file and logs."); return

        st.subheader("AI Search")
        nl_query = st.text_input(f"🔍 Ask about {selected_market}",placeholder="e.g., 'Beachfront under $1M'",help="Prices USD",disabled=not ai_enabled) if ai_enabled else ""
        if not ai_enabled: st.info("AI Search disabled.")

        st.subheader("Standard Filters")
        def get_unique_opts(df, col): return sorted(df[col].astype(str).unique()) if col in df.columns else []
        prop_type_opts = get_unique_opts(df_full,'Property Type')
        sel_prop_type = st.multiselect('Property Type',options=prop_type_opts,default=prop_type_opts)
        cat_opts = get_unique_opts(df_full,'Category')
        sel_category = st.multiselect('Transaction Type',options=cat_opts,default=cat_opts)
        par_opts = get_unique_opts(df_full,'Parish')
        def_parishes = [p for p in par_opts if p!='Unknown'] if 'Unknown' in par_opts and len(par_opts)>1 else par_opts
        sel_parish = st.multiselect('Parish',options=par_opts,default=def_parishes)
        
        p_min, p_max = 0.0,1.0
        if 'Price' in df_full:
            v_prices = pd.to_numeric(df_full['Price'],errors='coerce').dropna() # Corrected
            if not v_prices.empty: p_min,p_max = (float(v_prices.min()),float(v_prices.max()))
            if p_min==p_max: p_max+=1.0
            elif p_max>100000: p_max*=1.05
        p_step = max(1.0,(p_max-p_min)/200 if (p_max-p_min)>0 else 1.0)
        sel_price_range = st.slider('Price Range (USD)',min_value=p_min,max_value=p_max,value=(p_min,p_max),step=p_step,format="$%.0f")

        st.subheader("Map Features")
        amen_keys = list(MARKET_DATA_SOURCES.get(selected_market,{}).get('amenities',{}).keys())
        sel_amenities = st.multiselect("Amenities",options=amen_keys,default=[a for a in ['Beach','School','Restaurant'] if a in amen_keys])
        sel_school_zones = st.checkbox("School Zones (1km)",value=False) if 'School' in amen_keys else False
        
        st.subheader("Data Quality")
        sel_quality = st.slider("Min DQ Score",0,100,30,5) if 'Data Quality Score' in df_full else 0
        
    df_geocoded = geocode_properties(df_full.copy(),selected_market)
    
    conditions = (
        (safe_isin_filter(df_geocoded,'Property Type',sel_prop_type)) &
        (safe_isin_filter(df_geocoded,'Category',sel_category)) &
        (safe_isin_filter(df_geocoded,'Parish',sel_parish)) &
        (pd.to_numeric(df_geocoded['Price'],errors='coerce').fillna(0) >= sel_price_range[0]) & # Corrected
        (pd.to_numeric(df_geocoded['Price'],errors='coerce').fillna(0) <= sel_price_range[1]) & # Corrected
        (pd.to_numeric(df_geocoded['Data Quality Score'],errors='coerce').fillna(0) >= sel_quality) # Corrected
    )
    filtered_df = df_geocoded[conditions].copy()

    if nl_query:
        temp_df_nl = filtered_df.copy()
        if ai_enabled:
            res_nl = natural_language_query(nl_query,temp_df_nl)
            if res_nl is not None and not res_nl.empty: filtered_df=res_nl; st.success(f"AI refined to {len(filtered_df)} props.")
            elif len(temp_df_nl)>0: st.warning("AI no further results.")
            else: st.warning("No props match initial filters or AI query.")
        else:
            res_search = apply_search_filter(temp_df_nl,nl_query)
            if res_search is not None and not res_search.empty: filtered_df=res_search; st.info(f"Search found {len(filtered_df)} props.")
            elif len(temp_df_nl)>0: st.warning("Search no further results.")
            else: st.warning("No props match initial filters or search.")

    st.title(f"🏝️ Terra Caribbean Property Intelligence")
    st.markdown(f"""<span style="color:var(--text-neutral);font-size:1.1em;">Insights | <b>{selected_market}</b> | <b>{len(filtered_df):,}</b> props analyzed | Prices USD</span>""",unsafe_allow_html=True)

    if ai_enabled:
        with st.expander(f"💡 AI Insights for {selected_market}",expanded=len(filtered_df)>=5):
            if len(filtered_df)<5: st.info(f"Need min 5 props ({len(filtered_df)} selected).")
            elif st.button(f"Generate Insights ({len(filtered_df)} Props)",key="gen_ins_main"):
                with st.spinner("Analyzing..."): st.markdown(generate_ai_insights(filtered_df,df_geocoded,selected_market))
            else: st.caption("Click for AI insights (Prices USD, Sizes Sq.Ft.).")
    
    st.subheader(f'📊 {selected_market} Overview (Filtered)')
    tot_f,res_f,com_f = len(filtered_df),len(filtered_df[filtered_df.get('Type','')=='Residential']),len(filtered_df[filtered_df.get('Type','')=='Commercial'])
    hp_f = pd.to_numeric(filtered_df['Price'],errors='coerce').max() if not filtered_df.empty and 'Price' in filtered_df else 0 # Corrected
    aq_f = pd.to_numeric(filtered_df['Data Quality Score'],errors='coerce').mean() if not filtered_df.empty and 'Data Quality Score' in filtered_df else 0 # Corrected
    as_f = pd.to_numeric(filtered_df['Size_SqFt'],errors='coerce').mean() if not filtered_df.empty and 'Size_SqFt' in filtered_df else 0 # Corrected

    metrics_main = [("Total Props",f"{tot_f:,}"),("Residential",f"{res_f:,}"),("Commercial",f"{com_f:,}"),("Highest Price",format_currency(hp_f)),("Avg DQ",f"{aq_f:.1f}/100" if tot_f>0 and not np.isnan(aq_f) else "N/A"),("Avg Size (Sq Ft)",f"{as_f:,.0f}" if tot_f>0 and pd.notna(as_f) and as_f>0 else "N/A")]
    cols_main_metrics = st.columns(len(metrics_main))
    for i,(l,v) in enumerate(metrics_main):
        with cols_main_metrics[i]: st.markdown(f'<div class="metric-card"><h3>{l}</h3><p>{v}</p></div>',unsafe_allow_html=True)

    st.subheader('📈 Market Insights & Storytelling')
    s_c1,s_c2 = st.columns([0.6,0.4])
    with s_c1:
        if not filtered_df.empty and 'Price' in filtered_df:
            pr_gt0 = pd.to_numeric(filtered_df['Price'],errors='coerce').dropna(); pr_gt0=pr_gt0[pr_gt0>0] # Corrected
            if not pr_gt0.empty:
                st.markdown("### Price Distribution (USD)")
                av,md,p25,p75 = pr_gt0.mean(),pr_gt0.median(),pr_gt0.quantile(0.25),pr_gt0.quantile(0.75)
                fig=px.histogram(pr_gt0,nbins=30,title='Price Distribution',labels={'value':'Price (USD)'},color_discrete_sequence=[THEME_PLOTLY['primary']])
                fig.add_vline(x=av,line_dash="dash",line_color="red",annotation_text=f"Avg: {format_currency(av)}",annotation_position="top right")
                fig.add_vline(x=md,line_dash="dash",line_color="green",annotation_text=f"Median: {format_currency(md)}",annotation_position="top left")
                fig.update_layout(paper_bgcolor=THEME_PLOTLY["paper_bgcolor"],plot_bgcolor=THEME_PLOTLY["plot_bgcolor"],font_color=THEME_PLOTLY["font_color"],yaxis_gridcolor=THEME_PLOTLY["grid_color"],xaxis_gridcolor=THEME_PLOTLY["grid_color"])
                st.plotly_chart(fig,use_container_width=True)
                st.markdown(f"- Avg: **{format_currency(av)}** | Med: **{format_currency(md)}**\n- Range: **{format_currency(pr_gt0.min())}** to **{format_currency(pr_gt0.max())}**")
    with s_c2:
        if not filtered_df.empty and 'Parish' in filtered_df:
            st.markdown("### Parish Distribution")
            par_counts = filtered_df['Parish'].value_counts()
            if not par_counts.empty and par_counts.sum()>0:
                par_top = par_counts[par_counts.index!='Unknown'].nlargest(5)
                if not par_top.empty:
                    fig=px.pie(par_top,names=par_top.index,values=par_top.values,title='Top Parishes',color=par_top.index,color_discrete_sequence=THEME_PLOTLY["bar_palette"])
                    fig.update_traces(textposition='inside' if len(par_top)<=5 else 'outside',textinfo='percent+label')
                    fig.update_layout(paper_bgcolor=THEME_PLOTLY["paper_bgcolor"],plot_bgcolor=THEME_PLOTLY["plot_bgcolor"],font_color=THEME_PLOTLY["font_color"],showlegend=False,title_x=0.5)
                    st.plotly_chart(fig,use_container_width=True)
                    if tot_f>0: st.markdown(f"- **{par_top.index[0]}** most active ({par_top.values[0]} props).\n- Top {len(par_top)}: {par_top.sum()/tot_f*100:.0f}% of selection.")
    
    if not filtered_df.empty and 'Property Type' in filtered_df and 'Price' in filtered_df:
        apt_df_prices = pd.to_numeric(filtered_df['Price'],errors='coerce').fillna(0); apt_df = filtered_df[apt_df_prices>0] # Corrected
        if not apt_df.empty:
            apt = apt_df.groupby('Property Type')['Price'].mean().sort_values(ascending=False)
            if not apt.empty:
                st.markdown("### Avg Price by Property Type (USD)")
                fig = px.bar(apt,x=apt.values,y=apt.index,orientation='h',title='Avg Price by Property Type',color=apt.index,color_discrete_sequence=THEME_PLOTLY["bar_palette"])
                fig.update_layout(paper_bgcolor=THEME_PLOTLY["paper_bgcolor"],plot_bgcolor=THEME_PLOTLY["plot_bgcolor"],font_color=THEME_PLOTLY["font_color"],yaxis_title='Property Type',xaxis_title='Avg Price (USD)',showlegend=False,title_x=0.5)
                fig.update_traces(hovertemplate='<b>%{y}</b><br>Avg Price: $%{x:,.0f} USD'); st.plotly_chart(fig,use_container_width=True)
    
    if not filtered_df.empty and 'Bedrooms' in filtered_df:
        st.markdown("### Bedroom Analysis")
        beds0 = filtered_df[pd.to_numeric(filtered_df['Bedrooms'],errors='coerce').fillna(0)>0]['Bedrooms'].astype(int).value_counts().sort_index() # Corrected
        if not beds0.empty:
            fig=px.line(x=beds0.index,y=beds0.values,title='Properties by Bedrooms',markers=True,color_discrete_sequence=[THEME_PLOTLY['accent']])
            fig.update_layout(xaxis_title="Bedrooms",yaxis_title="Properties",paper_bgcolor=THEME_PLOTLY["paper_bgcolor"],plot_bgcolor=THEME_PLOTLY["plot_bgcolor"],font_color=THEME_PLOTLY["font_color"],yaxis_gridcolor=THEME_PLOTLY["grid_color"],xaxis_gridcolor=THEME_PLOTLY["grid_color"],xaxis_tickmode='linear')
            st.plotly_chart(fig,use_container_width=True); st.markdown(f"- Most common: **{int(beds0.idxmax())}-bedroom** ({beds0.max()} listings).")

    st.subheader(f'🌍 Interactive Map - {selected_market}'); st.caption("📍 Markers: parish-level approx. locations.")
    if not filtered_df.empty or sel_amenities or sel_school_zones:
        map_o = create_advanced_map(filtered_df,selected_market,sel_amenities,sel_school_zones)
        if map_o:
            folium_static(map_o,width=1200,height=600)
            leg_h = []
            map_cs = filtered_df['Category'].astype(str).str.upper().unique() if 'Category' in filtered_df and not filtered_df.empty else []
            if 'FOR SALE' in map_cs: leg_h.append('<div class="map-legend-item"><div class="map-legend-color-box" style="background:red;"></div>For Sale</div>')
            if 'FOR RENT' in map_cs: leg_h.append('<div class="map-legend-item"><div class="map-legend-color-box" style="background:green;"></div>For Rent</div>')
            if any(c not in ['FOR SALE','FOR RENT'] for c in map_cs): leg_h.append('<div class="map-legend-item"><div class="map-legend-color-box" style="background:gray;"></div>Other</div>')
            if sel_school_zones and 'School' in amen_keys: leg_h.append('<div class="map-legend-item"><div class="map-legend-color-box" style="background:orange;opacity:0.5;"></div>School Zone</div>')
            if leg_h: st.markdown(f"""<div style="display:flex;flex-wrap:wrap;gap:10px;margin-top:5px;justify-content:center;">{''.join(leg_h)}</div>""",unsafe_allow_html=True)
    
    show_data_quality_report(filtered_df)

    st.subheader('📑 Transaction Type Distribution')
    if not filtered_df.empty and 'Category' in filtered_df and 'Type' in filtered_df:
        use_cols_t = filtered_df['Type'].nunique(dropna=False)>1 and filtered_df['Category'].nunique(dropna=False)>1
        c_t1,c_t2 = st.columns(2) if use_cols_t else (st,None)
        with c_t1: create_transaction_pie_charts(filtered_df,THEME_PLOTLY)
        if c_t2:
            with c_t2:
                ttc = pd.crosstab(filtered_df['Type'],filtered_df['Category'])
                if not ttc.empty and ttc.values.sum()>0:
                    cmap_bar = {str(k).upper():THEME_PLOTLY["category_colors"].get(str(k).upper(),THEME_PLOTLY["neutral_grey"]) for k in ttc.columns}
                    fig_b = px.bar(ttc,barmode='group',color_discrete_map=cmap_bar,title="Transactions by Property Category")
                    fig_b.update_layout(title_x=0.5,paper_bgcolor=THEME_PLOTLY["paper_bgcolor"],plot_bgcolor=THEME_PLOTLY["plot_bgcolor"],font_color=THEME_PLOTLY["font_color"],legend_title_text='Transaction',xaxis_title=None,yaxis_gridcolor=THEME_PLOTLY["grid_color"],xaxis_gridcolor=THEME_PLOTLY["grid_color"])
                    st.plotly_chart(fig_b,use_container_width=True)
    
    st.subheader('🏘️ Property Type Breakdown')
    has_res_f = 'Type' in filtered_df and not filtered_df[filtered_df['Type'].astype(str).str.lower()=='residential'].empty
    has_com_f = 'Type' in filtered_df and not filtered_df[filtered_df['Type'].astype(str).str.lower()=='commercial'].empty
    c_p1,c_p2 = (st.columns(2) if has_res_f and has_com_f else (st,None))
    if has_res_f:
        with c_p1: st.markdown(f"<h4 style='text-align:center;color:{THEME_PLOTLY['primary']};'>Residential</h4>",unsafe_allow_html=True); create_property_type_bar_chart(filtered_df[filtered_df['Type'].astype(str).str.lower()=='residential'].copy(),"Residential Types",THEME_PLOTLY)
    if has_com_f and c_p2:
        with c_p2: st.markdown(f"<h4 style='text-align:center;color:{THEME_PLOTLY['primary']};'>Commercial</h4>",unsafe_allow_html=True); create_property_type_bar_chart(filtered_df[filtered_df['Type'].astype(str).str.lower()=='commercial'].copy(),"Commercial Types",THEME_PLOTLY,1)
    elif has_com_f: st.markdown(f"<h4 style='text-align:center;color:{THEME_PLOTLY['primary']};'>Commercial</h4>",unsafe_allow_html=True); create_property_type_bar_chart(filtered_df[filtered_df['Type'].astype(str).str.lower()=='commercial'].copy(),"Commercial Types",THEME_PLOTLY,1)
    if not has_res_f and not has_com_f: st.caption("No Residential/Commercial properties for breakdown.")

    st.subheader('📋 Property Data Preview')
    if not filtered_df.empty:
        disp_df = filtered_df.copy()
        for c,dv in {'Name':'N/A','Type':'N/A','Category':'N/A','Parish':'N/A','Property Type':'N/A','Description':'N/A'}.items():
            if c not in disp_df: disp_df[c]=dv
        disp_df['Price_f'] = pd.to_numeric(disp_df['Price'],errors='coerce').apply(lambda x:format_currency(x)) # Corrected
        disp_df['Size_f'] = pd.to_numeric(disp_df['Size_SqFt'],errors='coerce').apply(lambda x:f"{x:,.0f} Sq. Ft." if pd.notna(x)and x>0 else("0 Sq. Ft." if x==0 else "N/A")) # Corrected
        disp_df['DQ_f'] = pd.to_numeric(disp_df['Data Quality Score'],errors='coerce').apply(lambda x:f"<span class='{get_data_quality_class(x)}'>{x:.0f}/100</span>" if pd.notna(x) else "N/A") # Corrected
        for c in ['Bedrooms','Bathrooms']: disp_df[c]=pd.to_numeric(disp_df[c],errors='coerce').fillna(0).astype(int) # Corrected
        
        disp_order = ['Name','Type','Category','Parish','Property Type','Price_f','Size_f','Bedrooms','Bathrooms','DQ_f','Description']
        final_cols = [c for c in disp_order if c in disp_df]
        rename_map_disp = {'Price_f':'Price (USD)','Size_f':'Size','DQ_f':'Data Quality'}
        disp_df_renamed = disp_df[final_cols].rename(columns=rename_map_disp)
        final_cols_renamed = [rename_map_disp.get(c,c) for c in final_cols]

        if not disp_df_renamed.empty:
            st.write(f'<div style="max-height:500px;overflow-y:auto;">{disp_df_renamed[final_cols_renamed].to_html(escape=False,index=False,classes="dataframe",border=0)}</div>',unsafe_allow_html=True)
            dl_cols_drop = [c for c in ['lat','lon','coords'] if c in filtered_df] 
            dl_df = filtered_df.drop(columns=dl_cols_drop,errors='ignore').copy()
            st.download_button(label="📥 Download Data as CSV",data=dl_df.to_csv(index=False).encode('utf-8'),file_name=f'{selected_market.lower().replace(" ","_")}_properties.csv',mime='text/csv')
    else: st.caption(f"No property data for current filters in {selected_market}.")

    st.markdown(f"""---<div style="text-align:center;color:var(--text-neutral);font-size:0.9em;padding-top:10px;"><p>Data: <a href="https://www.terracaribbean.com" target="_blank" style="color:var(--accent);">Terra Caribbean</a> • {len(filtered_df):,} props analyzed</p><p>© {pd.Timestamp.now().year} Terra Caribbean Market Analytics • Prices USD</p><p>Created by <b>Matthew Blackman</b>. Assisted by <b>AI</b>.</p></div>""",unsafe_allow_html=True)

if __name__ == "__main__":
    main()
