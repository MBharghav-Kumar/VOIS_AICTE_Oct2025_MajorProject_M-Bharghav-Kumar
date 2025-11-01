"""
Netflix Dataset Analysis - Interactive Streamlit Dashboard
==========================================================
Author: Your Name
Date: 2025
Description: Interactive dashboard for Netflix content trends analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import re
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Netflix Content Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    div[data-testid="metric-container"] {
        background-color: #1f1f1f;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 5px;
    }
    .plot-container {
        background-color: #1a1a1a;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data
def load_data(file):
    """Load and return the Netflix dataset"""
    try:
        df = pd.read_csv(file, engine='python')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def parse_duration(duration_str):
    """Extract numeric value and type from duration string"""
    if pd.isna(duration_str) or duration_str == 'Unknown':
        return np.nan, 'Unknown'
    
    duration_str = str(duration_str).strip()
    
    # Check for minutes (Movies)
    min_match = re.search(r'(\d+)\s*min', duration_str, re.IGNORECASE)
    if min_match:
        return int(min_match.group(1)), 'Minutes'
    
    # Check for seasons (TV Shows)
    season_match = re.search(r'(\d+)\s*season', duration_str, re.IGNORECASE)
    if season_match:
        return int(season_match.group(1)), 'Seasons'
    
    return np.nan, 'Other'

def split_and_explode(dataframe, column, separator=','):
    """Split and explode multi-valued columns"""
    df_copy = dataframe.copy()
    df_copy[column] = df_copy[column].fillna('Unknown')
    df_copy[column] = df_copy[column].astype(str).str.split(separator)
    df_copy = df_copy.explode(column)
    df_copy[column] = df_copy[column].str.strip()
    df_copy = df_copy[df_copy[column] != '']
    df_copy = df_copy[df_copy[column] != 'Unknown']
    return df_copy

@st.cache_data
def clean_data(df):
    """Clean and preprocess the dataset"""
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Drop Show_Id if exists
    if 'Show_Id' in df.columns:
        df = df.drop(columns=['Show_Id'])
    
    # Standardize column names
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    
    # Handle typo: 'case' instead of 'cast'
    if 'case' in df.columns and 'cast' not in df.columns:
        df = df.rename(columns={'case': 'cast'})
    
    # Convert Release_Date
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year'] = df['release_date'].dt.year
    
    # Extract year from string if missing
    missing_year_mask = df['release_year'].isna()
    if missing_year_mask.sum() > 0:
        df.loc[missing_year_mask, 'release_year'] = (
            df.loc[missing_year_mask, 'release_date']
            .astype(str)
            .str.extract(r'(\d{4})', expand=False)
            .astype(float)
        )
    
    df['release_year'] = df['release_year'].astype('Int64')
    
    # Handle missing values
    categorical_cols = ['director', 'cast', 'country', 'rating']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
    
    # Clean string columns
    string_cols = ['title', 'director', 'cast', 'country', 'rating', 
                   'category', 'type', 'description', 'duration']
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace('nan', 'Unknown')
    
    # Parse duration
    df[['duration_value', 'duration_type']] = df['duration'].apply(
        lambda x: pd.Series(parse_duration(x))
    )
    
    # Standardize category
    if 'category' in df.columns:
        df['category'] = df['category'].str.strip().str.title()
    
    # Add decade column
    df['decade'] = (df['release_year'] // 10 * 10).astype(str) + 's'
    
    return df

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown("""
        <h1 style='text-align: center; color: #E50914;'>
            🎬 Netflix Content Analysis Dashboard
        </h1>
        <p style='text-align: center; font-size: 18px; color: #999;'>
            Comprehensive analysis of Netflix Movies and TV Shows catalog
        </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", width=200)
        st.markdown("## 📊 Dataset Options")
        
        # Dataset source selection
        data_source = st.radio(
            "Choose data source:",
            ["Use Default Dataset", "Upload Your Own CSV"],
            help="Select whether to use the built-in Netflix dataset or upload your own"
        )
        
        uploaded_file = None
        use_default = True
        
        if data_source == "Upload Your Own CSV":
            use_default = False
            uploaded_file = st.file_uploader(
                "Choose your Netflix CSV file",
                type=['csv'],
                help="Upload a Netflix dataset CSV file"
            )
        else:
            st.success("✅ Using default Netflix dataset")
            # Set a flag to use default dataset
            uploaded_file = "default"
        
        st.markdown("---")
        
        # Quick demo section
        st.markdown("### 🎬 What You'll Discover")
        
        demo_col1, demo_col2 = st.columns(2)
        
        with demo_col1:
            st.markdown("""
            #### 📈 Key Insights
            - Content growth trends
            - Popular genres identification
            - Geographic distribution patterns
            - Rating preferences
            - Duration statistics
            """)
        
        with demo_col2:
            st.markdown("""
            #### 🎯 Strategic Value
            - Data-driven decisions
            - Market gap analysis
            - Trend forecasting
            - Competitive intelligence
            - Investment guidance
            """)
        
        # Sample data structure
        st.markdown("---")
        st.markdown("### 📋 Dataset Format")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info("""
            **Expected CSV columns:**
            - Title, Director, Cast, Country
            - Release_Date, Rating, Duration
            - Category (Movie/TV Show)
            - Type (Genres), Description
            """)
        
        with col2:
            st.success("""
            **Start Exploring!**
            
            👈 Select "Use Default Dataset" in the sidebar to begin
            """)
        
        # Features
        st.markdown("---")
        
        st.markdown("### ✨ Dashboard Features")
        
        features_col1, features_col2, features_col3 = st.columns(3)
        
        with features_col1:
            st.markdown("""
                #### 📊 Overview Analysis
                - Content type distribution
                - Yearly trends
                - Decade-wise breakdown
                - Key metrics dashboard
            """)
        
        with features_col2:
            st.markdown("""
                #### 🎭 Genre & Geography
                - Top genres identification
                - Genre trends over time
                - Country contributions
                - Regional analysis
            """)
        
        with features_col3:
            st.markdown("""
                #### ⭐ Deep Insights
                - Rating distributions
                - Duration patterns
                - Top directors & cast
                - Interactive heatmaps
            """)
        
        st.markdown("---")
        
        # Call to action
        st.markdown("""
            <div style='text-align: center; padding: 30px; background-color: #1a1a1a; border-radius: 10px;'>
                <h3 style='color: #E50914;'>🚀 Ready to Explore?</h3>
                <p style='font-size: 16px;'>
                    Select <strong>"Use Default Dataset"</strong> in the sidebar to start your analysis journey!
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("## ℹ️ About")
        st.info("""
        This dashboard provides comprehensive analysis of Netflix content including:
        - Content trends over time
        - Genre analysis
        - Country-wise contributions
        - Rating distributions
        - Duration patterns
        - Top talent insights
        """)
        
        st.markdown("---")
        st.markdown("### 📝 Instructions")
        st.markdown("""
        1. Choose to use default dataset or upload your own
        2. Explore different tabs
        3. Use filters to customize views
        4. Download insights as needed
        """)
    
    # Main content
    if uploaded_file is not None:
        # Load data
        with st.spinner("Loading data..."):
            if uploaded_file == "default":
                # Load default dataset
                try:
                    df = pd.read_csv('Netflix Dataset.csv', engine='python')
                    st.success("✅ Default dataset loaded successfully!")
                except FileNotFoundError:
                    st.error("""
                    ❌ Default dataset not found! 
                    
                    Please ensure 'Netflix Dataset.csv' is in the C:\\Netflix\\ folder.
                    
                    Or switch to 'Upload Your Own CSV' option in the sidebar.
                    """)
                    st.stop()
                except Exception as e:
                    st.error(f"Error loading default dataset: {e}")
                    st.stop()
            else:
                # Load uploaded dataset
                df = load_data(uploaded_file)
        
        if df is not None:
            # Clean data
            with st.spinner("Cleaning data..."):
                df = clean_data(df)
            
            # Create exploded dataframes
            df_genres = split_and_explode(df, 'type')
            df_countries = split_and_explode(df, 'country')
            df_cast = split_and_explode(df, 'cast')
            df_directors = split_and_explode(df, 'director')
            
            # Success message
            st.success("✅ Data loaded and cleaned successfully!")
            
            # Key Metrics Row
            st.markdown("## 📈 Key Metrics")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Titles", f"{len(df):,}")
            with col2:
                st.metric("Movies", f"{len(df[df['category']=='Movie']):,}")
            with col3:
                st.metric("TV Shows", f"{len(df[df['category']=='Tv Show']):,}")
            with col4:
                st.metric("Countries", f"{df_countries['country'].nunique()}")
            with col5:
                st.metric("Genres", f"{df_genres['type'].nunique()}")
            
            st.markdown("---")
            
            # Tabs for different analyses
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📊 Overview",
                "🎭 Genre Analysis",
                "🌍 Geographic Insights",
                "⭐ Ratings & Duration",
                "👥 Talent Analysis",
                "📄 Raw Data"
            ])
            
            # ================================================================
            # TAB 1: OVERVIEW
            # ================================================================
            with tab1:
                st.markdown("## 📊 Content Overview")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Content Distribution Pie Chart
                    st.markdown("### Content Type Distribution")
                    category_counts = df['category'].value_counts()
                    fig = px.pie(
                        values=category_counts.values,
                        names=category_counts.index,
                        title="Movies vs TV Shows",
                        color_discrete_sequence=['#E50914', '#B20710'],
                        hole=0.4
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Content by Decade
                    st.markdown("### Content by Decade")
                    decade_counts = df['decade'].value_counts().sort_index()
                    fig = px.bar(
                        x=decade_counts.index,
                        y=decade_counts.values,
                        labels={'x': 'Decade', 'y': 'Count'},
                        title="Content Distribution by Decade",
                        color=decade_counts.values,
                        color_continuous_scale='Reds'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                # Yearly Trend
                st.markdown("### 📈 Yearly Content Trend (Movies vs TV Shows)")
                
                # Year range selector
                year_min = int(df['release_year'].min())
                year_max = int(df['release_year'].max())
                
                col1, col2 = st.columns([3, 1])
                with col2:
                    year_range = st.slider(
                        "Select Year Range",
                        min_value=year_min,
                        max_value=year_max,
                        value=(max(year_min, 2000), year_max)
                    )
                
                # Filter data
                yearly_trend = (df[df['release_year'].between(year_range[0], year_range[1])]
                               .groupby(['release_year', 'category'])
                               .size()
                               .reset_index(name='count'))
                
                fig = px.line(
                    yearly_trend,
                    x='release_year',
                    y='count',
                    color='category',
                    markers=True,
                    title=f"Content Trend ({year_range[0]}-{year_range[1]})",
                    labels={'release_year': 'Year', 'count': 'Number of Titles', 'category': 'Type'},
                    color_discrete_map={'Movie': '#E50914', 'Tv Show': '#564d4d'}
                )
                fig.update_layout(hovermode='x unified', height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary Statistics
                st.markdown("### 📊 Summary Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("#### Content Stats")
                    st.write(f"**Total Unique Titles:** {df['title'].nunique()}")
                    st.write(f"**Date Range:** {year_min} - {year_max}")
                    st.write(f"**Unique Ratings:** {df['rating'].nunique()}")
                
                with col2:
                    st.markdown("#### Movie Stats")
                    movies = df[df['category'] == 'Movie']
                    movie_duration = movies[movies['duration_type'] == 'Minutes']['duration_value']
                    st.write(f"**Total Movies:** {len(movies):,}")
                    st.write(f"**Avg Duration:** {movie_duration.mean():.0f} min")
                    st.write(f"**Duration Range:** {movie_duration.min():.0f}-{movie_duration.max():.0f} min")
                
                with col3:
                    st.markdown("#### TV Show Stats")
                    tv_shows = df[df['category'] == 'Tv Show']
                    tv_seasons = tv_shows[tv_shows['duration_type'] == 'Seasons']['duration_value']
                    st.write(f"**Total TV Shows:** {len(tv_shows):,}")
                    st.write(f"**Avg Seasons:** {tv_seasons.mean():.1f}")
                    st.write(f"**Season Range:** {tv_seasons.min():.0f}-{tv_seasons.max():.0f}")
            
            # ================================================================
            # TAB 2: GENRE ANALYSIS
            # ================================================================
            with tab2:
                st.markdown("## 🎭 Genre Analysis")
                
                col1, col2 = st.columns([2, 1])
                
                with col2:
                    top_n = st.selectbox("Number of top genres to display", [10, 15, 20, 25], index=1)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Top Genres
                    st.markdown("### Top Genres Overall")
                    top_genres = df_genres['type'].value_counts().head(top_n)
                    fig = px.bar(
                        x=top_genres.values,
                        y=top_genres.index,
                        orientation='h',
                        title=f"Top {top_n} Genres",
                        labels={'x': 'Count', 'y': 'Genre'},
                        color=top_genres.values,
                        color_continuous_scale='Reds'
                    )
                    fig.update_layout(showlegend=False, height=600)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Genre by Content Type
                    st.markdown("### Genres by Content Type")
                    genre_by_category = (df_genres.groupby(['type', 'category'])
                                        .size()
                                        .reset_index(name='count')
                                        .sort_values('count', ascending=False)
                                        .head(top_n))
                    
                    fig = px.bar(
                        genre_by_category,
                        x='count',
                        y='type',
                        color='category',
                        orientation='h',
                        title=f"Top {top_n} Genres by Type",
                        labels={'count': 'Count', 'type': 'Genre', 'category': 'Content Type'},
                        color_discrete_map={'Movie': '#E50914', 'Tv Show': '#564d4d'}
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                # Genre Trends Over Time
                st.markdown("### 📈 Genre Trends Over Time")
                
                # Select genres
                available_genres = df_genres['type'].value_counts().head(15).index.tolist()
                selected_genres = st.multiselect(
                    "Select genres to compare",
                    options=available_genres,
                    default=available_genres[:5]
                )
                
                if selected_genres:
                    genre_yearly = (df_genres[df_genres['type'].isin(selected_genres)]
                                   .groupby(['release_year', 'type'])
                                   .size()
                                   .reset_index(name='count'))
                    
                    genre_yearly = genre_yearly[genre_yearly['release_year'] >= 2000]
                    
                    fig = px.line(
                        genre_yearly,
                        x='release_year',
                        y='count',
                        color='type',
                        markers=True,
                        title="Genre Popularity Trends (2000 onwards)",
                        labels={'release_year': 'Year', 'count': 'Number of Titles', 'type': 'Genre'}
                    )
                    fig.update_layout(hovermode='x unified', height=500)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Please select at least one genre to display trends.")
                
                # Add Genre-Category Heatmap
                st.markdown("---")
                st.markdown("### 🔥 Genre vs Content Type Heatmap")
                
                # Prepare data for genre-category heatmap
                top_genres_for_heatmap = df_genres['type'].value_counts().head(15).index.tolist()
                genre_category_data = (df_genres[df_genres['type'].isin(top_genres_for_heatmap)]
                                      .groupby(['type', 'category'])
                                      .size()
                                      .reset_index(name='count'))
                
                # Pivot for heatmap
                heatmap_pivot = genre_category_data.pivot_table(
                    index='type',
                    columns='category',
                    values='count',
                    fill_value=0
                )
                
                # Create Plotly heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=heatmap_pivot.values,
                    x=heatmap_pivot.columns,
                    y=heatmap_pivot.index,
                    colorscale='RdYlBu_r',
                    text=heatmap_pivot.values,
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    hoverongaps=False,
                    hovertemplate='Content Type: %{x}<br>Genre: %{y}<br>Count: %{z}<extra></extra>'
                ))
                
                fig.update_layout(
                    title='Top 15 Genres: Movies vs TV Shows Distribution',
                    xaxis_title='Content Type',
                    yaxis_title='Genre',
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # ================================================================
            # TAB 3: GEOGRAPHIC INSIGHTS
            # ================================================================
            with tab3:
                st.markdown("## 🌍 Geographic Insights")
                
                col1, col2 = st.columns([2, 1])
                
                with col2:
                    top_countries_n = st.selectbox("Number of top countries", [10, 15, 20, 25], index=1, key='countries')
                
                # Top Countries
                st.markdown("### Top Contributing Countries")
                top_countries = df_countries['country'].value_counts().head(top_countries_n)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(
                        x=top_countries.values,
                        y=top_countries.index,
                        orientation='h',
                        title=f"Top {top_countries_n} Countries by Content",
                        labels={'x': 'Number of Titles', 'y': 'Country'},
                        color=top_countries.values,
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(showlegend=False, height=600)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Pie chart for top 10
                    fig = px.pie(
                        values=top_countries.head(10).values,
                        names=top_countries.head(10).index,
                        title="Top 10 Countries Distribution",
                        color_discrete_sequence=px.colors.sequential.RdBu
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                # Country Trends
                st.markdown("### 📈 Country Production Trends")
                
                top_5_countries = top_countries.head(5).index.tolist()
                selected_countries = st.multiselect(
                    "Select countries to compare",
                    options=top_5_countries,
                    default=top_5_countries[:3]
                )
                
                if selected_countries:
                    country_yearly = (df_countries[df_countries['country'].isin(selected_countries)]
                                     .groupby(['release_year', 'country'])
                                     .size()
                                     .reset_index(name='count'))
                    
                    country_yearly = country_yearly[country_yearly['release_year'] >= 2010]
                    
                    fig = px.line(
                        country_yearly,
                        x='release_year',
                        y='count',
                        color='country',
                        markers=True,
                        title="Country Production Trends (2010 onwards)",
                        labels={'release_year': 'Year', 'count': 'Number of Titles', 'country': 'Country'}
                    )
                    fig.update_layout(hovermode='x unified', height=500)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Please select at least one country to display trends.")
                
                # Country by Content Type
                st.markdown("### 🎬 Content Type by Country")
                
                country_category = (df_countries.groupby(['country', 'category'])
                                   .size()
                                   .reset_index(name='count'))
                
                top_countries_list = top_countries.head(10).index.tolist()
                country_category_top = country_category[country_category['country'].isin(top_countries_list)]
                
                fig = px.bar(
                    country_category_top,
                    x='count',
                    y='country',
                    color='category',
                    orientation='h',
                    title="Content Type Distribution by Top Countries",
                    labels={'count': 'Count', 'country': 'Country', 'category': 'Type'},
                    color_discrete_map={'Movie': '#E50914', 'Tv Show': '#564d4d'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Genre-Year Heatmap
                st.markdown("---")
                st.markdown("### 🔥 Genre Popularity Heatmap (2010 onwards)")
                
                # Prepare data for heatmap
                top_8_genres_heatmap = df_genres['type'].value_counts().head(8).index.tolist()
                
                genre_year_heatmap = (df_genres[
                    (df_genres['type'].isin(top_8_genres_heatmap)) & 
                    (df_genres['release_year'] >= 2010)
                ].groupby(['release_year', 'type'])
                 .size()
                 .reset_index(name='count'))
                
                # Create pivot table for heatmap
                heatmap_data = genre_year_heatmap.pivot_table(
                    index='type', 
                    columns='release_year', 
                    values='count', 
                    fill_value=0
                )
                
                # Create Plotly heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=heatmap_data.values,
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    colorscale='Reds',
                    hoverongaps=False,
                    hovertemplate='Year: %{x}<br>Genre: %{y}<br>Count: %{z}<extra></extra>'
                ))
                
                fig.update_layout(
                    title='Genre Popularity Heatmap by Year',
                    xaxis_title='Release Year',
                    yaxis_title='Genre',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # ================================================================
            # TAB 4: RATINGS & DURATION
            # ================================================================
            with tab4:
                st.markdown("## ⭐ Ratings & Duration Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Rating Distribution
                    st.markdown("### Rating Distribution")
                    rating_counts = df['rating'].value_counts()
                    fig = px.bar(
                        x=rating_counts.values,
                        y=rating_counts.index,
                        orientation='h',
                        title="Content Ratings",
                        labels={'x': 'Count', 'y': 'Rating'},
                        color=rating_counts.values,
                        color_continuous_scale='Reds'
                    )
                    fig.update_layout(showlegend=False, height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Rating by Category
                    st.markdown("### Rating by Content Type")
                    rating_category = df.groupby(['rating', 'category']).size().unstack(fill_value=0)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(name='Movie', y=rating_category.index, 
                                        x=rating_category['Movie'], 
                                        orientation='h', marker_color='#E50914'))
                    fig.add_trace(go.Bar(name='TV Show', y=rating_category.index, 
                                        x=rating_category['Tv Show'], 
                                        orientation='h', marker_color='#564d4d'))
                    
                    fig.update_layout(
                        barmode='stack',
                        title="Rating Distribution by Content Type",
                        xaxis_title="Count",
                        yaxis_title="Rating",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                # Duration Analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🎬 Movie Duration Distribution")
                    movies_df = df[(df['category'] == 'Movie') & (df['duration_type'] == 'Minutes')]
                    movie_duration = movies_df['duration_value'].dropna()
                    
                    fig = px.histogram(
                        movie_duration,
                        nbins=50,
                        title="Movie Duration (Minutes)",
                        labels={'value': 'Duration (minutes)', 'count': 'Frequency'},
                        color_discrete_sequence=['#E50914']
                    )
                    fig.add_vline(x=movie_duration.mean(), line_dash="dash", 
                                 line_color="yellow", annotation_text=f"Mean: {movie_duration.mean():.0f} min")
                    fig.add_vline(x=movie_duration.median(), line_dash="dash", 
                                 line_color="green", annotation_text=f"Median: {movie_duration.median():.0f} min")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Movie duration stats
                    st.markdown("#### Movie Duration Statistics")
                    st.write(f"**Mean:** {movie_duration.mean():.2f} minutes")
                    st.write(f"**Median:** {movie_duration.median():.2f} minutes")
                    st.write(f"**Range:** {movie_duration.min():.0f} - {movie_duration.max():.0f} minutes")
                    st.write(f"**Standard Deviation:** {movie_duration.std():.2f} minutes")
                
                with col2:
                    st.markdown("### 📺 TV Show Seasons Distribution")
                    tv_df = df[(df['category'] == 'Tv Show') & (df['duration_type'] == 'Seasons')]
                    tv_seasons = tv_df['duration_value'].dropna()
                    
                    seasons_counts = tv_seasons.value_counts().sort_index()
                    fig = px.bar(
                        x=seasons_counts.index,
                        y=seasons_counts.values,
                        title="TV Show Seasons",
                        labels={'x': 'Number of Seasons', 'y': 'Frequency'},
                        color=seasons_counts.values,
                        color_continuous_scale='Reds'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # TV show duration stats
                    st.markdown("#### TV Show Season Statistics")
                    st.write(f"**Mean:** {tv_seasons.mean():.2f} seasons")
                    st.write(f"**Median:** {tv_seasons.median():.2f} seasons")
                    st.write(f"**Range:** {tv_seasons.min():.0f} - {tv_seasons.max():.0f} seasons")
                    st.write(f"**Mode:** {tv_seasons.mode()[0]:.0f} season(s)")
            
            # ================================================================
            # TAB 5: TALENT ANALYSIS
            # ================================================================
            with tab5:
                st.markdown("## 👥 Talent Analysis")
                
                col1, col2 = st.columns([2, 1])
                with col2:
                    talent_n = st.selectbox("Number of top talents", [10, 15, 20], index=1)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Top Directors
                    st.markdown("### 🎬 Top Directors")
                    top_directors = df_directors['director'].value_counts().head(talent_n)
                    fig = px.bar(
                        x=top_directors.values,
                        y=top_directors.index,
                        orientation='h',
                        title=f"Top {talent_n} Directors",
                        labels={'x': 'Number of Titles', 'y': 'Director'},
                        color=top_directors.values,
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(showlegend=False, height=600)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Top Cast
                    st.markdown("### 🌟 Top Cast Members")
                    top_cast = df_cast['cast'].value_counts().head(talent_n)
                    fig = px.bar(
                        x=top_cast.values,
                        y=top_cast.index,
                        orientation='h',
                        title=f"Top {talent_n} Cast Members",
                        labels={'x': 'Number of Titles', 'y': 'Cast Member'},
                        color=top_cast.values,
                        color_continuous_scale='Reds'
                    )
                    fig.update_layout(showlegend=False, height=600)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                # Talent Summary
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Unique Directors", f"{df_directors['director'].nunique():,}")
                with col2:
                    st.metric("Unique Cast Members", f"{df_cast['cast'].nunique():,}")
                with col3:
                    st.metric("Most Prolific Director", top_directors.index[0])
                with col4:
                    st.metric("Most Featured Actor", top_cast.index[0])
            
            # ================================================================
            # TAB 6: RAW DATA
            # ================================================================
            with tab6:
                st.markdown("## 📄 Raw Data Explorer")
                
                # Filters
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    category_filter = st.multiselect(
                        "Filter by Content Type",
                        options=df['category'].unique(),
                        default=df['category'].unique()
                    )
                
                with col2:
                    rating_filter = st.multiselect(
                        "Filter by Rating",
                        options=df['rating'].unique(),
                        default=df['rating'].unique()
                    )
                
                with col3:
                    year_filter = st.slider(
                        "Filter by Year",
                        min_value=int(df['release_year'].min()),
                        max_value=int(df['release_year'].max()),
                        value=(int(df['release_year'].min()), int(df['release_year'].max()))
                    )
                
                # Apply filters
                filtered_df = df[
                    (df['category'].isin(category_filter)) &
                    (df['rating'].isin(rating_filter)) &
                    (df['release_year'].between(year_filter[0], year_filter[1]))
                ]
                
                st.markdown(f"### Showing {len(filtered_df):,} of {len(df):,} records")
                
                # Display data
                st.dataframe(
                    filtered_df,
                    use_container_width=True,
                    height=600
                )
                
                # Download button
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Filtered Data as CSV",
                    data=csv,
                    file_name=f'netflix_filtered_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv',
                )
                
                # Data Summary
                st.markdown("### Data Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Records", f"{len(filtered_df):,}")
                with col2:
                    st.metric("Movies", f"{len(filtered_df[filtered_df['category']=='Movie']):,}")
                with col3:
                    st.metric("TV Shows", f"{len(filtered_df[filtered_df['category']=='Tv Show']):,}")
                with col4:
                    st.metric("Unique Titles", f"{filtered_df['title'].nunique():,}")
    
    else:
        # Landing page when no file is uploaded
        st.markdown("""
            <div style='text-align: center; padding: 50px;'>
                <h2>👋 Welcome to Netflix Content Analysis Dashboard</h2>
                <p style='font-size: 18px; color: #999;'>
                    Choose "Use Default Dataset" to start analyzing immediately,<br>
                    or upload your own Netflix dataset CSV file
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Show sample data info
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                ### 📊 Default Dataset
                - **7,789 titles** (Movies & TV Shows)
                - **Time Period**: 2008-2021
                - **50+ countries** represented
                - **20+ genres** analyzed
                - Ready to explore instantly!
            """)
        
        with col2:
            st.markdown("""
                ### 🎭 Analysis Features
                - Content distribution
                - Genre trends over time
                - Geographic insights
                - Rating & duration analysis
                - Top talent identification
                - Interactive visualizations
            """)
        
        with col3:
            st.markdown("""
                ### 🚀 Getting Started
                1. Choose data source in sidebar
                2. Click "Use Default Dataset"
                3. Explore 6 analysis tabs
                4. Use filters for deep dives
                5. Export insights as CSV
            """)
        
        st.markdown("---")
        
        # Footer
        st.markdown("""
            <div style='text-align: center; color: #666; padding: 20px;'>
                <p>Built with ❤️ using Streamlit | Netflix Dataset Analysis Project</p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()