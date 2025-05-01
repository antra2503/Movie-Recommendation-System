import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
from PIL import Image
import base64
import zipfile

# Set page config
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)),
                    url('https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white;
        padding: 2rem;
    }
    
    .stTitle {
        font-size: 3rem;
        color: #FFD700;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .stSelectbox {
        background-color: rgba(0, 0, 0, 0.7);
        color: white;
        border-radius: 10px;
        padding: 1rem;
    }
    
    .stButton>button {
        background-color: #FFD700;
        color: black;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    .stButton>button:hover {
        background-color: #FFA500;
    }
    
    .movie-card {
        background-color: rgba(0, 0, 0, 0.7);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem;
        transition: transform 0.3s;
    }
    
    .movie-card:hover {
        transform: scale(1.05);
    }
    
    .movie-title {
        color: #FFD700;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .movie-overview {
        color: white;
        font-size: 0.9rem;
    }
    
    .sidebar .sidebar-content {
        background-color: rgba(0, 0, 0, 0.8);
    }
    </style>
    """, unsafe_allow_html=True)

# Title with movie icon
st.markdown("""
    <div style="text-align: center;">
        <h1 style="color: #FFD700; font-size: 3rem; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);">
            🎬 Movie Recommendation System
        </h1>
    </div>
""", unsafe_allow_html=True)

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1

@st.cache_data
def load_and_process_data():
    # Load the movies data
    movies = pd.read_csv('tmdb_5000_movies.csv')
    
    # Check if the credits file is a zip and extract if necessary
    try:
        with zipfile.ZipFile('tmdb_5000_credits.zip', 'r') as zip_ref:
            zip_ref.extract('tmdb_5000_credits.csv', path='.')
    except Exception as e:
        st.error(f"Error extracting credits file: {e}")
        return None, None
    
    # Now load the extracted credits file
    credits = pd.read_csv('tmdb_5000_credits.csv')
    
    # Merge the datasets
    movies = movies.merge(credits, on='title')
    
    # Select relevant columns
    movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
    
    # Drop rows with missing values
    movies.dropna(inplace=True)
    
    # Convert string representations to lists
    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert3)
    movies['crew'] = movies['crew'].apply(fetch_director)
    
    # Collapse spaces in names
    movies['cast'] = movies['cast'].apply(collapse)
    movies['crew'] = movies['crew'].apply(collapse)
    movies['genres'] = movies['genres'].apply(collapse)
    movies['keywords'] = movies['keywords'].apply(collapse)
    
    # Create tags column
    movies['tags'] = movies['overview'] + ' ' + \
                    movies['genres'].apply(lambda x: ' '.join(x)) + ' ' + \
                    movies['keywords'].apply(lambda x: ' '.join(x)) + ' ' + \
                    movies['cast'].apply(lambda x: ' '.join(x)) + ' ' + \
                    movies['crew'].apply(lambda x: ' '.join(x))
    
    # Create similarity matrix
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tags']).toarray()
    similarity = cosine_similarity(vectors)
    
    return movies, similarity

def get_recommendations(movie_title, movies_df, similarity):
    # Get the index of the movie
    idx = movies_df[movies_df['title'] == movie_title].index[0]
    
    # Get similarity scores for the movie
    sim_scores = list(enumerate(similarity[idx]))
    
    # Sort movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top 5 similar movies (excluding the movie itself)
    sim_scores = sim_scores[1:6]
    
    # Get movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return recommended movies
    return movies_df.iloc[movie_indices][['title', 'overview']]

try:
    # Load and process data
    movies_df, similarity = load_and_process_data()
    
    # Create movie selector with custom styling
    st.markdown("""
        <div style="background-color: rgba(0, 0, 0, 0.7); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
            <h2 style="color: #FFD700; text-align: center;">Select Your Favorite Movie</h2>
        </div>
    """, unsafe_allow_html=True)
    
    selected_movie = st.selectbox(
        "",
        options=movies_df['title'].values,
        key="movie_selector"
    )
    
    if selected_movie:
        # Display selected movie details with custom styling
        movie_info = movies_df[movies_df['title'] == selected_movie].iloc[0]
        
        st.markdown("""
            <div style="background-color: rgba(0, 0, 0, 0.7); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
                <h2 style="color: #FFD700; text-align: center;">Selected Movie</h2>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
                <div class="movie-card">
                    <h3 class="movie-title">{movie_info['title']}</h3>
                </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
                <div class="movie-card">
                    <h3 style="color: #FFD700;">Overview</h3>
                    <p class="movie-overview">{movie_info['overview']}</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Get and display recommendations with custom styling
        st.markdown("""
            <div style="background-color: rgba(0, 0, 0, 0.7); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
                <h2 style="color: #FFD700; text-align: center;">Recommended Movies</h2>
            </div>
        """, unsafe_allow_html=True)
        
        recommendations = get_recommendations(selected_movie, movies_df, similarity)
        
        # Display recommendations in a nice grid
        cols = st.columns(5)
        for idx, (_, row) in enumerate(recommendations.iterrows()):
            with cols[idx]:
                st.markdown(f"""
                    <div class="movie-card">
                        <h3 class="movie-title">{row['title']}</h3>
                        <div class="movie-overview">
                            {row['overview'][:150]}...
                        </div>
                    </div>
                """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please make sure both 'tmdb_5000_movies.csv' and 'tmdb_5000_credits.csv' files are present in the directory and try again.")
