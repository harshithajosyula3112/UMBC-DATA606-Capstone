
import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import traceback
import os

# Page configuration
st.set_page_config(
    page_title="Netflix AI Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS - ONLY for main page elements, NOT for expanders
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Poppins:wght@300;400;600;700&display=swap');

    .main {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
        color: #ffffff;
    }

    .main-header {
        font-family: 'Bebas Neue', cursive;
        font-size: 4.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #E50914 0%, #B20710 50%, #E50914 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 2rem 0;
        letter-spacing: 4px;
        animation: glow 2s ease-in-out infinite alternate;
    }

    @keyframes glow {
        from { filter: drop-shadow(0 0 5px #E50914); }
        to { filter: drop-shadow(0 0 20px #E50914); }
    }

    .subtitle {
        font-family: 'Poppins', sans-serif;
        font-size: 1.2rem;
        color: #b3b3b3;
        text-align: center;
        margin-top: -1.5rem;
        margin-bottom: 2rem;
        font-weight: 300;
        letter-spacing: 2px;
    }

    .metric-card {
        background: linear-gradient(135deg, #1f1f1f 0%, #2d2d2d 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem;
        border: 1px solid #333;
        box-shadow: 0 8px 16px rgba(0,0,0,0.4);
        transition: all 0.3s ease;
        text-align: center;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(229, 9, 20, 0.3);
        border-color: #E50914;
    }

    .metric-value {
        font-family: 'Bebas Neue', cursive;
        font-size: 3rem;
        color: #E50914;
        font-weight: bold;
    }

    .metric-label {
        font-family: 'Poppins', sans-serif;
        font-size: 0.9rem;
        color: #b3b3b3;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .content-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #252525 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #E50914;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }

    .content-card:hover {
        transform: translateX(5px);
        box-shadow: 0 6px 20px rgba(229, 9, 20, 0.2);
    }

    .section-header {
        font-family: 'Bebas Neue', cursive;
        font-size: 2.5rem;
        color: #ffffff;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #E50914;
        letter-spacing: 2px;
    }

    .stButton>button {
        background: linear-gradient(135deg, #E50914 0%, #B20710 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        border: none;
        box-shadow: 0 4px 12px rgba(229, 9, 20, 0.4);
        transition: all 0.3s ease;
        font-family: 'Poppins', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #B20710 0%, #E50914 100%);
        box-shadow: 0 6px 20px rgba(229, 9, 20, 0.6);
        transform: translateY(-2px);
    }

    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #1f1f1f 0%, #2a2a2a 100%);
        border-radius: 10px;
        border: 1px solid #333;
        color: white;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
    }

    .streamlit-expanderHeader:hover {
        border-color: #E50914;
        background: linear-gradient(135deg, #2a2a2a 0%, #1f1f1f 100%);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_system():
    """Load the recommendation system"""
    try:
        if not os.path.exists('netflix_recommendation_system.pkl'):
            return {'error': 'File Not Found', 'message': 'netflix_recommendation_system.pkl not found.'}

        with open('netflix_recommendation_system.pkl', 'rb') as f:
            system = pickle.load(f)

        required_keys = ['df', 'cosine_sim', 'title_to_index']
        missing_keys = [key for key in required_keys if key not in system]

        if missing_keys:
            return {'error': 'Incomplete Data', 'message': f'Missing: {", ".join(missing_keys)}'}

        return system

    except Exception as e:
        return {'error': type(e).__name__, 'message': str(e)}

def get_recommendations(system, title, n=10):
    """Get recommendations for a given title"""
    df = system['df']
    cosine_sim = system['cosine_sim']
    title_to_index = system['title_to_index']

    if title not in title_to_index:
        return None

    idx = title_to_index[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    indices = [i[0] for i in sim_scores]

    recs = df.iloc[indices][['title', 'type', 'release_year', 'rating', 'duration', 'listed_in', 'description']].copy()
    recs['similarity'] = [score[1] for score in sim_scores]
    return recs

def display_home(df):
    """Beautiful home page"""
    st.markdown('<div class="main-header">🎬 NETFLIX AI RECOMMENDER</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Powered by Machine Learning & Content-Based Filtering</div>', unsafe_allow_html=True)

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">Total Titles</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df[df['type']=='Movie']):,}</div>
            <div class="metric-label">Movies</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df[df['type']=='TV Show']):,}</div>
            <div class="metric-label">TV Shows</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        countries = df['country'].nunique() if 'country' in df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{countries}</div>
            <div class="metric-label">Countries</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Recent additions
    st.markdown('<div class="section-header">🆕 Recently Added to Netflix</div>', unsafe_allow_html=True)

    recent = df.nlargest(8, 'release_year')[['title', 'type', 'release_year', 'rating', 'listed_in']]

    for idx, row in recent.iterrows():
        genres = row['listed_in'].split(',')[:3] if isinstance(row['listed_in'], str) else []
        genre_tags = ' • '.join([g.strip() for g in genres])

        st.markdown(f"""
        <div class="content-card">
            <h3 style="color: #E50914; margin: 0;">🎬 {row['title']}</h3>
            <p style="color: #b3b3b3; margin: 0.5rem 0;">
                <strong style="background: #ffd700; color: #000; padding: 0.2rem 0.5rem; border-radius: 3px;">{row['rating']}</strong> •
                <strong>{row['type']}</strong> •
                {row['release_year']}
            </p>
            <p style="color: #808080; font-size: 0.9rem; margin-top: 0.5rem;">{genre_tags}</p>
        </div>
        """, unsafe_allow_html=True)

def display_recommendations(system):
    """Recommendations page - COMPLETELY HTML-FREE IN EXPANDERS"""
    st.markdown('<div class="main-header">🔍 DISCOVER YOUR NEXT WATCH</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-Powered Content Recommendations Just For You</div>', unsafe_allow_html=True)

    df = system['df']

    # Search interface
    col1, col2 = st.columns([4, 1])

    with col1:
        title = st.selectbox(
            "🎬 Select a title you enjoyed:",
            sorted(df['title'].unique()),
            help="Choose a movie or TV show you like"
        )

    with col2:
        n = st.slider("📊 Results:", 5, 20, 10)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("✨ Get Recommendations", type="primary"):
        selected = df[df['title'] == title].iloc[0]

        # Selected title - HTML is OK here (outside expander)
        genres = selected['listed_in'].split(',')[:4] if isinstance(selected['listed_in'], str) else []
        genre_list = ' • '.join([g.strip() for g in genres])

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1f1f1f 0%, #2a2a2a 100%);
                    border: 3px solid #E50914; border-radius: 15px; padding: 1.5rem; margin: 1rem 0;">
            <h2 style="color: #E50914; margin-top: 0;">📺 You Selected:</h2>
            <h1 style="color: white; margin: 0.5rem 0;">{selected['title']}</h1>
            <p style="color: #b3b3b3; font-size: 1.1rem;">
                <span style="background: #ffd700; color: #000; padding: 0.3rem 0.6rem; border-radius: 3px; font-weight: bold;">{selected['rating']}</span> •
                <strong style="color: #E50914;">{selected['type']}</strong> •
                {selected['release_year']} •
                {selected['duration']}
            </p>
            <p style="color: #808080; margin: 1rem 0;">{genre_list}</p>
            <p style="color: #d0d0d0; line-height: 1.6; margin-top: 1rem;">
                {selected['description']}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Get recommendations
        with st.spinner('🎬 Finding perfect matches...'):
            recs = get_recommendations(system, title, n)

        if recs is not None:
            st.markdown('<div class="section-header">✨ Recommended For You</div>', unsafe_allow_html=True)

            # Display recommendations - ZERO HTML, ONLY NATIVE STREAMLIT COMPONENTS
            for i, (idx, row) in enumerate(recs.iterrows(), 1):
                similarity_pct = row['similarity'] * 100

                # Determine match emoji and label
                if similarity_pct >= 80:
                    match_emoji = "🟢"
                    match_label = "Excellent Match"
                elif similarity_pct >= 60:
                    match_emoji = "🟡"
                    match_label = "Great Match"
                else:
                    match_emoji = "🔴"
                    match_label = "Good Match"

                # Create expander with title
                expander_title = f"#{i} • {row['title']} • {row['type']} • {match_emoji} {similarity_pct:.0f}% {match_label}"

                # INSIDE EXPANDER: ONLY NATIVE STREAMLIT - NO HTML AT ALL
                with st.expander(expander_title, expanded=(i <= 3)):
                    # Title
                    st.subheader(f"🎬 {row['title']}")

                    # Create 3 columns for info
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Type", row['type'])
                        st.metric("Year", row['release_year'])

                    with col2:
                        st.metric("Rating", row['rating'])
                        st.metric("Duration", row['duration'])

                    with col3:
                        st.metric("Match Score", f"{similarity_pct:.0f}%")
                        st.write(f"{match_emoji} **{match_label}**")

                    # Genres
                    st.write("**Genres:**")
                    genres = row['listed_in'].split(',') if isinstance(row['listed_in'], str) else []
                    genre_list = ' • '.join([g.strip() for g in genres[:5]])
                    st.write(genre_list)

                    # Divider
                    st.divider()

                    # Description
                    st.write("**Description:**")
                    st.write(row['description'])
        else:
            st.error("❌ Could not generate recommendations")

def display_stats(df):
    """Statistics page"""
    st.markdown('<div class="main-header">📊 CONTENT ANALYTICS</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Explore Netflix Content Trends & Insights</div>', unsafe_allow_html=True)

    # Content Type Distribution
    st.markdown('<div class="section-header">🎭 Content Type Distribution</div>', unsafe_allow_html=True)

    type_counts = df['type'].value_counts()
    fig_type = go.Figure(data=[go.Pie(
        labels=type_counts.index,
        values=type_counts.values,
        hole=0.5,
        marker=dict(colors=['#E50914', '#B20710']),
        textfont=dict(size=16, color='white'),
        hovertemplate='<b>%{label}</b><br>%{value} titles<extra></extra>'
    )])

    fig_type.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=14),
        height=500
    )

    st.plotly_chart(fig_type, use_container_width=True)

    # Top Genres
    if 'listed_in' in df.columns:
        st.markdown('<div class="section-header">🎬 Most Popular Genres</div>', unsafe_allow_html=True)

        all_genres = []
        for genres in df['listed_in'].dropna():
            all_genres.extend([g.strip() for g in str(genres).split(',')])
        genre_counts = Counter(all_genres)
        top_genres = pd.DataFrame(genre_counts.most_common(15), columns=['Genre', 'Count'])

        fig_genres = go.Figure(data=[go.Bar(
            y=top_genres['Genre'],
            x=top_genres['Count'],
            orientation='h',
            marker=dict(
                color=top_genres['Count'],
                colorscale=[[0, '#B20710'], [0.5, '#E50914'], [1, '#ff6b6b']]
            ),
            text=top_genres['Count'],
            textposition='outside'
        )])

        fig_genres.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(26,26,26,0.5)',
            font=dict(color='white', size=12),
            yaxis=dict(categoryorder='total ascending'),
            height=600
        )

        st.plotly_chart(fig_genres, use_container_width=True)

    # Timeline
    if 'release_year' in df.columns:
        st.markdown('<div class="section-header">📅 Content Timeline</div>', unsafe_allow_html=True)

        year_type = df.groupby(['release_year', 'type']).size().reset_index(name='count')

        fig_timeline = go.Figure()

        for content_type in year_type['type'].unique():
            data = year_type[year_type['type'] == content_type]
            color = '#E50914' if content_type == 'Movie' else '#4da6ff'

            fig_timeline.add_trace(go.Scatter(
                x=data['release_year'],
                y=data['count'],
                mode='lines',
                name=content_type,
                line=dict(color=color, width=3),
                fill='tozeroy'
            ))

        fig_timeline.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(26,26,26,0.5)',
            font=dict(color='white'),
            height=500
        )

        st.plotly_chart(fig_timeline, use_container_width=True)

def main():
    """Main application"""

    system = load_system()

    if isinstance(system, dict) and 'error' in system:
        st.error(f"❌ {system['error']}: {system['message']}")
        return

    df = system['df']

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="color: #E50914; font-family: 'Bebas Neue', cursive; font-size: 2.5rem;">
                🎬 NETFLIX AI
            </h1>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        page = st.radio("NAVIGATION", ["🏠 Home", "🔍 Discover", "📊 Analytics"])

        st.markdown("---")

        st.info(f"""
        **📊 Quick Stats**

        **Total:** {len(df):,}
        **Movies:** {len(df[df['type']=='Movie']):,}
        **TV Shows:** {len(df[df['type']=='TV Show']):,}
        """)

    # Main content
    if page == "🏠 Home":
        display_home(df)
    elif page == "🔍 Discover":
        display_recommendations(system)
    elif page == "📊 Analytics":
        display_stats(df)

if __name__ == "__main__":
    main()
