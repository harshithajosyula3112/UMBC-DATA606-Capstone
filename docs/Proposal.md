# UMBC DATA606 Capstone – Netflix Movie Recommendation System Proposal

**Project Title:** Building a Content-Based Movie Recommendation System Using Netflix Catalog Data

**Prepared for:** UMBC Data Science Master Degree Capstone by Dr. Chaojie (Jay) Wang  

**Author:** Harshitha Josyula 

**GitHub Repository:** https://github.com/harshithajosyula3112/UMBC-DATA606-Capstone/tree/main

**LinkedIn Profile:** https://www.linkedin.com/in/harshitha-josyula-a348b91a8/ 

**PowerPoint Presentation:** 

## Background

Movie recommendation systems are critical components of modern streaming platforms, directly impacting user engagement, content discovery, and platform retention rates. With the overwhelming volume of content available on platforms like Netflix, users often struggle to find movies that match their preferences. Content-based filtering represents a fundamental approach to recommendation systems that analyzes item characteristics to suggest similar content.

The Netflix catalog contains rich metadata including genres, directors, cast members, plot descriptions, and content ratings that can be leveraged to build effective recommendation algorithms. Unlike collaborative filtering, content-based systems can provide recommendations immediately for new users and can explain why specific movies are suggested.

## Project Objective:
This project aims to build a comprehensive movie recommendation system using content-based filtering techniques to predict user preferences and suggest relevant movies based on content similarity and genre classification.

## Why it Matters:
Accurate movie recommendations help:
- **Streaming Platforms** increase user engagement and reduce churn rates
- **Content Creators** understand audience preferences for better content production  
- **Users** discover relevant movies efficiently, improving viewing satisfaction
- **Businesses** optimize content acquisition and licensing strategies

## Research Questions:

1. **Can we effectively predict movie similarity using content-based features (genres, cast, directors, descriptions)?**
2. **Which content characteristics (genre combinations, cast overlap, director style) are most predictive of user preferences?**
3. **Can we automatically classify movie genres from plot descriptions using machine learning?**
4. **How do content trends and patterns in the Netflix catalog evolve over time and geography?**

## Data

### Data Source:
- **Netflix Movies and TV Shows Dataset** (Open-source, publicly available)
- **Original Source:** Kaggle/Public datasets aggregating Netflix catalog information
- **Local Copy :** Available in this GitHub repository  
  https://github.com/harshithajosyula3112/UMBC-DATA606-Capstone/blob/151d491835d76d9fc8ce07bcac0dd31d0bc7e796/data/netflix_titles.csv

### Data Overview:
- **Dataset size:** ~2.5 MB
- **Shape:** 8,807 rows × 12 columns  
- **Content scope:** Global Netflix catalog including movies and TV shows
- **Time period:** Content spanning multiple decades up to recent releases
- Each row represents one title (movie or TV show) with comprehensive metadata including genres, cast, production details, and content descriptions.

### Column Data Types and Dictionary:

| Column Name | Data Type | Definition | Example Values |
|-------------|-----------|------------|----------------|
| show_id | String | Unique identifier for each title | s1, s2, s3 |
| type | String | Content type classification | Movie, TV Show |
| title | String | Name of the movie or TV show | "The Irishman", "Stranger Things" |
| director | String | Director(s) of the content | Martin Scorsese, Unknown Director |
| cast | String | Main actors and actresses | Robert De Niro, Al Pacino, Joe Pesci |
| country | String | Country/countries of origin | United States, India, United Kingdom |
| date_added | String | Date when content was added to Netflix | December 1, 2019 |
| release_year | Integer | Original release year of the content | 2019, 2020, 2021 |
| rating | String | Content rating classification | R, PG-13, TV-MA, PG |
| duration | String | Runtime for movies (minutes) or seasons for TV shows | 209 min, 3 Seasons |
| listed_in | String | Genres and categories | Dramas, Crime Movies, Comedies |
| description | String | Plot summary and content description | "Hit man Frank Sheeran looks back..." |

### Target Variables and Feature Candidates:

**Primary Task: Content-Based Recommendation (Similarity Prediction)**
- **Similarity Score** – Continuous value (0-1) representing content similarity between movies
- Calculated using cosine similarity of TF-IDF vectors created from combined content features

**Secondary Task: Genre Classification**
- **Genre Category** – Multi-label classification predicting primary genres from descriptions
- Target: Extracted from `listed_in` field (Drama, Comedy, Action, Thriller, etc.)

### Features:

These features capture content characteristics, production details, and textual information that enable the model to predict content similarity and classify genres effectively.

**Content Features:**
- **Genre combinations** (`listed_in`) – primary content categorization for similarity matching
- **Cast overlap** (`cast`) – shared actors indicating similar audience appeal
- **Director style** (`director`) – directorial patterns and filmmaking approaches
- **Plot content** (`description`) – textual analysis for thematic similarity

**Metadata Features:**
- **Release year** (`release_year`) – temporal patterns and era-specific content
- **Content rating** (`rating`) – audience targeting and content appropriateness
- **Geographic origin** (`country`) – cultural and regional content characteristics
- **Duration** (`duration`) – content length preferences and viewing patterns

**Engineered Features:**
- **TF-IDF vectors** from combined text (description + genres + cast)
- **Genre encoding** for multi-label classification
- **Content age categories** (Classic, Recent, New)
- **Market categories** (Hollywood, International, Independent)

**Explanation:** These features enable the recommendation system to understand content relationships through multiple dimensions - thematic similarity through descriptions, audience overlap through cast and directors, and categorical matching through genres. The combination creates a comprehensive content profile for accurate similarity calculation and meaningful recommendations.
