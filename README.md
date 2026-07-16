# Netflix Content Recommendation System

**UMBC DATA 606 Capstone** · Harshitha Josyula · [LinkedIn](https://www.linkedin.com/in/harshitha-josyula-a348b91a8/) · [YouTube presentation](https://youtu.be/vXRN-VEEKVM)

A content-based recommendation system for Netflix's catalog of ~8,800 movies and TV shows. Instead of relying on user viewing history (collaborative filtering), it analyzes each title's own metadata — genre, cast, director, and plot description — to find and explain similar content. The project also includes a genre classification model and an interactive Streamlit app for exploring recommendations.

## Overview

Netflix's catalog is large enough that manual browsing and simple genre tags no longer help users find what they want. This project builds a content-based filtering pipeline that:

- Combines each title's description, genres, cast, director, and type into a single text profile
- Vectorizes those profiles with TF-IDF and computes pairwise cosine similarity across all 8,807 titles
- Returns the top-N most similar titles for any given title, with similarity scores
- Trains a Random Forest multi-label classifier to predict genres directly from plot descriptions
- Surfaces everything through a Streamlit dashboard for interactive exploration

## Results

- Mean similarity score across top-10 recommendations: **0.542**
- Manual validation across 50 titles: **94%** of recommendations rated satisfactory or better
- Description and genre fields are the most predictive features, together accounting for ~70% of recommendation quality (removing description drops mean similarity by 28.6%)
- Genre classification model: Random Forest with `MultiOutputClassifier`, trained on an 80/20 split with TF-IDF text features

Full methodology, EDA, and evaluation are in [docs/Report.md](docs/Report.md).

## Repository Structure

```
├── app/                    # Streamlit application (deployable copy)
│   └── streamlit_app.py
├── streamlit_app.py         # Root copy of the app, used for local runs
├── data/                    # Raw and cleaned datasets
│   ├── netflix_titles.csv           # Original Kaggle Netflix catalog (8,807 rows x 12 cols)
│   ├── netflix_cleaned.csv          # Cleaned/imputed version
│   └── netflix_content_database.csv # Feature-engineered data used by the app
├── notebooks/                # End-to-end pipeline, run in order
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_ml_models.ipynb
│   └── 04_streamlit.ipynb
└── docs/                      # Proposal, final report, resume, slides
    ├── Proposal.md
    ├── Report.md
    └── Final Presentation.pptx
```

## Data

Source: the publicly available Netflix Movies and TV Shows dataset (originally from Kaggle), included locally at [`data/netflix_titles.csv`](data/netflix_titles.csv).

- 8,807 titles × 12 columns: `show_id`, `type`, `title`, `director`, `cast`, `country`, `date_added`, `release_year`, `rating`, `duration`, `listed_in`, `description`
- Movies make up 69.6% of the catalog, TV shows 30.4%
- Notable missingness: `director` (30.7%), `cast` (9.2%), `country` (6.5%)

## Method

1. **Cleaning** (`01_data_cleaning.ipynb`): impute missing values, validate release years, engineer temporal features.
2. **EDA** (`02_eda.ipynb`): distribution analysis, genre co-occurrence, correlation checks, text analysis of descriptions and cast/director frequency.
3. **Modeling** (`03_ml_models.ipynb`):
   - Content-based recommender: TF-IDF (max 1,500 features, unigrams + bigrams, `min_df=2`, `max_df=0.8`) over combined text, then cosine similarity across the full 8,807 × 8,807 matrix.
   - Genre classifier: Random Forest (`n_estimators=100`, `max_depth=20`, `class_weight='balanced'`) wrapped in `MultiOutputClassifier` for multi-label genre prediction from descriptions.
4. **App** (`04_streamlit.ipynb` → `streamlit_app.py`): serves recommendations and visualizations through an interactive UI.

## Running the App

1. Download the pre-trained similarity/model artifacts from Google Drive: https://drive.google.com/file/d/1R-BzstyrIQhj7k4pP_KMzGCUssgHXX6d/view?usp=drive_link
2. Place the downloaded model alongside `data/netflix_content_database.csv`.
3. Install dependencies:
   ```bash
   pip install streamlit pandas plotly scikit-learn
   ```
4. Launch the app:
   ```bash
   streamlit run streamlit_app.py
   ```

Alternatively, open `notebooks/04_streamlit.ipynb` to see how the app is wired to the underlying data and model.

## Tech Stack

Python, pandas, scikit-learn (TF-IDF, cosine similarity, Random Forest), Streamlit, Plotly.

## Documentation

- [Project Proposal](docs/Proposal.md)
- [Final Report](docs/Report.md) — full background, EDA, modeling, results, limitations, and future work
- [Final Presentation](docs/Final%20Presentation.pptx)
- [YouTube walkthrough](https://youtu.be/vXRN-VEEKVM)

## Author

**Harshitha Josyula** -  UMBC Data Science Master's Capstone, advised by Dr. Chaojie (Jay) Wang.
