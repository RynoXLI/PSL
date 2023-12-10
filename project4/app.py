"""
# Project 4: Movie Recommender System

CS 598 Practical Statistical Learning

2023-12-10

UIUC Fall 2023

**Authors**
* Ryan Fogle
    - rsfogle2@illinois.edu
    - UIN: 652628818
* Sean Enright
    - seanre2@illinois.edu
    - UIN: 661791377
"""
from urllib import request
from math import ceil
import numpy as np
import streamlit as st
import pandas as pd

@st.cache_data
def myIBCF(s, newuser, mov_rate_genre, genre_top_recs, num_recs=10):

    recs = newuser.copy(deep=True).rename("PredictedRating")
    recs.iloc[:] = np.nan
    
    i_in_w = ~np.isnan(newuser)
    # Compute predicted rating for all non-rated movies
    for l in np.arange(newuser.shape[0])[np.isnan(newuser)]:
        s_li = s.iloc[l, :]
        i_in_sl = ~np.isnan(s_li)
        col_mask = np.logical_and(i_in_sl, i_in_w)
        if s_li[col_mask].sum() == 0:
            continue
        recs.iloc[l] = (
            1 / (s_li[col_mask].sum())
            * np.dot(s_li[col_mask], newuser[col_mask])
        )
    recs = recs[~np.isnan(recs)] 
    
    movie_recs = mov_rate_genre.join(recs, how="inner")
    movie_recs.sort_values(by=["PredictedRating", "WeightedRating"],
                           axis=0, ascending=False, inplace=True)
    if movie_recs.shape[0] >= num_recs:
        movie_recs = movie_recs.iloc[:num_recs, :]
        rec_movie_ids = movie_recs.index.tolist()
    else:
        # Begin with all available recommendations
        rec_movie_ids = movie_recs.index.tolist()
        
        # Add remainding recommendations based on most rated genre
        addl_recs = num_recs - movie_recs.shape[0]
        
        # Identify most-rated genre
        rated_genres = mov_rate_genre["Genres"][~np.isnan(newuser)]
        genre_tup = np.unique(np.concatenate(rated_genres.values),
                              return_counts=True)
        most_watched_genre = genre_tup[0][np.argsort(genre_tup[1])[-1]]
        
        # Identify highest-rated movies in this genre
        genre_recs = genre_top_recs[
            genre_top_recs["Genre"] == most_watched_genre]
        
        # Check that top movies in genre are unrated
        genre_recs.loc[:, "MovieID"] = (
            "m" + genre_recs.loc[:, "MovieID"].astype(str))
        unwatched = [m not in newuser[i_in_w].index.tolist() 
                        for m in genre_recs["MovieID"].tolist()]
        genre_recs = genre_recs[unwatched]["MovieID"][:addl_recs].tolist()
        
        rec_movie_ids += genre_recs
    return rec_movie_ids

def get_img(mid):
    return f"https://liangfgithub.github.io/MovieImages/{mid}.jpg?raw=true"

def make_rating_df(source_df, mids):
    df = source_df.loc[mids].reset_index()
    img_urls = df.loc[:, "MovieID"].apply(lambda x: get_img(x[1:]))
    df = df.assign(Image = img_urls)
    return df

def show_movies(title_df, num_cols=5, get_input=False):
    """Display a grid of movies, showing the title, image, rating and number
    of ratings for each.

    Args:
        title_df (pandas.DatFrame): dataframe with movie, img and rating
        num_cols (int, optional): # of columns. Defaults to 5.
    """
    num_movies = title_df.shape[0]
    num_rows = ceil(num_movies / num_cols)
    start_idx = 0
    widget_key = [None] * num_cols
    emt = [None] * num_cols
    for _ in range(num_rows):
        idx = range(start_idx, start_idx + np.min((num_cols,
                                                   num_movies - start_idx)))
        with st.container():
            style = "<style>p {align: center;}</style>"
            st.markdown(style, unsafe_allow_html=True)
            cols = st.columns(num_cols)
            for i, title in enumerate(title_df.loc[idx, "Title"]):
                cols[i].markdown(f"<b>{title}</b>", unsafe_allow_html=True)
                widget_key[i] = title
        with st.container():
            cols = st.columns(num_cols)
            for i, img in enumerate(title_df.loc[idx, "Image"]):
                cols[i].image(img)
        with st.container():#
            cols = st.columns(num_cols)
            for i, (mid, rating, num_rating) in enumerate(
                zip(title_df.loc[idx, "MovieID"],
                    title_df.loc[idx, "WeightedRating"],
                    title_df.loc[idx, "# of Ratings"])):
                if get_input:
                    if "click-" + mid not in st.session_state:
                        st.session_state["click-" + mid] = False
                    emt[i] = cols[i].empty()
                    btn = emt[i].button("Rate", key="btn-" + mid)
                    if btn:
                        st.session_state["click-" + mid] = True
                    if st.session_state["click-" + mid]:
                        emt[i].slider(
                        label="My rating", min_value=1, max_value=5,
                        step=1, key="rate-" + mid,
                        value=3,
                        disabled=False
                    )
                else:
                    cols[i].slider(
                        label="Avg. Rating", min_value=1., max_value=5.,
                        step=1., key=mid,
                        value=round(rating, 2),
                        disabled=True
                    )
                rating_str = (
                              f"<center>{rating:.2f} / 5"
                              f"<br>({num_rating} reviews)</center>")
                cols[i].markdown(rating_str, unsafe_allow_html=True)
        st.write("\n\n\n") # spacer
        start_idx += num_cols

def recommend_by_genre():
    st.title("Recommender by Genre")
    genre = st.selectbox("**Select your favorite Genre**:", genres)
    st.divider()

    # Find top titles for selected genre
    top_titles = sysI_recs[sysI_recs["Genre"] == genre].reset_index()
    img_urls = top_titles.loc[:, "MovieID"].apply(lambda x: get_img(x))
    top_titles = top_titles.assign(Image = img_urls)
    top_titles = top_titles[
        ["MovieID", "Title", "WeightedRating", "# of Ratings", "Image"]]

    if st.button("See Recommendations"):
        st.header("Recommendations")
        st.write() # spacer
        show_movies(top_titles)

def recommend_by_rating():
    st.title("Recommender by Rating")
    st.info("Please rate these movies to help find recommendations.")
    st.divider()
    
    show_movies(init_df, get_input=True)
    st.divider()

    submit = st.button("Recommend some new movies")

    if submit:
        # Find which movies were rated
        rated_keys = [x for x in list(st.session_state) if "rate" in x]
        rated_mid = []
        rated_val = []
        for key in rated_keys:
            rated_mid.append(key.split("-")[1])
            rated_val.append(st.session_state[key])

        # Construct rating profile
        new_user = s.iloc[0, :].copy(deep=True)
        new_user.iloc[:] = np.nan
        new_user.loc[rated_mid] = rated_val

        pred_mid = myIBCF(s, new_user, mov_rate_genre, sysI_recs)
        pred_df = make_rating_df(mov_rate_genre, pred_mid)
        
        st.header("Recommendations")
        show_movies(pred_df)


def main():
    st.set_page_config(page_title="Movie Recommender", page_icon="🎥",
                       layout="wide")
    system = ""
    with st.sidebar:
        system = st.radio("Select a Page",
                          ["Recommender by Genre", "Recommender by Rating"])

    if system == "Recommender by Genre":
        recommend_by_genre()
    elif system == "Recommender by Rating":
        recommend_by_rating()

base_url = 'https://raw.githubusercontent.com/RynoXLI/PSL/main/project4/'
recs_file = 'sysI_recs.csv'
sim_file = 'similarity.csv'
mrg_file = "movie_ratings_genre.csv"
init_file = "initial_suggestions.txt"

sysI_recs = pd.read_csv(base_url + recs_file)
s = pd.read_csv(base_url + sim_file, index_col=0)
mov_rate_genre = pd.read_csv(base_url + mrg_file, index_col=0,
                             converters={"Genres": pd.eval})

# Initial title suggestions
init_url = request.urlopen(base_url + init_file).readlines()
mid_suggs = list(map(lambda x: x.decode("utf-8").strip(), init_url))
init_df = make_rating_df(mov_rate_genre, mid_suggs)

genres = sorted(sysI_recs["Genre"].unique().tolist())

main()