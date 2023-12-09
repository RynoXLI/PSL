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
import numpy as np
import streamlit as st
import pandas as pd


base_url = 'https://raw.githubusercontent.com/RynoXLI/PSL/main/project4/'
recs_file = 'sysI_recs.csv'
sim_file = 'similarity.csv'
mrg_file = "movie_ratings_genre.csv"

st.set_page_config(page_title="Movie Recommender", page_icon="🎥", layout="wide")

sysI_recs = pd.read_csv(base_url + recs_file)
s = pd.read_csv(base_url + sim_file, index_col=0)
mov_rate_genre = pd.read_csv(base_url + mrg_file, index_col=0)

sysI_recs_full = pd.read_csv(
    "https://raw.githubusercontent.com/RynoXLI/PSL/main/project4/sysI_recs_full.csv"
)

genres = sorted(sysI_recs["Genre"].unique().tolist())
movies = sysI_recs_full["Title"].unique().tolist()
title_to_mid = dict(zip(sysI_recs_full["Title"], sysI_recs_full["MovieID"]))
mid_to_title = dict(zip(sysI_recs_full["MovieID"], sysI_recs_full["Title"]))


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


system = "System I"
with st.sidebar:
    system = st.radio(
        "Select a Page", ["Recommender by Genre", "Recommender by Rating"]
    )


if system == "Recommender by Genre":
    st.title("Recommendations Based on Genres")

    genre = st.selectbox("**Select your favorite Genre**:", genres)

    titles = sysI_recs[sysI_recs["Genre"] == genre].reset_index()
    titles["Image"] = titles["MovieID"].apply(
        lambda x: f"https://liangfgithub.github.io/MovieImages/{x}.jpg?raw=true"
    )
    titles = titles.drop(columns=["MovieID", "Genre", "index"])[
        ["Image", "Title", "WeightedRating", "AverageRating", "# of Ratings"]
    ]

    st.divider()
    if st.button("See Recommendations"):
        st.header("Recommendations")

        cols1 = st.columns(5)
        st.write()
        cols2 = st.columns(5)
        for i, x in enumerate(titles.itertuples()):
            if i < 5:
                with cols1[i]:
                    st.write(f"**{x[2]}**")
                    st.image(x[1])
                    st.metric("Weighted Rating", f"{x[3]:.2f}")
            else:
                with cols2[i - 5]:
                    st.write(f"**{x[2]}**")
                    st.image(x[1])
                    st.metric("Weighted Rating", f"{x[3]:.2f}")
elif system == "Recommender by Rating":
    st.title("Recommendations based by Rating")
    st.info("Hover over the dataframe and click the (+) sign to start adding"
            " new movies with their respective ratings.")
    df = pd.DataFrame([], columns=["Movie", "Rating"])
    ratings = st.data_editor(
        df,
        column_config={
            "Movie": st.column_config.SelectboxColumn(
                "Movie",
                help="Title of the Movie to Rate",
                options=movies,
                required=True,
            ),
            "Rating": st.column_config.NumberColumn(
                "Rating (1-5)",
                help="The Rating of the Movie",
                min_value=0,
                max_value=5,
                step=0.1,
                required=True,
            ),
        },
        use_container_width=True,
        num_rows="dynamic",
        hide_index=True,
    )
    st.divider()
    submit = st.button("Find Top Movie Recommendations")

    if submit and ratings.shape[0] > 0:
        if ratings["Movie"].nunique() != ratings.shape[0]:
            st.warning("Please remove duplicate rating entries. Select the row"
                       " and hit the delete key on your keyboard.")
        else:
            st.header("Recommendations")
            new_user = s.iloc[0, :].copy(deep=True)
            new_user.iloc[:] = np.nan

            mids = [f"m{title_to_mid[title] }" for title in ratings["Movie"]]
            ratings = ratings["Rating"].tolist()
            new_user.loc[mids] = ratings
            ibcf_ratings = myIBCF(s, new_user, mov_rate_genre, sysI_recs)

            data = [(mid_to_title[int(mid[1:])], mid) for mid in ibcf_ratings]
            data_df = pd.DataFrame(data, columns=["Title", "MovieID"])
            data_df["Image"] = data_df["MovieID"].apply(
                lambda x: f"https://liangfgithub.github.io/MovieImages/{x[1:]}.jpg?raw=true"
            )

            cols1 = st.columns(5)
            st.write()
            cols2 = st.columns(5)
            for i, x in enumerate(data_df[["Image", "Title", "MovieID"]].itertuples()):
                if i < 5:
                    with cols1[i]:
                        st.write(f"**{x[2]}**")
                        st.image(x[1])
                        st.metric("Rank", i + 1)
                else:
                    with cols2[i - 5]:
                        st.write(f"**{x[2]}**")
                        st.image(x[1])
                        st.metric("Rank", i + 1)
    elif submit and ratings.shape[0] == 0:
        st.header("Recommendations")
        rankings = (
            sysI_recs_full.sort_values(by="WeightedRating", ascending=False)[
                ["MovieID", "Title"]
            ]
            .drop_duplicates()
            .head(10)
        )
        rankings["Image"] = rankings["MovieID"].apply(
            lambda x: f"https://liangfgithub.github.io/MovieImages/{x}.jpg?raw=true"
        )

        cols1 = st.columns(5)
        st.write()
        cols2 = st.columns(5)
        for i, x in enumerate(rankings[["Image", "Title", "MovieID"]].itertuples()):
            if i < 5:
                with cols1[i]:
                    st.write(f"**{x[2]}**")
                    st.image(x[1])
                    st.metric("Rank", i + 1)
            else:
                with cols2[i - 5]:
                    st.write(f"**{x[2]}**")
                    st.image(x[1])
                    st.metric("Rank", i + 1)
