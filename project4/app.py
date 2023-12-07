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

st.set_page_config(page_title="Movie Recommender",
                   page_icon="🎥",
                   layout="wide")

sysI_recs = pd.read_csv('https://raw.githubusercontent.com/RynoXLI/PSL/main/project4/sysI_recs.csv')
s = pd.read_csv('https://raw.githubusercontent.com/RynoXLI/PSL/rf-proj4-p2/project4/similarity.csv', index_col=0)
sysI_recs_full = pd.read_csv('https://raw.githubusercontent.com/RynoXLI/PSL/rf-proj4-p2/project4/sysI_recs_full.csv')
genres = sorted(sysI_recs['Genre'].unique().tolist())
movies = sysI_recs_full['Title'].unique().tolist()
title_to_mid = dict(zip(sysI_recs_full['Title'], sysI_recs_full['MovieID']))
mid_to_title = dict(zip(sysI_recs_full['MovieID'], sysI_recs_full['Title']))

@st.cache_data
def myIBCF(s, newuser, sysI, num_recs=10):

    recs = newuser.copy(deep=True)
    recs.iloc[:] = np.nan

    i_in_w = ~np.isnan(newuser)
    # Compute IBCF for non-rated movies
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

    # Create mappings needed for ranking
    mid_to_rating = dict(zip(sysI['MovieID'], sysI['WeightedRating']))
    mid_to_genre = dict(zip(sysI['MovieID'], sysI['Genre']))
    #print(f"# ratings: {np.count_nonzero(~np.isnan(newuser))}")
    #print(f"   # recs: {recs.shape[0]}")
    if recs.shape[0] >= num_recs:
        #print(recs.iloc[recs.argsort().iloc[-num_recs:]])
        rec_df = recs.iloc[recs.argsort().iloc[-num_recs:]]

        # Find (mid, IBCF value, Weighted Rating from System I) pairs
        recnames = [(mid, val, mid_to_rating[int(mid[1:])]) for mid, val in zip(rec_df.index, rec_df.values)]

        # Sort by (IBCF value, Weighted rating from System I, then mid) descending
        recs = [x[0] for x in sorted(recnames, key=lambda x: (x[1], x[2], int(x[0][1:])))][::-1]
        return recs
    else:
        additional_recs = num_recs - recs.shape[0]

        # Run through regular logic
        rec_df = recs.iloc[recs.argsort().iloc[-num_recs:]]
        mids = [int(mid[1:]) for mid in rec_df.index]
        recnames = [(mid, val, mid_to_rating[int(mid[1:])]) for mid, val in zip(rec_df.index, rec_df.values)]
        recs = [x[0] for x in sorted(recnames, key=lambda x: (x[1], x[2], int(x[0][1:])))][::-1]

        # From the movies rated by the user, find the most watched genre and return top recommendations from it
        # If there is a tie for most watched genre, then both are considered. 
        # Select the top movies by WeightedRating from System I for the given top genre(s). 
        # Make sure that the movies from the genre are not the same movies the user rated and also not already included from the IBCF recommendations.
        rated_movies = newuser[~np.isnan(newuser)]
        genre_mids = [int(movie[1:]) for movie in rated_movies.index]
        genres = np.unique([mid_to_genre[mid] for mid in genre_mids])
        mids.extend(genre_mids)
        movie_ids = sysI[sysI['Genre'].isin(genres) & ~sysI['MovieID'].isin(mids)].sort_values(by=['WeightedRating', 'MovieID'], ascending=[False, True])[:additional_recs]['MovieID']
        movie_ids = [f'm{mid}' for mid in movie_ids.values]

        return recs + movie_ids

system = 'System I'
with st.sidebar:
    system = st.radio('Select a Page', ['Recommender by Genre', 'Recommender by Rating'])


if system == 'Recommender by Genre':
    st.title('Recommendations Based on Genres')

    genre = st.selectbox('**Select your favorite Genre**:', genres)

    titles = sysI_recs[sysI_recs['Genre'] == genre].reset_index().drop(columns=['MovieID', 'Genre', 'index'])

    st.divider()
    if st.button('See Recommendations'):
        st.header('Recommendations')


        st.dataframe(titles, hide_index=True, use_container_width=True)
elif system == 'Recommender by Rating':
    st.title('Recommendations based by Rating')
    st.info('Hover over the dataframe and click the (+) sign to start adding new movies with their respective ratings.')
    df = pd.DataFrame([], columns=['Movie', 'Rating'])
    ratings = st.data_editor(
        df,
        column_config={
            "Movie": st.column_config.SelectboxColumn(
                "Movie",
                help="Title of the Movie to Rate",
                options = movies,
                required=True
            ),
            "Rating": st.column_config.NumberColumn(
                "Rating (1-5)",
                help="The Rating of the Movie",
                min_value=0,
                max_value=5,
                step=0.1,
                required=True
            )
        },
        use_container_width=True,
        num_rows="dynamic",
        hide_index=True
    )
    st.divider()
    submit = st.button('Find Top Movie Recommendations')

    if submit and ratings.shape[0] > 0:

        if ratings['Movie'].nunique() != ratings.shape[0]:
            st.warning('Please remove duplicate rating entries. Select the row and hit the delete key on your keyboard.')
        else:
            st.header('Recommendations')
            new_user = s.iloc[0, :].copy(deep=True)
            new_user.iloc[:] = np.nan

            mids = [f"m{title_to_mid[title] }"for title in ratings['Movie']]
            ratings = ratings['Rating'].tolist()
            new_user.loc[mids] = ratings
            ibcf_ratings = myIBCF(s, new_user, sysI_recs_full)

            data = [(mid_to_title[int(mid[1:])], mid) for mid in ibcf_ratings]
            st.dataframe(pd.DataFrame(data, columns=['Title', 'MovieID']), use_container_width=True, hide_index=True)
    elif submit and ratings.shape[0] == 0:
        st.warning('Please complete a movie review before submitting')
