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
import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

sysI_recs = pd.read_csv('https://raw.githubusercontent.com/RynoXLI/PSL/rf-proj4-deploy/project4/sysI_recs.csv')
genres = sorted(sysI_recs['Genre'].unique().tolist())

system = 'System I'
with st.sidebar:
    system = st.radio('Select a System', ['System I', 'System II'])


if system == 'System I':
    st.title('System I: Recommendation Based on Genres')

    genre = st.selectbox('Select your favorite Genre:', genres)

    titles = sysI_recs[sysI_recs['Genre'] == genre].reset_index().drop(columns=['MovieID', 'Genre', 'index'])
    st.table(titles)
elif system == 'System II':
    st.title('System II: Recommendation Based on IBCF')
