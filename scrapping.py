import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import streamlit as st
import pandas as pd
from io import StringIO
import json

@st.cache_data
def scrap_city():
    df = pd.read_csv('noida_properties.csv')
    area_counts_df = df['area'].value_counts().reset_index()
    area_counts_df.columns = ['Area', 'ProjectCount']
    return area_counts_df



def sub_scrap(suburb):
    df = pd.read_csv('noida_properties.csv')
    df=df[df['area']==suburb]
    return df
    

    