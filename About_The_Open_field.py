import streamlit as st
import pandas as pd
import numpy as np
import plotly_express as px
import json

st.sidebar.image("data/logo_mlit.jpg")
st.sidebar.info(
    """
    This app is Open Source dashboard.
    """
)
st.sidebar.info("Site of MLIT JINR: "
                "[Link](https://lit.jinr.ru/).")
st.sidebar.info("The project is being created within the framework of the ML/DL/HPC ecosystem of the HybriLIT platform. Link: "
                "[here](https://hlit.jinr.ru).")
st.sidebar.image("data/logo_hlit.png")
st.sidebar.header("Dashboard")

st.markdown("# About The Open field test-system")

st.markdown("""
The open field test is a generally accepted paradigm for measurement of explorative and locomotor behavior in animals. 
It allows the responses to new and unfamiliar environments as well as habituation to the environment to be evaluated. 
Rats and mice tend to avoid brightly illuminated open spaces, so the open field environment acts as an anxiogenic stimulus and allows the measurement of anxiety-induced behaviors.
"""
)

st.markdown("""
The considered behavioral test has a form of round arena with the chequered-marked sectors and holes. 
The observation procedure on the laboratory animals takes 3-6 minutes. 

The “Open Field” test-system allows to register the general activity of animals. 
To this aim, we fix the quantity of passed sectors together with the number of intersections of the marked center. 
Also, we check how many holes, standings upright with/without supports, standings still and motions on one place the animals did. 
""")
            
st.markdown("""
The module of the information system is necessary to solve the following tasks:
""")    
st.markdown("""
1. Improving the detection of rodents in test installations;
2. Automation of recognition of individual behavioral acts (grooming, racks, etc.) and their registration;
3. Storage of video material of the conducted experiments.
""")

st.image("data/field.png")