import streamlit as st
import leafmap.foliumap as leafmap

# Set the page configuration: wide layout with a custom page title and icon
st.set_page_config(layout="wide", page_title="Wildfire Risk Thesis", page_icon="🔥")

# Customize the sidebar with thesis details
sidebar_info = """
# Wildfire Risk Prediction Thesis
**Title:** Development of a wildfire risk prediction system based on Deep Learning methods and Remote Sensing  
**Author:** Jhony Alexander Sánchez Vargas  
**Institution:** University of Münster, Institute for Geoinformatics  
**Degree:** Master of Science in Geospatial Technologies  
**Semester:** Winter Semester 2025  
"""

st.sidebar.title("Thesis Overview")
st.sidebar.info(sidebar_info)
logo = "https://source.unsplash.com/featured/?wildfire"
st.sidebar.image(logo)

# Main page title and introduction
st.title("Wildfire Risk Prediction Thesis")
st.markdown(
    """
    ## Abstract  
    Wildfires pose a significant threat to ecosystems, human life, and infrastructure. This thesis develops a wildfire risk prediction system leveraging deep learning methods and remote sensing data. The objectives include:
    - **Exploring** how Earth Observation data cubes can enhance spatiotemporal analysis,
    - **Assessing** the suitability of various deep learning algorithms (RF, LSTM, CNN, ConvLSTM) for wildfire risk prediction,
    - **Integrating** static and dynamic variables to improve real-time risk assessments.
    
    **Key Findings:**
    - LSTM achieved an AUROC of 89% with an F1-score of 72%, balancing temporal dependencies and classification trade-offs.
    - CNN exhibited the highest sensitivity (84%) but with lower precision.
    - RF reached the highest AUROC (91%) yet faced challenges with sensitivity.
    - Feature importance analysis highlighted human modification, land surface temperature, and temperature as critical predictors.
    
    This work contributes to advancing scalable, data-driven models for wildfire prediction and offers valuable insights for enhancing fire management strategies in South America.
    """
)

st.header("Thesis Document Structure")
st.markdown(
    """
    1. **Introduction:** Importance of forest management and wildfire risk.
    2. **Literature Review:** Analysis of remote sensing, deep learning methods, and existing wildfire monitoring platforms.
    3. **Methodology:** Data sources, experimental design, and model architectures (RF, LSTM, CNN, ConvLSTM).
    4. **Results and Discussion:** Evaluation of model performance and implications for wildfire management.
    5. **Conclusion:** Summary of findings and recommendations for future research.
    """
)