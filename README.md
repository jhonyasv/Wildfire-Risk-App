# Wildfire Risk Prediction Thesis Web App 
This repository contains the source code for the **Wildfire Risk Prediction Thesis Web App**. This project is part of the Master of Science in Geospatial Technologies thesis by **Jhony Alexander Sánchez Vargas** at the [University of Münster](https://www.uni-muenster.de) (Institute for Geoinformatics). The app integrates **remote sensing data** and **deep learning models** (RF, LSTM, CNN, ConvLSTM) to predict wildfire risk, with a focus on South American regions. 
## Table of Contents 
- [Overview](#overview) 
- [Features](#features) 
- [Prerequisites](#prerequisites) 
- [Installation](#installation) 
- [Usage](#usage) 
- [Project Structure](#project-structure) 
- [Git LFS](#git-lfs) 
- [Contributing](#contributing) 
- [License](#license) 
- [Acknowledgements](#acknowledgements) 
## Overview 
Wildfires are a growing threat to ecosystems, communities, and infrastructure worldwide. This thesis project develops a robust **wildfire risk prediction system** leveraging: 
- **Remote Sensing:** Utilizes Earth Observation data cubes from [Google Earth Engine](https://earthengine.google.com). 
- **Deep Learning Models:** Implements various models including **Random Forest (RF)**, **Long Short-Term Memory (LSTM)**, **Convolutional Neural Networks (CNN)**, and **Convolutional LSTM (ConvLSTM)** for wildfire risk forecasting. 
- **Interactive Web App:** Built with [Streamlit](https://streamlit.io) and [leafmap](https://leafmap.org) for geospatial visualization and analysis.
 The app provides a user-friendly interface to **select dates**, **draw a region of interest (ROI)** on an interactive map, **choose the desired model**, and **visualize wildfire risk predictions**. 
 ## Features 
 - **Interactive Map:** Explore geospatial data using an embedded interactive map with [leafmap](https://leafmap.org). 
 - **Model Selector:** Choose among **RF**, **LSTM**, and **CNN**-based models (Available LSTM only due to hosting issues). 
 - **Date Input:** Limits date selection to the most recent available **MODIS** data. 
 - **Wildfire Risk Prediction:** Visualizes predictions on a geospatial map. 
 - **Thesis Documentation:** Serves as a companion tool for the thesis project **"Development of a wildfire risk prediction system based on Deep Learning methods and Remote Sensing."** 
 ## Prerequisites 
 - Python 3.8 or higher - [Streamlit](https://streamlit.io/) 
 - [leafmap](https://leafmap.org) 
 - [xarray](https://xarray.pydata.org) 
 - [Cartopy](https://scitools.org.uk/cartopy) 
 - **TensorFlow**  
 - Other dependencies as listed in `requirements.txt` 
 ## Installation 
 1. **Clone the repository:** 
 ```bash git clone https://github.com/yourusername/Wildfire-Risk-App.git cd Wildfire-Risk-App ``` 
 2. **Create a virtual environment (optional but recommended):** 
 ```bash python -m venv venv source venv/bin/activate # On Windows: venv\Scripts\activate ``` 
 3. **Install required packages:**
  ```bash pip install -r requirements.txt ``` 
  4. **Git Large File Storage (LFS):**
   If your models exceed **GitHub**'s file size limit, use [Git LFS](https://git-lfs.github.com) to manage them. 
   ```bash git lfs install git lfs track "models/*.pkl" git add .gitattributes git commit -m "Track large model files with Git LFS" ``` 
   ## Usage 
   1. **Run the app:** 
   ```bash streamlit run Home.py ``` 
   2. **Navigate the app:** 
   - Use the sidebar to review thesis details. 
   - On the main page, read the abstract and instructions. 
   - Interact with the embedded map, select a date, and choose a model. 
   - Draw a polygon or rectangle on the map to define your region of interest. 
   - Click on the "Predict Wildfire Risk" button to run the prediction model. 
   - The app will display the predicted wildfire risk map based on your selection. 
   ## Project Structure 
   ``` Wildfire-Risk-App/```
   ```├── app.py # Main Streamlit app entry point ```
   ```├── pages/ # Additional pages for different aspects of the project ```
   ```│ ├── WildfireRisk.py # Wildfire risk prediction page ```
   ```│ └── ... # Other related pages ```
   ```├── models/ # Folder containing saved models (tracked via Git LFS) ```
   ```├── README.md # This file ```
   ```├── requirements.txt # List of Python dependencies ```
   ```└── assets/ # Images and additional resources```
   ## Git LFS 
   Since some model files exceed 100 MB, **Git Large File Storage (LFS)** is used to manage these files. If you encounter issues when pushing large files, please refer to the [Git LFS documentation](https://git-lfs.github.com). 
   ## Contributing 
   Contributions to this project are welcome! Please open an issue or submit a pull request for any improvements or bug fixes. 
   ## License 
   This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details. 
   ## Acknowledgements 
   **Thesis Supervisors:** 
   - Prof. Dr. Mana Gharun 
   - Johannes Heisig
   - Prof. Dr. Marco Painho.

   **University of Münster, Institute for Geoinformatics** 

   Thanks to the open-source community for tools such as [Streamlit](https://streamlit.io) and [leafmap](https://leafmap.org). 