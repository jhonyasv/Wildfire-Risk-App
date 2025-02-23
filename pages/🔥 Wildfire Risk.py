import ee
import xarray as xr
import math
import bottleneck as bn
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_folium import st_folium
import json
import joblib
import folium
from google.oauth2.credentials import Credentials
from geemap import geojson_to_ee
import datetime
from numpy.lib.stride_tricks import sliding_window_view
# Load credentials from local JSON file
with open("credentialsgee.json", "r") as f:
    creds_json = json.load(f)

CLIENT_ID = creds_json["client_id"]
CLIENT_SECRET = creds_json["client_secret"]
REFRESH_TOKEN = creds_json["refresh_token"]

def initialize_gee():
    creds = Credentials(
        None,
        refresh_token=REFRESH_TOKEN,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
    )

    # Initialize Earth Engine
    ee.Initialize(creds)

initialize_gee()
# -------------------------
# Retrieve last available date from MODIS
# -------------------------
collection = ee.ImageCollection('MODIS/061/MOD11A1')
latest_image = collection.sort('system:time_start', False).first()
latest_date_str = ee.Date(latest_image.get('system:time_start')).format("YYYY-MM-dd").getInfo()
max_date = datetime.datetime.strptime(latest_date_str, "%Y-%m-%d").date()

# -------------------------
# Data Preparation Functions
# -------------------------

def prepare_data_time(ds):
    """
    Placeholder for RF-specific data preparation.
    Modify as needed.
    """
    da = ds.to_array(dim="variable")
    da = da.transpose("time", "lon", "lat", "variable")
    return da.values

def prepare_data_mean(ds):
    """
    For CNN: Aggregate over time (e.g., using mean) and return a spatial image.
    Returns array of shape (height, width, channels)
    """
    da = ds.to_array(dim="variable")
    # Aggregate over time; adjust aggregation as needed
    da = da.mean(dim="time")
    da = da.transpose("lon", "lat", "variable")  # (lon, lat, channels)
    return da.values

# Function to remove a 12-pixel border
def remove_border(data, border=12):
    return data.isel(lon=slice(border, -border), lat=slice(border, -border))
# -------------------------
# Other Helper Functions
# -------------------------
def overlay_xarray_on_map(data_array):
    if model_choice == "CNN":
        data_array = remove_border(data_array)
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    data_array.plot(
        x="lon",
        y="lat",
        cmap="RdYlGn_r",
        transform=ccrs.PlateCarree(),
        cbar_kwargs={"label": "Fire Risk Probability"},
        vmin=0,
        vmax=1,
        ax=ax
    )
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAKES, facecolor="white")
    plt.title(f"Predicted Fire Risk Map")
    lon_min = float(data_array.lon.min())
    lon_max = float(data_array.lon.max())
    lat_min = float(data_array.lat.min())
    lat_max = float(data_array.lat.max())
    ax.set_xticks(np.linspace(lon_min, lon_max, 5), crs=ccrs.PlateCarree())
    ax.set_yticks(np.linspace(lat_min, lat_max, 5), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}°"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}°"))
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    st.subheader(f"📍 Wildfire Risk Map with {model_choice}")
    return st.pyplot(fig)

def merge_data(ds_dyn, ds_static):
    time_range = ds_dyn.coords['time']
    ds_static = ds_static.reindex(time=time_range, method='nearest')
    ds = ds_dyn.combine_first(ds_static)
    return ds

def impute_missing_values(ds):
    ds = ds.interpolate_na(dim="lat", method="nearest", fill_value="extrapolate")
    ds = ds.interpolate_na(dim="lon", method="nearest", fill_value="extrapolate")
    ds = ds.ffill(dim="time").bfill(dim="time")
    return ds

def normalize_dataset(ds):
    min_max_dict = {
        'LST_Day_1km': {'min': 7500, 'max': 65535},
        'EVI': {'min': -2000, 'max': 10000},
        'NDVI': {'min': -2000, 'max': 10000},
        'Fpar_500m': {'min': 0, 'max': 255},
        'Lai_500m':  {'min': 0, 'max': 255},
        'ET':  {'min': 0, 'max': 1000},
        'dewpoint_temperature_2m':  {'min': 200, 'max': 300},
        'surface_pressure':  {'min': 50000, 'max': 103000},
        'temperature_2m':  {'min': 220, 'max': 310},
        'u_component_of_wind_10m' :  {'min': -12, 'max': 12},
        'v_component_of_wind_10m' :  {'min': -10, 'max': 10},
        'elevation' : {'min': -500, 'max': 8800},
        'aspect' : {'min': 0, 'max': 360},
        'slope' : {'min': 0, 'max': 90}
    }
    ds_norm = ds.copy()
    for var in ds.data_vars:
        if var in min_max_dict:
            min_val = min_max_dict[var]['min']
            max_val = min_max_dict[var]['max']
            ds_norm[var] = (ds[var] - min_val) / (max_val - min_val)
    return ds_norm

def predict_fire_risk(date, geojson, model_choice, prepare_data):
    st.write("Processing data...")
    end_date = ee.Date(str(date))
    start_date = end_date.advance(-17, 'day')
    roi = geojson_to_ee(geojson)
    centroid = roi.centroid(maxError=1)
    mean_lat = ee.Number(centroid.coordinates().get(1))
    scale = ee.Number(1000).divide(
        ee.Number(111320).multiply(mean_lat.multiply(math.pi).divide(180).cos())
    )
    scale_value = scale.getInfo()
    area = roi.bounds()
    collections = [
        ee.ImageCollection("MODIS/061/MOD13A2").select("NDVI", "EVI").filterDate(start_date, end_date),
        ee.ImageCollection('MODIS/061/MOD11A1').select(['LST_Day_1km']).filterDate(start_date, end_date),
        ee.ImageCollection('MODIS/061/MOD15A2H').select(['Lai_500m', 'Fpar_500m']).filterDate(start_date, end_date),
        ee.ImageCollection('MODIS/061/MOD16A2').select(['ET']).filterDate(start_date, end_date),
        ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
          .select(['temperature_2m','total_precipitation_sum','u_component_of_wind_10m',
                   'v_component_of_wind_10m','surface_pressure','dewpoint_temperature_2m'])
          .filterDate(start_date, end_date)
    ]
    try:
        ds_dynamic = xr.open_mfdataset(
            collections,
            engine='ee',
            crs='EPSG:4326',
            scale=scale_value,
            geometry=area
        )
    except Exception as e:
        st.error(f"Failed to process dynamic data: {e}")
        return
    static_bands = [
        ee.ImageCollection(ee.Image('CGIAR/SRTM90_V4').set({'system:time_start': ee.Date('2000-01-01').millis()})),
        ee.ImageCollection(ee.Terrain.slope(ee.Image('CGIAR/SRTM90_V4')).set({'system:time_start': ee.Date('2000-01-01').millis()})),
        ee.ImageCollection(ee.Terrain.aspect(ee.Image('CGIAR/SRTM90_V4')).set({'system:time_start': ee.Date('2000-01-01').millis()})),
        ee.ImageCollection(ee.ImageCollection('CSP/HM/GlobalHumanModification').first().set({'system:time_start': ee.Date('2000-01-01').millis()}))
    ]
    try:
        ds_static = xr.open_mfdataset(
            static_bands,
            engine='ee',
            crs='EPSG:4326',
            scale=scale_value,
            geometry=area
        )
    except Exception as e:
        st.error(f"Failed to process static data: {e}")
        return

    # Merge, impute, and normalize datasets
    xarray_data = merge_data(ds_dynamic, ds_static)
    xarray_data = xarray_data.chunk({'lon': -1}).chunk({'lat': -1})
    xarray_data = impute_missing_values(xarray_data)
    xarray_data = normalize_dataset(xarray_data)

    if model_choice == "CNN":
        # For CNN, aggregate over time via prepare_data_for_cnn
        array_data = prepare_data(xarray_data)  # Expected shape: (height, width, channels)
        patch_size = 25
        half_size = patch_size // 2

        # Prepare output map (fire risk probabilities)
        fire_risk_map = np.zeros((array_data.shape[0], array_data.shape[1]))

        # Step 1: Extract sliding window patches
        patches = sliding_window_view(array_data, (patch_size, patch_size, array_data.shape[2]))

        # Step 2: Flatten spatial dimensions for batching
        patches = patches.reshape(-1, patch_size, patch_size, array_data.shape[2])

        # Step 3: Perform batch predictions
        batch_size = 512
        num_patches = patches.shape[0]

        for batch_start in range(0, num_patches, batch_size):
            batch_patches = patches[batch_start:batch_start + batch_size]
            if batch_patches.shape[1:] == (patch_size, patch_size, array_data.shape[2]):
                predictions = model.predict(batch_patches, verbose=0)
                # Step 4: Map predictions back to the center pixel of each patch
                for idx, prediction in enumerate(predictions):
                    # Compute row (i) and column (j) based on sliding window grid size
                    grid_width = array_data.shape[1] - patch_size + 1
                    i = (batch_start + idx) // grid_width
                    j = (batch_start + idx) % grid_width
                    fire_risk_map[i + half_size, j + half_size] = float(prediction[1])
        fire_risk_xr = xr.DataArray(
            fire_risk_map,
            coords={"lon": xarray_data.lon, "lat": xarray_data.lat},
            dims=["lon", "lat"]
        )
    elif model_choice == "RF":
        # Prepare data for RF: expected shape (lon, lat, channels)
        array_data = prepare_data(xarray_data)  # e.g., shape: (247, 127, 16)
        # Extract spatial dimensions
        lon_size, lat_size, V = array_data.shape
        # Reshape to (lon_size*lat_size, V) preserving spatial order
        batch_data = array_data.reshape(-1, V)  # Shape: (247*127, 16)
        # Run RF prediction using predict_proba
        predictions = model.predict_proba(batch_data)  # Expected output: (num_samples, 2)
        # Extract fire probability (class 1) and reshape to (lon_size, lat_size)
        fire_risk_map = predictions[:, 1].reshape(lon_size, lat_size)
        fire_risk_xr = xr.DataArray(
            fire_risk_map,
            coords={"lon": xarray_data.lon, "lat": xarray_data.lat},
            dims=["lon", "lat"]
        )


    else:
        # For RF and LSTM, assume data shape: (time_steps, height, width, channels)
        array_data = prepare_data(xarray_data)
        # Extract spatial dimensions
        T, lon_size, lat_size, V = array_data.shape
        batch_data = array_data.transpose(1, 2, 0, 3).reshape(-1, T, V)
        predictions = model.predict(batch_data, verbose=0)
        fire_risk_map = predictions[:, 1].reshape(lon_size, lat_size)
        fire_risk_xr = xr.DataArray(
            fire_risk_map,
            coords={"lon": xarray_data.lon, "lat": xarray_data.lat},
            dims=["lon", "lat"]
        )
    return fire_risk_xr

# -------------------------
# Streamlit UI
# -------------------------
st.title("🔥 Wildfire Risk Prediction App")

# Date selection (limit dates to those available)
date = st.date_input("Select a Date", max_value=max_date)

# Model selection
model_choice = st.selectbox("Select a Model", options=["RF", "LSTM"])

# Load the corresponding model and set the appropriate data-preparation function
if model_choice == "RF":
    model = joblib.load('models/XR_RF_model.pkl')
    prepare_data = prepare_data_mean  # Use RF-specific data preparation
elif model_choice == "CNN":
    model = load_model('models/XR_cnn_model_5.keras')
    prepare_data = prepare_data_mean # Use CNN-specific data preparation
else:
    model = load_model('models/XR_LSTM_model.keras')
    prepare_data = prepare_data_time  # Use LSTM-specific data preparation

# Initialize session state for polygon & risk map
if "geojson" not in st.session_state:
    st.session_state.geojson = None
if "fire_risk_map" not in st.session_state:
    st.session_state.fire_risk_map = None

st.subheader("Draw a rectangle on the Map")
m = folium.Map(location=[-10, -60], zoom_start=4)
folium.plugins.Draw(export=True).add_to(m)
map_data = st_folium(m, width=700, height=500)

if map_data and "last_active_drawing" in map_data:
    st.session_state.geojson = map_data["last_active_drawing"]

if st.session_state.geojson is not None:
    st.write("✅ Polygon Selected!")

if st.button("Predict Wildfire Risk"):
    if st.session_state.geojson is not None:
        # Pass model_choice to use the proper branch
        st.session_state.fire_risk_map = predict_fire_risk(date, st.session_state.geojson, model_choice, prepare_data)

if st.session_state.fire_risk_map is not None:
    st.subheader(f"📍 Wildfire Risk Map with {model_choice}")
    overlay_xarray_on_map(st.session_state.fire_risk_map)