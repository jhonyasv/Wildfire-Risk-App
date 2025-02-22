import geemap
import ee
import xee
import xarray as xr
import keras
import math
import bottleneck as bn
import numpy as np
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
import shapely
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_folium import st_folium
import json
import folium
from geemap import geojson_to_ee, ee_to_geojson
from matplotlib.colors import LinearSegmentedColormap
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

st.write("✅ Google Earth Engine authenticated successfully!")
# Function to convert Xarray DataArray to an RGB image using a color map
def xarray_to_colormap(data_array):
    rgba_image = cmap(data_array)  # Apply colormap directly
    return np.uint8(rgba_image[:, :, :3] * 255)  # Convert to 0-255 RGB format

import streamlit as st
import folium
import cartopy
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
from folium.plugins import Draw
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import zoom
from cartopy.img_transform import warp_array
import cartopy.crs as ccrs
import matplotlib.cm

# Define color palette (Green → Yellow → Orange → Red)
COLORS = ["green", "yellow", "orange", "red"]
cmap = LinearSegmentedColormap.from_list("wildfire_risk", COLORS)

# # Function to overlay Xarray wildfire risk prediction on Folium Map
# def overlay_xarray_on_map(data_array):
#     # Detect latitude & longitude dimension names
#     lat_dim = [dim for dim in data_array.dims if "lat" in dim.lower()][0]
#     lon_dim = [dim for dim in data_array.dims if "lon" in dim.lower()][0]

#     # Extract latitude and longitude bounds
#     lat_min, lat_max = data_array[lat_dim].min().item(), data_array[lat_dim].max().item()
#     lon_min, lon_max = data_array[lon_dim].min().item(), data_array[lon_dim].max().item()
#     bounds = [[lat_min, lon_min], [lat_max, lon_max]]

#     # Ensure consistent spatial resolution (1km per pixel)
#     lat_res = abs(data_array[lat_dim][1] - data_array[lat_dim][0]).item()  # Latitude resolution in degrees
#     lon_res = abs(data_array[lon_dim][1] - data_array[lon_dim][0]).item()  # Longitude resolution in degrees

#     # Compute the number of pixels based on 1km resolution
#     earth_circumference_km = 40075  # Approximate circumference of Earth in km
#     lat_km_per_degree = earth_circumference_km / 360  # Approx km per lat degree
#     lon_km_per_degree = lat_km_per_degree * np.cos(np.radians((lat_min + lat_max) / 2))  # Adjust for latitude

#     lat_pixels = int((lat_max - lat_min) * lat_km_per_degree / 1)  # Number of pixels needed for 1km resolution
#     lon_pixels = int((lon_max - lon_min) * lon_km_per_degree / 1)  # Number of pixels needed for 1km resolution

#     # Convert Xarray DataArray to a NumPy array
#     array = np.array(data_array)

#     # Resample the image to match the correct aspect ratio
#     resized_array = zoom(array, (lat_pixels / array.shape[0], lon_pixels / array.shape[1]))

#     # Normalize data (0 to 1)
#     resized_array = np.clip(resized_array, 0, 1)  # Ensure values are within range

#     # Apply color mapping
#     rgba_image = cmap(resized_array)[:, :, :3]  # Extract RGB without alpha
#     rgb_image = np.uint8(rgba_image * 255)  # Convert to 8-bit format

#     # Create a Folium map centered on the dataset
#     m = folium.Map(location=[(lat_min + lat_max) / 2, (lon_min + lon_max) / 2], zoom_start=8)

#     # Overlay wildfire risk map on Folium
#     folium.raster_layers.ImageOverlay(
#         image=rgb_image,
#         bounds=bounds,
#         opacity=0.9,
#         interactive=True,
#     ).add_to(m)

#     return st_folium(m, width=700, height=500)

# Function to overlay Xarray wildfire risk prediction on Folium Map
def overlay_xarray_on_map(data_array):
    # Detect latitude & longitude dimension names
    # lat_dim = [dim for dim in data_array.dims if "lat" in dim.lower()][0]
    # lon_dim = [dim for dim in data_array.dims if "lon" in dim.lower()][0]

    # # Extract latitude and longitude bounds
    # lat_min, lat_max = data_array[lat_dim].min().item(), data_array[lat_dim].max().item()
    # lon_min, lon_max = data_array[lon_dim].min().item(), data_array[lon_dim].max().item()
    # bounds = [[lat_min, lon_min], [lat_max, lon_max]]
    # source_extent = [lon_min, lon_max, lat_min, lat_max]
    # # Ensure consistent spatial resolution (1km per pixel)
    # lat_res = abs(data_array[lat_dim][1] - data_array[lat_dim][0]).item()  # Latitude resolution in degrees
    # lon_res = abs(data_array[lon_dim][1] - data_array[lon_dim][0]).item()  # Longitude resolution in degrees

    # # Compute the number of pixels based on 1km resolution
    # earth_circumference_km = 40075  # Approximate circumference of Earth in km
    # lat_km_per_degree = earth_circumference_km / 360  # Approx km per lat degree
    # lon_km_per_degree = lat_km_per_degree * np.cos(np.radians((lat_min + lat_max) / 2))  # Adjust for latitude

    # lat_pixels = int((lat_max - lat_min) * lat_km_per_degree / 1)  # Number of pixels needed for 1km resolution
    # lon_pixels = int((lon_max - lon_min) * lon_km_per_degree / 1)  # Number of pixels needed for 1km resolution

    # Convert Xarray DataArray to a NumPy array
    array = np.flipud(data_array.values.astype(np.float64))
    lon, lat = np.meshgrid(data_array.lon.values.astype(np.float64), data_array.lat.values.astype(np.float64))
    st.write(f"lon: {lon}")
    st.write(f"lat: {lat}")
    # Resample the image to match the correct aspect ratio
    # resized_array = zoom(array, (lat_pixels / array.shape[0], lon_pixels / array.shape[1]))
    riskmap = cmap(array)
    # Normalize data (0 to 1)
    # resized_array = np.clip(resized_array, 0, 1)  # Ensure values are within range

    # # Apply color mapping
    # rgba_image = cmap(resized_array)[:, :, :3]  # Extract RGB without alpha
    # rgb_image = np.uint8(rgba_image * 255)  # Convert to 8-bit format

    # Create a Folium map centered on the dataset
    m = folium.Map(location=[(lat.min() + lat.max()) / 2, (lon.min() + lon.max()) / 2], zoom_start=8)

    # # Overlay wildfire risk map on Folium
    # folium.raster_layers.ImageOverlay(
    #     image=rgb_image,
    #     bounds=bounds,
    #     opacity=0.9,
    #     interactive=True,
    # ).add_to(m)
    folium.raster_layers.ImageOverlay(riskmap,
                        [[lat.min(), lon.min()], [lat.max(), lon.max()]],
                        mercator_project=True,
                        opacity=0.9,interactive=True).add_to(m)
    return st_folium(m, width=700, height=500)

def merge_data(ds_dyn, ds_static):
    """Load and merge NetCDF datasets, ensuring alignment in time."""
    # Extract primary dataset's time range
    time_range = ds_dyn.coords['time']

    # Repeat secondary data along the time dimension
    ds_static = ds_static.reindex(time=time_range, method='nearest')

    # Merge datasets (primary takes priority, secondary fills missing values)
    ds = ds_dyn.combine_first(ds_static)

    return ds

def impute_missing_values(ds):
    """Impute missing values using forward/backward fill and interpolation."""
    ds = ds.interpolate_na(dim="lat", method="nearest", fill_value="extrapolate")  # Interpolate spatially
    ds = ds.interpolate_na(dim="lon", method="nearest", fill_value="extrapolate")
    ds = ds.ffill(dim="time").bfill(dim="time")  # Fill missing values along the time axis
    # ds = ds.interpolate_na(dim="time", method="nearest", fill_value="extrapolate")  # Interpolate in time
    return ds

def normalize_dataset(ds):
  min_max_dict = {'LST_Day_1km': {'min': 7500, 'max': 65535}, 'EVI': {'min': -2000, 'max': 10000}, 'NDVI': {'min': -2000, 'max': 10000}, 'Fpar_500m': {'min': 0, 'max': 255},'Lai_500m':  {'min': 0, 'max': 255},
                'ET':  {'min': 0, 'max': 1000}, 'dewpoint_temperature_2m':  {'min': 200, 'max': 300}, 'surface_pressure':  {'min': 50000, 'max': 103000},
                  'temperature_2m':  {'min': 220, 'max': 310},'u_component_of_wind_10m' :  {'min': -12, 'max': 12}, 'v_component_of_wind_10m' :  {'min': -10, 'max': 10}, 'elevation' : {'min': -500, 'max': 8800},
                'aspect' : {'min': 0, 'max': 360}, 'slope' : {'min': 0, 'max': 90}}
  ds_norm = ds.copy()

  for var in ds.data_vars:
      if var in min_max_dict:  # Check if the variable is in the dictionary
          min_val = min_max_dict[var]['min']
          max_val = min_max_dict[var]['max']

          # Apply Min-Max normalization: (X - min) / (max - min)
          ds_norm[var] = (ds[var] - min_val) / (max_val - min_val)

  return ds_norm

def prepare_data_for_convlstm(ds):
    """
    Prepare data for ConvLSTM model using all bands and full time steps.

    Parameters:
    - ds: xarray.Dataset containing the spatiotemporal data.

    Returns:
    - X: NumPy array of shape (1, time_steps, height, width, channels)
    """

    # Convert dataset to DataArray and stack variables as channels
    da = ds.to_array(dim="variable")  # Shape: (variable, time, lat, lon)
    da = da.transpose("time", "lon", "lat", "variable")  # Shape: (time, lat, lon, channels)

    # Convert to NumPy array
    X = da.values  # Shape: (time_steps, height, width, channels)

    return X

# Function to process input and predict wildfire risk
def predict_fire_risk(date, geojson):
    st.write("Processing data...")

    end_date = ee.Date(str(date))
    # end_date = ee.Date(riskDate).advance(-1, 'day')
    start_date= end_date.advance(-17, 'day')
    # Compute the centroid
    roi = geojson_to_ee(geojson)
    centroid = roi.centroid(maxError=1)

    # Get the latitude of the centroid
    mean_lat = ee.Number(centroid.coordinates().get(1))

    # Compute the scale
    scale = ee.Number(1000).divide(
        ee.Number(111320).multiply(mean_lat.multiply(math.pi).divide(180).cos())
    )
    # Convert scale to Python float (if needed outside GEE)
    scale_value = scale.getInfo()
    area = roi.bounds()
    collections = [ee.ImageCollection("MODIS/061/MOD13A2").select("NDVI", "EVI").filterDate(start_date, end_date),
                ee.ImageCollection('MODIS/061/MOD11A1').select(['LST_Day_1km']).filterDate(start_date, end_date),
                ee.ImageCollection('MODIS/061/MOD15A2H').select(['Lai_500m', 'Fpar_500m']).filterDate(start_date, end_date),
                ee.ImageCollection('MODIS/061/MOD16A2').select(['ET']).filterDate(start_date, end_date),
                ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
                .select(['temperature_2m','total_precipitation_sum','u_component_of_wind_10m','v_component_of_wind_10m','surface_pressure','dewpoint_temperature_2m'])
                .filterDate(start_date, end_date)
                ]
    # Convert the image collection to an xarray dataset
    try:
        ds_dynamic = xr.open_mfdataset(
            collections,
            engine='ee',
            crs='EPSG:4326',
            scale=scale_value,  # Adjust as necessary
            geometry=area
        )
    except Exception as e:
        print(f"Failed to process: {e}")
    #STATIC BANDS
    static_bands = [ee.ImageCollection(ee.Image('CGIAR/SRTM90_V4').set({'system:time_start': ee.Date('2000-01-01').millis()})),
                ee.ImageCollection(ee.Terrain.slope(ee.Image('CGIAR/SRTM90_V4')).set({'system:time_start': ee.Date('2000-01-01').millis()})),
                ee.ImageCollection(ee.Terrain.aspect(ee.Image('CGIAR/SRTM90_V4')).set({'system:time_start': ee.Date('2000-01-01').millis()})),
                ee.ImageCollection(ee.ImageCollection('CSP/HM/GlobalHumanModification').first().set({'system:time_start': ee.Date('2000-01-01').millis()}))]
    # Convert the image collection to an xarray dataset
    try:
        ds_static = xr.open_mfdataset(
            static_bands,
            engine='ee',
            crs='EPSG:4326',
            scale=scale_value,  # Adjust as necessary
            geometry=area
        )
    except Exception as e:
        print(f"Failed to process: {e}")
    xarray_data = merge_data(ds_dynamic, ds_static)
    xarray_data = xarray_data.chunk({'lon': -1})  # Ensure lon is in a single chunk
    xarray_data = xarray_data.chunk({'lat': -1})  # Might also be needed
    xarray_data = impute_missing_values(xarray_data)
    xarray_data = normalize_dataset(xarray_data)
    array_data = prepare_data_for_convlstm(xarray_data)

    fire_risk_map = np.zeros((array_data.shape[1], array_data.shape[2]))

    # Extract spatial dimensions
    T, lon_size, lat_size, V = array_data.shape  # (17, 247, 127, 16)

    # Reshape to (lon_size * lat_size, 17, 16), preserving (lon, lat) order
    batch_data = array_data.transpose(1, 2, 0, 3).reshape(-1, T, V)  # (247*127, 17, 16)

    # Run batch prediction
    predictions = model.predict(batch_data, verbose=0)  # Output shape: (247*127, 2)

    # Extract fire probability (Class 1)
    fire_risk_map = predictions[:, 1].reshape(lon_size, lat_size)  # Reshape to (247, 127)

    # Convert to Xarray with correct lat/lon mapping
    fire_risk_xr = xr.DataArray(
        fire_risk_map,
        coords={"lon": xarray_data.lon, "lat": xarray_data.lat},
        dims=["lon", "lat"]
    )

    return fire_risk_xr

model = load_model('models/XR_LSTM_model.keras')

# Streamlit UI
st.title("🔥 Wildfire Risk Prediction App")

# Date selection
date = st.date_input("Select a Date")

# Initialize session state for polygon & risk map
if "geojson" not in st.session_state:
    st.session_state.geojson = None
if "fire_risk_map" not in st.session_state:
    st.session_state.fire_risk_map = None

# Folium Map for user input
st.subheader("Draw a Polygon on the Map")

# Create an interactive Folium map
m = folium.Map(location=[-10, -60], zoom_start=4)
folium.plugins.Draw(export=True).add_to(m)

# Display the map in Streamlit
map_data = st_folium(m, width=700, height=500)

# Get user-drawn polygon data
if map_data and "last_active_drawing" in map_data:
    st.session_state.geojson = map_data["last_active_drawing"]

# Ensure the polygon persists
if st.session_state.geojson is not None:
    st.write("✅ Polygon Selected!")

# Prediction button
if st.button("Predict Wildfire Risk"):
    if st.session_state.geojson is not None:
        # Run wildfire risk prediction
        st.session_state.fire_risk_map = predict_fire_risk(date, st.session_state.geojson)

# If a wildfire risk map exists, overlay it on the map
if st.session_state.fire_risk_map is not None:
    st.subheader("📍 Wildfire Risk Map Overlay")
    overlay_xarray_on_map(st.session_state.fire_risk_map)