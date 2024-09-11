import h5py
import numpy as np

def map_geox_geoy_to_lat_lon(geo_x, geo_y, upper_lat, lower_lat, left_lon, right_lon):
    """
    Maps GeoX and GeoY pixel coordinates to latitude and longitude based on bounding box coordinates.
    
    Parameters:
    - geo_x: Array of GeoX values (pixel indices in the x-direction)
    - geo_y: Array of GeoY values (pixel indices in the y-direction)
    - upper_lat: Upper latitude of the image (degrees)
    - lower_lat: Lower latitude of the image (degrees)
    - left_lon: Left longitude of the image (degrees)
    - right_lon: Right longitude of the image (degrees)
    
    Returns:
    - latitudes: Array of mapped latitudes
    - longitudes: Array of mapped longitudes
    """
    # Normalize GeoX and GeoY to range [0, 1]
    geo_x_norm = geo_x / geo_x.max()
    geo_y_norm = geo_y / geo_y.max()
    
    # Map to latitudes and longitudes using linear interpolation
    latitudes = lower_lat + (upper_lat - lower_lat) * (1 - geo_y_norm)  # Inverted y-axis
    longitudes = left_lon + (right_lon - left_lon) * geo_x_norm
    
    return latitudes, longitudes

def extract_and_map_geocoordinates(file_path):
    """
    Extracts GeoX, GeoY, and IMG_MIR_TEMP datasets from the HDF5 file and maps GeoX, GeoY to lat/lon.
    
    Parameters:
    - file_path: Path to the HDF5 file
    
    Returns:
    - latitudes: Mapped latitude array
    - longitudes: Mapped longitude array
    - img_mir_temp: Numpy array of IMG_MIR_TEMP values (MIR Temperature)
    """
    with h5py.File(file_path, 'r') as hdf_file:
        # Extract GeoX, GeoY, and IMG_MIR_TEMP datasets
        geo_x = hdf_file['GeoX'][:]
        geo_y = hdf_file['GeoY'][:]
        img_mir_temp = hdf_file['IMG_MIR_TEMP'][:]
        
        # Bounding box information from metadata
        upper_lat = 81.04153
        lower_lat = -81.04153
        left_lon = 0.8432964
        right_lon = 163.15671
        
        # Map GeoX and GeoY to latitudes and longitudes
        latitudes, longitudes = map_geox_geoy_to_lat_lon(geo_x, geo_y, upper_lat, lower_lat, left_lon, right_lon)
        
        # Print the results
        print(f"GeoX Shape: {geo_x.shape}, GeoY Shape: {geo_y.shape}")
        print(f"Latitude Sample: {latitudes[:5]}")
        print(f"Longitude Sample: {longitudes[:5]}")
        print(f"IMG_MIR_TEMP Sample: {img_mir_temp[:5]}")
        
    return latitudes, longitudes, img_mir_temp

# Example usage
hdf5_file = r"C:\Users\Hii\Downloads\3DIMG_18JUN2024_0000_L1B_STD_V01R00 (1).h5"
latitudes, longitudes, img_mir_temp = extract_and_map_geocoordinates(hdf5_file)





#CRS MISMATCHED 
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import numpy as np

# Sample data to demonstrate - replace these with your actual data
latitudes = np.random.uniform(low=0, high=90, size=(2816, 2805))  # Replace with actual latitude array
longitudes = np.random.uniform(low=0, high=180, size=(2816, 2805))  # Replace with actual longitude array
img_mir_temp_resampled = np.random.uniform(low=300, high=350, size=(2816, 2805))  # Replace with actual resampled temperature data

# Create GeoDataFrame from satellite data
def create_geodataframe(latitudes, longitudes, img_mir_temp_resampled):
    # Flatten arrays to create a DataFrame
    data = pd.DataFrame({
        'Latitude': latitudes.flatten(),
        'Longitude': longitudes.flatten(),
        'Temperature': img_mir_temp_resampled.flatten()
    })

    # Create Point geometries from Latitude and Longitude
    geometry = [Point(xy) for xy in zip(data['Longitude'], data['Latitude'])]
    
    # Create GeoDataFrame with geometry
    gdf = gpd.GeoDataFrame(data, geometry=geometry)
    
    # Assign a CRS (assuming EPSG:4326 for geographic coordinates)
    gdf.set_crs(epsg=4326, inplace=True)
    
    return gdf

# Create the satellite GeoDataFrame
satellite_gdf = create_geodataframe(latitudes, longitudes, img_mir_temp_resampled)

# Load Indian State Boundaries Shapefile
states_gdf = gpd.read_file(r'C:\Users\Hii\Downloads\Indian States Shape file\Indian States dataset\output\Gujarat_boundary.shp')

# Check and print CRS of both datasets
print(f"Satellite Data CRS: {satellite_gdf.crs}")
print(f"Indian States CRS: {states_gdf.crs}")

# Ensure both GeoDataFrames are in the same CRS
# Reproject satellite_gdf to match the states_gdf CRS (if needed)
if satellite_gdf.crs != states_gdf.crs:
    satellite_gdf = satellite_gdf.to_crs(states_gdf.crs)

# Perform the intersection between satellite data and Indian state boundaries
intersection_gdf = gpd.overlay(satellite_gdf, states_gdf, how='intersection')

# Display the result
print(intersection_gdf.head())

# Optionally, you can save the intersection result to a new Shapefile
intersection_gdf.to_file("satellite_state_intersection.shp")


#INTERSECTION DONE
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point

def perform_intersection(satellite_gdf, states_shapefile):
    # Load the Indian states shapefile
    states_gdf = gpd.read_file(r'C:\Users\Hii\Downloads\Indian States Shape file\Indian States dataset\output\Gujarat_boundary.shp')

    # Ensure the CRS of both GeoDataFrames are the same
    if satellite_gdf.crs != states_gdf.crs:
        print(f"Transforming satellite data from CRS {satellite_gdf.crs} to CRS {states_gdf.crs}")
        satellite_gdf = satellite_gdf.to_crs(states_gdf.crs)
    
    # Perform intersection
    intersection = gpd.overlay(satellite_gdf, states_gdf, how='intersection')
    
    return intersection

def main():
    # Create a GeoDataFrame for the satellite data
    # Ensure the CRS is EPSG:4326 for the input data
    satellite_data = {
        'Latitude': [21.923394, 22.692675, 22.242117, 22.792990, 20.925572],
        'Longitude': [70.656187, 70.360145, 71.485200, 72.223051, 72.952940],
        'Temperature': [306.002318, 319.531791, 334.673177, 337.521895, 348.992169],
        'State_Name': ['Gujarat'] * 5,
    }
    satellite_df = pd.DataFrame(satellite_data)
    satellite_gdf = gpd.GeoDataFrame(
        satellite_df,
        geometry=gpd.points_from_xy(satellite_df.Longitude, satellite_df.Latitude),
        crs='EPSG:4326'
    )
    
    # Perform intersection
    intersection_gdf = perform_intersection(satellite_gdf, 'C:/Users/Hii/Downloads/Indian States Shape file/Indian States dataset/output/Gujarat_boundary.shp')
    
    # Save or plot the results
    intersection_gdf.to_file('intersection_output.geojson', driver='GeoJSON')
    intersection_gdf.plot()
    plt.show()

if __name__ == "__main__":
    main()



