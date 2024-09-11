#FINAL CODE
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

def perform_intersection(satellite_gdf, shapefile_path):
    # Load Indian states shapefile
    states_gdf = gpd.read_file(shapefile_path)
    
    # Ensure CRS is consistent
    if satellite_gdf.crs != states_gdf.crs:
        print(f"Transforming satellite data from CRS {satellite_gdf.crs} to CRS {states_gdf.crs}")
        satellite_gdf = satellite_gdf.to_crs(states_gdf.crs)
    
    # Perform intersection using 'predicate' instead of 'op'
    intersection_gdf = gpd.sjoin(satellite_gdf, states_gdf, how='inner', predicate='intersects')
    
    return intersection_gdf

def plot_intersection(intersection_gdf, states_gdf):
    # Plot state boundaries
    fig, ax = plt.subplots(figsize=(10, 10))
    
    states_gdf.plot(ax=ax, color='lightgrey', edgecolor='black', alpha=0.5, label='Indian States')
    
    # Plot intersection results
    intersection_gdf.plot(ax=ax, color='red', markersize=5, label='Satellite Data')
    
    plt.title('Intersection of Satellite Data with Gujarat')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.show()

def main():
    # Create a GeoDataFrame for the satellite data
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
    shapefile_path = r'C:/Users/Hii/Downloads/Indian States Shape file/Indian States dataset/output/Gujarat_boundary.shp'
    intersection_gdf = perform_intersection(satellite_gdf, shapefile_path)
    
    # Load Indian states shapefile again for plotting
    states_gdf = gpd.read_file(shapefile_path)
    
    # Plot the results
    plot_intersection(intersection_gdf, states_gdf)

if __name__ == "__main__":
    main()


# understnding the file structure of the dataset 
# import h5py

# def inspect_hdf5(file_path):
#     with h5py.File(file_path, 'r') as hdf_file:
#         def print_attrs(name, obj):
#             print(name)
#         hdf_file.visititems(print_attrs)



# Example usage
# import h5py
# import numpy as np

# def extract_hdf5_datasets(file_path):
#     """
#     Extracts Latitude, Longitude, and IMG_MIR_TEMP datasets from the HDF5 file.
    
#     Parameters:
#     - file_path: Path to the HDF5 file
    
#     Returns:
#     - latitude: Numpy array of Latitude values
#     - longitude: Numpy array of Longitude values
#     - img_mir_temp: Numpy array of IMG_MIR_TEMP values (MIR Temperature)
#     """
    
#     with h5py.File(file_path, 'r') as hdf_file:
#         # Extract datasets
#         latitude = hdf_file['Latitude'][:]
#         longitude = hdf_file['Longitude'][:]
#         img_mir_temp = hdf_file['IMG_MIR_TEMP'][:]

#         # Identify any invalid or placeholder values in Latitude and Longitude (e.g., 32767)
#         invalid_value = 32767
#         valid_latitude = np.where(latitude != invalid_value, latitude, np.nan)
#         valid_longitude = np.where(longitude != invalid_value, longitude, np.nan)

#         print(f"Latitude Shape: {latitude.shape}")
#         print(f"Longitude Shape: {longitude.shape}")
#         print(f"IMG_MIR_TEMP Shape: {img_mir_temp.shape}")
        
#     return valid_latitude, valid_longitude, img_mir_temp

# # Example usage
# hdf5_file = r"C:\Users\Hii\Downloads\3DIMG_18JUN2024_0000_L1B_STD_V01R00 (1).h5"
# latitude, longitude, img_mir_temp = extract_hdf5_datasets(hdf5_file)

# # Optional: Print or inspect a few data points from the cleaned datasets
# print(f"Latitude Sample: {latitude[:5, :5]}")  # Assuming valid values exist
# print(f"Longitude Sample: {longitude[:5, :5]}")
# print(f"IMG_MIR_TEMP Sample: {img_mir_temp[:5]}")  # Extract as 1D sample


# import h5py
# import numpy as np

# def extract_hdf5_datasets(file_path):
#     """
#     Extracts Latitude, Longitude, and IMG_MIR_TEMP datasets from the HDF5 file.
    
#     Parameters:
#     - file_path: Path to the HDF5 file
    
#     Returns:
#     - latitude: Numpy array of Latitude values (after invalid value removal)
#     - longitude: Numpy array of Longitude values (after invalid value removal)
#     - img_mir_temp: Numpy array of IMG_MIR_TEMP values (MIR Temperature)
#     """
    
#     with h5py.File(file_path, 'r') as hdf_file:
#         # Extract datasets
#         latitude = hdf_file['Latitude'][:]
#         longitude = hdf_file['Longitude'][:]
#         img_mir_temp = hdf_file['IMG_MIR_TEMP'][:]

#         # Identify any invalid or placeholder values in Latitude and Longitude (e.g., 32767)
#         invalid_value = 32767
#         valid_latitude = np.where(latitude != invalid_value, latitude, np.nan)
#         valid_longitude = np.where(longitude != invalid_value, longitude, np.nan)

#         print(f"Latitude Shape: {latitude.shape}")
#         print(f"Longitude Shape: {longitude.shape}")
#         print(f"IMG_MIR_TEMP Shape: {img_mir_temp.shape}")
        
#     return valid_latitude, valid_longitude, img_mir_temp

# def inspect_attrs(file_path):
#     """
#     Inspects the attributes of Latitude and Longitude datasets to check for scaling factors, offsets, etc.
    
#     Parameters:
#     - file_path: Path to the HDF5 file
#     """
#     with h5py.File(file_path, 'r') as hdf_file:
#         lat_attrs = hdf_file['Latitude'].attrs
#         lon_attrs = hdf_file['Longitude'].attrs

#         print("Latitude Attributes:")
#         for key, value in lat_attrs.items():
#             print(f"{key}: {value}")

#         print("\nLongitude Attributes:")
#         for key, value in lon_attrs.items():
#             print(f"{key}: {value}")

# def inspect_geo_datasets(file_path):
#     """
#     Inspects the alternative GeoX and GeoY datasets, if present, for georeferencing.
    
#     Parameters:
#     - file_path: Path to the HDF5 file
#     """
#     with h5py.File(file_path, 'r') as hdf_file:
#         if 'GeoX' in hdf_file and 'GeoY' in hdf_file:
#             geo_x = hdf_file['GeoX'][:]
#             geo_y = hdf_file['GeoY'][:]

#             print(f"GeoX Shape: {geo_x.shape}, Sample: {geo_x[:5]}")
#             print(f"GeoY Shape: {geo_y.shape}, Sample: {geo_y[:5]}")
#         else:
#             print("GeoX and GeoY datasets not found in the file.")

# # Example usage
# hdf5_file = r"C:\Users\Hii\Downloads\3DIMG_18JUN2024_0000_L1B_STD_V01R00 (1).h5"

# # Step 1: Extract datasets (Latitude, Longitude, IMG_MIR_TEMP)
# latitude, longitude, img_mir_temp = extract_hdf5_datasets(hdf5_file)

# # Step 2: Inspect attributes for scaling factors or offsets
# inspect_attrs(hdf5_file)

# # Step 3: Check for alternative georeferencing datasets (GeoX, GeoY)
# inspect_geo_datasets(hdf5_file)

# # Optional: Print or inspect a few data points from the cleaned datasets
# print(f"Latitude Sample: {latitude[:5, :5]}")  # Assuming valid values exist
# print(f"Longitude Sample: {longitude[:5, :5]}")
# print(f"IMG_MIR_TEMP Sample: {img_mir_temp[:5]}")  # Extract as 1D sample




# import h5py
# import numpy as np

# def extract_alternative_geocoordinates(file_path):
#     """
#     Uses GeoX and GeoY datasets as a substitute for Latitude and Longitude if necessary.
    
#     Parameters:
#     - file_path: Path to the HDF5 file
    
#     Returns:
#     - geo_x: GeoX dataset (potentially for longitude)
#     - geo_y: GeoY dataset (potentially for latitude)
#     - img_mir_temp: Numpy array of IMG_MIR_TEMP values (MIR Temperature)
#     """
#     with h5py.File(file_path, 'r') as hdf_file:
#         # Extract GeoX, GeoY, and IMG_MIR_TEMP datasets
#         geo_x = hdf_file['GeoX'][:]
#         geo_y = hdf_file['GeoY'][:]
#         img_mir_temp = hdf_file['IMG_MIR_TEMP'][:]
        
#         print(f"GeoX Shape: {geo_x.shape}")
#         print(f"GeoY Shape: {geo_y.shape}")
#         print(f"IMG_MIR_TEMP Shape: {img_mir_temp.shape}")

#     return geo_x, geo_y, img_mir_temp

# # Example usage
# hdf5_file = r"C:\Users\Hii\Downloads\3DIMG_18JUN2024_0000_L1B_STD_V01R00 (1).h5"

# # Step 1: Extract GeoX, GeoY, and IMG_MIR_TEMP as a fallback
# geo_x, geo_y, img_mir_temp = extract_alternative_geocoordinates(hdf5_file)

# # Step 2: Inspect the GeoX and GeoY values
# print(f"GeoX Sample: {geo_x[:5]}")
# print(f"GeoY Sample: {geo_y[:5]}")
# print(f"IMG_MIR_TEMP Sample: {img_mir_temp[:5]}")




# import h5py

# def inspect_metadata(file_path):
#     """
#     Inspect the HDF5 file for any geospatial metadata like projection or affine transformation.
    
#     Parameters:
#     - file_path: Path to the HDF5 file
#     """
#     with h5py.File(file_path, 'r') as hdf_file:
#         # Print metadata and attributes of the root group
#         for key, value in hdf_file.attrs.items():
#             print(f"{key}: {value}")

# # Example usage
# hdf5_file =  r"C:\Users\Hii\Downloads\3DIMG_18JUN2024_0000_L1B_STD_V01R00 (1).h5"
# inspect_metadata(hdf5_file)



# -------------------------------------MAIN CODE STARTS FROM HERE---------------------------------------------------




# import h5py
# import numpy as np

# def map_geox_geoy_to_lat_lon(geo_x, geo_y, upper_lat, lower_lat, left_lon, right_lon):
#     """
#     Maps GeoX and GeoY pixel coordinates to latitude and longitude based on bounding box coordinates.
    
#     Parameters:
#     - geo_x: Array of GeoX values (pixel indices in the x-direction)
#     - geo_y: Array of GeoY values (pixel indices in the y-direction)
#     - upper_lat: Upper latitude of the image (degrees)
#     - lower_lat: Lower latitude of the image (degrees)
#     - left_lon: Left longitude of the image (degrees)
#     - right_lon: Right longitude of the image (degrees)
    
#     Returns:
#     - latitudes: Array of mapped latitudes
#     - longitudes: Array of mapped longitudes
#     """
#     # Normalize GeoX and GeoY to range [0, 1]
#     geo_x_norm = geo_x / geo_x.max()
#     geo_y_norm = geo_y / geo_y.max()
    
#     # Map to latitudes and longitudes using linear interpolation
#     latitudes = lower_lat + (upper_lat - lower_lat) * (1 - geo_y_norm)  # Inverted y-axis
#     longitudes = left_lon + (right_lon - left_lon) * geo_x_norm
    
#     return latitudes, longitudes

# def extract_and_map_geocoordinates(file_path):
#     """
#     Extracts GeoX, GeoY, and IMG_MIR_TEMP datasets from the HDF5 file and maps GeoX, GeoY to lat/lon.
    
#     Parameters:
#     - file_path: Path to the HDF5 file
    
#     Returns:
#     - latitudes: Mapped latitude array
#     - longitudes: Mapped longitude array
#     - img_mir_temp: Numpy array of IMG_MIR_TEMP values (MIR Temperature)
#     """
#     with h5py.File(file_path, 'r') as hdf_file:
#         # Extract GeoX, GeoY, and IMG_MIR_TEMP datasets
#         geo_x = hdf_file['GeoX'][:]
#         geo_y = hdf_file['GeoY'][:]
#         img_mir_temp = hdf_file['IMG_MIR_TEMP'][:]
        
#         # Bounding box information from metadata
#         upper_lat = 81.04153
#         lower_lat = -81.04153
#         left_lon = 0.8432964
#         right_lon = 163.15671
        
#         # Map GeoX and GeoY to latitudes and longitudes
#         latitudes, longitudes = map_geox_geoy_to_lat_lon(geo_x, geo_y, upper_lat, lower_lat, left_lon, right_lon)
        
#         # Print the results
#         print(f"GeoX Shape: {geo_x.shape}, GeoY Shape: {geo_y.shape}")
#         print(f"Latitude Sample: {latitudes[:5]}")
#         print(f"Longitude Sample: {longitudes[:5]}")
#         print(f"IMG_MIR_TEMP Sample: {img_mir_temp[:5]}")
        
#     return latitudes, longitudes, img_mir_temp

# # Example usage
# hdf5_file = r"C:\Users\Hii\Downloads\3DIMG_18JUN2024_0000_L1B_STD_V01R00 (1).h5"
# latitudes, longitudes, img_mir_temp = extract_and_map_geocoordinates(hdf5_file)




# --------------------------INTERSERSECTION OF THE DATA WITH THE SHAPEFILE-----------------------------------------



# import h5py
# import numpy as np
# import pandas as pd
# import geopandas as gpd
# from scipy.interpolate import griddata
# from shapely.geometry import Point

# # Function to extract latitude, longitude, and MIR temperature data
# def extract_data(hdf5_file):
#     with h5py.File(hdf5_file, 'r') as hdf_file:
#         latitudes = hdf_file['Latitude'][:]
#         longitudes = hdf_file['Longitude'][:]
#         img_mir_temp = hdf_file['IMG_MIR_TEMP'][:]  # Assuming this is the 1D data

#         # Handle _FillValue for latitude and longitude (replace with NaN)
#         lat_fill_value = hdf_file['Latitude'].attrs['_FillValue'][0]
#         lon_fill_value = hdf_file['Longitude'].attrs['_FillValue'][0]
        
#         latitudes = np.where(latitudes == lat_fill_value, np.nan, latitudes)
#         longitudes = np.where(longitudes == lon_fill_value, np.nan, longitudes)

#         # Scale latitudes and longitudes (apply scale factor if needed)
#         latitudes = latitudes * 0.01  # Scale by 0.01 degrees
#         longitudes = longitudes * 0.01

#     return latitudes, longitudes, img_mir_temp

# # Function to interpolate the 1D MIR temperature data to a 2D grid
# def interpolate_to_grid(latitudes, longitudes, data_1d):
#     # Create a 2D grid of lat/lon indices
#     grid_x, grid_y = np.meshgrid(np.arange(longitudes.shape[1]), np.arange(latitudes.shape[0]))
    
#     # Generate 1D index for the 1D data
#     data_grid_x = np.linspace(0, longitudes.shape[1] - 1, len(data_1d))
#     data_grid_y = np.linspace(0, latitudes.shape[0] - 1, len(data_1d))

#     # Interpolate the 1D data to fit the 2D grid
#     data_2d = griddata((data_grid_x, data_grid_y), data_1d, (grid_x, grid_y), method='nearest')
    
#     return data_2d

# # Function to create a GeoDataFrame from lat, lon, and temperature data
# def create_geodataframe(latitudes, longitudes, img_mir_temp_resampled):
#     # Flatten the 2D arrays to 1D
#     latitudes_flat = latitudes.flatten()
#     longitudes_flat = longitudes.flatten()
#     img_mir_temp_flat = img_mir_temp_resampled.flatten()

#     # Create a pandas DataFrame
#     data = pd.DataFrame({
#         'Latitude': latitudes_flat,
#         'Longitude': longitudes_flat,
#         'Temperature': img_mir_temp_flat
#     })

#     # Drop rows with NaN values (if necessary)
#     data = data.dropna()

#     # Convert DataFrame to GeoDataFrame
#     geometry = [Point(xy) for xy in zip(data['Longitude'], data['Latitude'])]
#     gdf = gpd.GeoDataFrame(data, geometry=geometry)
    
#     return gdf

# # Main function to execute the extraction, interpolation, and GeoDataFrame creation
# def main(hdf5_file):
#     # Step 1: Extract the data
#     latitudes, longitudes, img_mir_temp = extract_data(hdf5_file)
#     print(f"Latitude Shape: {latitudes.shape}")
#     print(f"Longitude Shape: {longitudes.shape}")
#     print(f"IMG_MIR_TEMP Shape: {img_mir_temp.shape}")

#     # Step 2: Interpolate the MIR temperature data to 2D
#     img_mir_temp_resampled = interpolate_to_grid(latitudes, longitudes, img_mir_temp)
#     print(f"Resampled IMG_MIR_TEMP Shape: {img_mir_temp_resampled.shape}")

#     # Step 3: Create the GeoDataFrame
#     satellite_gdf = create_geodataframe(latitudes, longitudes, img_mir_temp_resampled)
#     print(satellite_gdf.head())

# # Specify the HDF5 file path
# hdf5_file = r"C:\Users\Hii\Downloads\3DIMG_18JUN2024_0000_L1B_STD_V01R00 (1).h5"

# # Execute the main function
# if __name__ == "__main__":
#     main(hdf5_file)



#-------------------------------------INTERSECTION DONE ------------------------------------------------------
# import geopandas as gpd
# import pandas as pd
# import matplotlib.pyplot as plt
# from shapely.geometry import Point

# def perform_intersection(satellite_gdf, states_shapefile):
#     # Load the Indian states shapefile
#     states_gdf = gpd.read_file(r'C:\Users\Hii\Downloads\Indian States Shape file\Indian States dataset\output\Gujarat_boundary.shp')

#     # Ensure the CRS of both GeoDataFrames are the same
#     if satellite_gdf.crs != states_gdf.crs:
#         print(f"Transforming satellite data from CRS {satellite_gdf.crs} to CRS {states_gdf.crs}")
#         satellite_gdf = satellite_gdf.to_crs(states_gdf.crs)
    
#     # Perform intersection
#     intersection = gpd.overlay(satellite_gdf, states_gdf, how='intersection')
    
#     return intersection

# def main():
#     # Create a GeoDataFrame for the satellite data
#     # Ensure the CRS is EPSG:4326 for the input data
#     satellite_data = {
#         'Latitude': [21.923394, 22.692675, 22.242117, 22.792990, 20.925572],
#         'Longitude': [70.656187, 70.360145, 71.485200, 72.223051, 72.952940],
#         'Temperature': [306.002318, 319.531791, 334.673177, 337.521895, 348.992169],
#         'State_Name': ['Gujarat'] * 5,
#     }
#     satellite_df = pd.DataFrame(satellite_data)
#     satellite_gdf = gpd.GeoDataFrame(
#         satellite_df,
#         geometry=gpd.points_from_xy(satellite_df.Longitude, satellite_df.Latitude),
#         crs='EPSG:4326'
#     )
    
#     # Perform intersection
#     intersection_gdf = perform_intersection(satellite_gdf, 'C:/Users/Hii/Downloads/Indian States Shape file/Indian States dataset/output/Gujarat_boundary.shp')
    
#     # Save or plot the results
#     intersection_gdf.to_file('intersection_output.geojson', driver='GeoJSON')
#     intersection_gdf.plot()
#     plt.show()

# if __name__ == "__main__":
#     main()


#-----------------------------------INTERSECTION WITH PLOTTING -----------------------------------------------

