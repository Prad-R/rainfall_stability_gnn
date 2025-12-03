import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import os

## This function is to load a .csv file into a pandas dataframe of desired format
def load_dataset(dataset_path, log_transform=False, bias=1e-2):
    print("...Loading Dataset...")
    raw_dataset = pd.read_csv(dataset_path)

    ## To add LOC_ID column
    loc_id = pd.Series([i for i in range(raw_dataset.shape[0])], name='LOC_ID')
    raw_dataset = pd.concat([loc_id.reset_index(drop=True), raw_dataset.reset_index(drop=True)], axis=1)

    ## If the data is to be bias + log_transformed
    if (log_transform == True):
        meta_columns = ['LOC_ID', 'LATITUDE', 'LONGITUDE', 'VALID_POINTS']
        dataset = raw_dataset.copy()
        dataset.loc[:, ~dataset.columns.isin(meta_columns)] = np.log1p(raw_dataset.loc[:, ~dataset.columns.isin(meta_columns)] + bias)
    else:
        dataset = raw_dataset

    return dataset

# --- 1. Load Your Data ---
full_india_dataset = load_dataset(
    dataset_path='../../extracted_gsmap_isro_data/full_india_grid_timeseries_20_years.csv',
    bias=1e-2,
    log_transform=True
)
print(f"There are {full_india_dataset.shape[0]} gridpoints in the original dataset.")

# --- 2. Convert to GeoDataFrame ---
gdf_points = gpd.GeoDataFrame(
    full_india_dataset, geometry=gpd.points_from_xy(full_india_dataset.LONGITUDE, full_india_dataset.LATITUDE), crs="EPSG:4326"
)


# --- 3. Get the Boundary of India from your local state shapefile ---
#    (Replaces the entire download block)
#    This is the path from your plotting function.
your_shapefile_path = "/home/prad/code/precipitation_gauge_gnn/shapefiles/india/india_updated_state_boundary.shp"

print(f"Reading state shapefile from: {your_shapefile_path}")
# Read the shapefile containing all the states
states_gdf = gpd.read_file(your_shapefile_path)

# --- This is the key change ---
# To create a single polygon for all of India, we dissolve the state boundaries.
# 1. Create a dummy column to dissolve by.
states_gdf['COUNTRY'] = 'India'
# 2. Dissolve all state polygons into one single polygon.
india_polygon = states_gdf.dissolve(by='COUNTRY')
# -----------------------------


# --- 4. Filter the Points Using a Spatial Join ---
# Ensure both GeoDataFrames use the same CoordinateReferenceSystem (CRS)
if gdf_points.crs != india_polygon.crs:
    print("Warning: CRS mismatch. Re-projecting points to match shapefile...")
    gdf_points = gdf_points.to_crs(india_polygon.crs)
    
land_points = gpd.sjoin(gdf_points, india_polygon, how="inner", predicate='within')
print(f"Number of points after filtering to landmass: {len(land_points)}")


# --- 5. Save the Filtered Data to a New CSV File ---
original_columns = full_india_dataset.columns.tolist()
final_df_to_save = land_points[original_columns]
output_csv_path = '../../extracted_gsmap_isro_data/filtered_india_land_points_20_years_new.csv'
final_df_to_save.to_csv(output_csv_path, index=False)

print(f"✅ Filtered data has been successfully saved to: {output_csv_path}")


# --- (Optional) 6. Visualize the Result ---
print("Generating visualization map...")
fig, ax = plt.subplots(figsize=(10, 10))
# Plot the dissolved India polygon
india_polygon.plot(ax=ax, color='lightgray', edgecolor='black')
# Plot a sample of the points
plot_sample = land_points.sample(min(1000, len(land_points)))
plot_sample.plot(ax=ax, color='red', markersize=5, label=f'Land Points (Sample of {len(plot_sample)})')
plt.title("Filtered Grid Points within India's Landmass", fontsize=16)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.grid(True)
plt.savefig("india_land_points.png", dpi=300)
plt.show()