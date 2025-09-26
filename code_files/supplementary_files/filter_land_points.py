import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import requests
import os # To handle file paths

## This function is to load a .csv file into a pandas dataframe of desired format
def load_dataset(dataset_path, log_transform=False, bias=1e-2):
    print("...Loading Dataset...")
    raw_dataset = pd.read_csv(dataset_path)

    ## To add LOC_ID column
    loc_id = pd.Series([i for i in range(raw_dataset.shape[0])], name='LOC_ID')
    raw_dataset = pd.concat([loc_id.reset_index(drop=True), raw_dataset.reset_index(drop=True)], axis=1)

    ## If the data is to be bias + log_transformed
    if (log_transform == True):
        bias = 1e-2
        meta_columns = ['LOC_ID', 'LATITUDE', 'LONGITUDE', 'VALID_POINTS']
        dataset = raw_dataset.copy()
        dataset.loc[:, ~dataset.columns.isin(meta_columns)] = np.log1p(raw_dataset.loc[:, ~dataset.columns.isin(meta_columns)] + bias)
    else:
        dataset = raw_dataset

    return dataset

# --- 1. Load Your Data ---
full_india_dataset = load_dataset(
    dataset_path='../../extracted_gsmap_isro_data/full_india_grid_timeseries.csv',
    bias=1e-2,
    log_transform=True
)
print(f"There are {full_india_dataset.shape[0]} gridpoints in the original dataset.")


# --- 2. Convert to GeoDataFrame ---
gdf_points = gpd.GeoDataFrame(
    full_india_dataset, geometry=gpd.points_from_xy(full_india_dataset.LONGITUDE, full_india_dataset.LATITUDE), crs="EPSG:4326"
)


# --- 3. Get the Boundary of India ---
world_map_url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
local_zip_path = "natural_earth.zip"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Check if the file already exists to avoid re-downloading
if not os.path.exists(local_zip_path):
    print(f"Downloading shapefile from {world_map_url}...")
    response = requests.get(world_map_url, headers=headers)
    response.raise_for_status()
    with open(local_zip_path, "wb") as f:
        f.write(response.content)
    print("Download complete.")
else:
    print(f"Using existing shapefile: {local_zip_path}")

world = gpd.read_file(local_zip_path)
india_polygon = world[world.ADMIN == 'India']


# --- 4. Filter the Points Using a Spatial Join ---
land_points = gpd.sjoin(gdf_points, india_polygon, how="inner", predicate='within')
print(f"Number of points after filtering to landmass: {len(land_points)}")


# --- 5. Save the Filtered Data to a New CSV File ---
# Get the list of columns from your original dataset to keep
original_columns = full_india_dataset.columns.tolist()

# Select only the original columns (this drops 'index_right', 'ADMIN', 'geometry', etc.)
final_df_to_save = land_points[original_columns]

# Define the output path and save the file
output_csv_path = '../../extracted_gsmap_isro_data/filtered_india_land_points.csv'
final_df_to_save.to_csv(output_csv_path, index=False)

print(f"✅ Filtered data has been successfully saved to: {output_csv_path}")


# --- (Optional) 6. Visualize the Result ---
print("Generating visualization map...")
fig, ax = plt.subplots(figsize=(10, 10))
india_polygon.plot(ax=ax, color='lightgray', edgecolor='black')
# Use a smaller sample for plotting if the dataset is very large, to avoid clutter
plot_sample = land_points.sample(min(1000, len(land_points)))
plot_sample.plot(ax=ax, color='red', markersize=5, label=f'Land Points (Sample of {len(plot_sample)})')
plt.title("Filtered Grid Points within India's Landmass", fontsize=16)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.grid(True)
plt.savefig("india_land_points.png", dpi=300)
plt.show()