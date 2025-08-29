import os
import h5py
import numpy as np
import pandas as pd
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from collections import defaultdict
import argparse

# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description="Generate grid rainfall timeseries CSV")

    parser.add_argument('--lat_min', type=float, default=12.95)
    parser.add_argument('--lat_max', type=float, default=13.05)
    parser.add_argument('--lon_min', type=float, default=77.45)
    parser.add_argument('--lon_max', type=float, default=77.55)
    parser.add_argument('--output_csv', type=str, default="/home/prad/code/precipitation_gauge_gnn/grid_timeseries_test.csv")

    return parser.parse_args()

# args = parse_args()

# LAT_MIN = args.lat_min
# LAT_MAX = args.lat_max
# LON_MIN = args.lon_min
# LON_MAX = args.lon_max
# OUTPUT_CSV = args.output_csv

LAT_MIN = 8
LAT_MAX = 38
LON_MIN = 68
LON_MAX = 97.5
OUTPUT_CSV = "/home/prad/code/precipitation_gauge_gnn/full_india_grid_timeseries.csv"

# --- User Settings ---
BASE_DIR = "/home/prad/code/precipitation_gauge_gnn/gsmap_isro"
LAT_MIN, LAT_MAX = LAT_MIN, LAT_MAX
LON_MIN, LON_MAX = LON_MIN, LON_MAX
OUTPUT_CSV = OUTPUT_CSV

print("The task is to write rainfall data for locations between:")
print(f"Latitude: [{LAT_MIN}, {LAT_MAX}]")
print(f"Longitude: [{LON_MIN}, {LON_MAX}]")
print(f"To the file: {OUTPUT_CSV}")

# --- Grid Specs ---
LAT_START, LON_START = -89.95, -179.95
LAT_STEP, LON_STEP = 0.1, 0.1

def lat_to_idx(lat):
    return int(round((lat - LAT_START) / LAT_STEP))

def lon_to_idx(lon):
    return int(round((lon - LON_START) / LON_STEP))

lat_start_idx = lat_to_idx(LAT_MIN)
lat_end_idx = lat_to_idx(LAT_MAX) + 1
lon_start_idx = lon_to_idx(LON_MIN)
lon_end_idx = lon_to_idx(LON_MAX) + 1

# --- Prepare grid points ---
lat_idxs = list(range(lat_start_idx, lat_end_idx))
lon_idxs = list(range(lon_start_idx, lon_end_idx))
grid_points = [(i, j) for i in lon_idxs for j in lat_idxs]  # (lon_idx, lat_idx)
print(f"Number of longitudes: {len(lon_idxs)}")
print(f"Number of latitudes: {len(lat_idxs)}")
print(f"Fetching daily rainfall data for a total of {len(grid_points)} locations...")

# --- Build list of filepaths and associated dates ---
filepaths = []
dates = []

for year in range(2015, 2023):
    year_dir = os.path.join(BASE_DIR, str(year))
    if not os.path.exists(year_dir):
        print(f"{year_dir} is not a valid directory!")
        continue
    for fname in sorted(os.listdir(year_dir)):
        if not fname.endswith(".h5"):
            print(f"{fname} is not a valid file (.h5)!")
            continue
        try:
            ts_str = fname.split("_")[2]
            dt = datetime.strptime(ts_str, "%y%m%d%H%M")
            filepaths.append(os.path.join(year_dir, fname))
            dates.append(dt.strftime("%Y-%m-%d"))
        except:
            continue

# --- Initialize storage ---
grid_time_series = {
    (i, j): {
        "rainfall_mm": np.zeros(len(set(dates)), dtype=np.float32),
        "valid_counts": np.zeros(len(set(dates)), dtype=np.int32)
    }
    for i, j in grid_points
}
unique_dates = sorted(list(set(dates)))
date_to_index = {date: idx for idx, date in enumerate(unique_dates)}

# --- Helper to process one file ---
def process_single_file(filepath):
    fname = os.path.basename(filepath)
    try:
        ts_str = fname.split("_")[2]
        dt = datetime.strptime(ts_str, "%y%m%d%H%M")
        date_str = dt.strftime("%Y-%m-%d")
    except:
        return None

    try:
        with h5py.File(filepath, 'r') as f:
            data = f['/Grid/hourlyPrecipRateGC'][lon_start_idx:lon_end_idx, lat_start_idx:lat_end_idx]
            data = np.array(data)
            fill_val = f['/Grid/hourlyPrecipRateGC'].attrs.get('_FillValue', -9999.9)
            data[data == fill_val] = np.nan
            return date_str, data
    except:
        return None

# --- Parallel Processing ---
intermediate_results = defaultdict(list)

with ProcessPoolExecutor() as executor:
    futures = [executor.submit(process_single_file, path) for path in filepaths]
    for f in tqdm(as_completed(futures), total=len(futures), desc="Processing HDF5"):
        result = f.result()
        if result is None:
            continue
        date_str, data = result
        date_idx = date_to_index[date_str]
        for x, (i, j) in enumerate(grid_points):
            val = data[i - lon_start_idx, j - lat_start_idx]
            if not np.isnan(val):
                grid_time_series[(i, j)]["rainfall_mm"][date_idx] += val
                grid_time_series[(i, j)]["valid_counts"][date_idx] += 1

# --- Write output ---
records = []
for (i, j), values in grid_time_series.items():
    lon = round(LON_START + i * LON_STEP, 2)
    lat = round(LAT_START + j * LAT_STEP, 2)
    valid_points = np.sum(values["valid_counts"])
    row = {
        "LONGITUDE": lon,
        "LATITUDE": lat,
        "VALID_POINTS": valid_points
    }
    row.update({date: rain for date, rain in zip(unique_dates, values["rainfall_mm"])})
    records.append(row)

df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved grid timeseries to {OUTPUT_CSV}")