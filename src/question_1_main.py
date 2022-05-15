import pandas as pd
import os
import glob

demand_path = os.getcwd() + "\dataset\demand"
source_path = os.getcwd() + "\dataset\sources"

demand_csv_files = glob.glob(os.path.join(demand_path, "*.csv"))
source_csv_files = glob.glob(os.path.join(source_path, "*.csv"))

demand_frames = []
for csv in demand_csv_files:
    try:
        n_ds = pd.read_csv(csv)
        n_df = pd.DataFrame(n_ds)

        demand_frames.append(n_df)
    except pd.errors.EmptyDataError:
        continue

demands = pd.concat(demand_frames, axis=0, ignore_index=True)

source_frames = []
for csv in source_csv_files:
    try:
        n_ds = pd.read_csv(csv)
        n_df = pd.DataFrame(n_ds)

        source_frames.append(n_df)
    except pd.errors.EmptyDataError:
        continue

sources = pd.concat(source_frames, axis=0, ignore_index=True)

frames = [demands[:-1], sources]
df = pd.concat(frames, axis=1)

