
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

def merge_dataframes(df1, df2):
    df1['times'] = pd.to_datetime(df1['times'])
    df2['times'] = pd.to_datetime(df2['times'])
    merged_df = pd.merge(df1, df2, on='times', how='outer')
    return merged_df

ds = xr.open_dataset("output/raw_datasets/raw_nwp_dataset.nc")
df1 = pd.read_csv("raw_data/targets/price.csv") 
df2 = pd.read_csv("raw_data/targets/production.csv")
df = merge_dataframes(df1, df2)

valid_times = pd.to_datetime(ds.time.values)

df['hour'] = df['times'].dt.floor('h')
df['minute'] = df['times'].dt.minute

new_df = pd.DataFrame({'times': valid_times})

columns = [
    'System_Load_Actual_Value', 'System_Load_Forecast_Value',
    'Wind_and_Solar_Actual_Value', 'Wind_and_Solar_Forecast_Value',
    'Tie_Line_Actual_Value', 'Tie_Line_Forecast_Value',
    'Wind_Power_Actual_Value', 'Wind_Power_Forecast_Value',
    'Photovoltaic_Actual_Value', 'Photovoltaic_Forecast_Value',
    'Hydro_Power_Actual_Value', 'Hydro_Power_Forecast_Value',
    'Non_marketized_Unit_Actual_Value', 'Non_marketized_Unit_Forecast_Value',
    'A'
]

for col in columns:
    if col in df.columns:
        pivoted = df.pivot(index='hour', columns='minute', values=col)
        pivoted = pivoted.reindex(columns=[0, 15, 30, 45])
        pivoted_filtered = pivoted.reindex(valid_times)
        new_df[col] = pivoted_filtered.values.tolist()

print(new_df.head())
print(new_df['A'].iloc[0])

new_df.to_csv("output/raw_datasets/target_aligned_dataset.csv", index=False)




