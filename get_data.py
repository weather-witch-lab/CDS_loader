import cdsapi
from datetime import datetime, timedelta
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os
import time
import zipfile
import shutil
import schedule
import pandas as pd

def create_data_folders(year, month, day):
    """Create folder structure for data storage"""
    base_path = 'era5_data'
    date_folder = f"{year}-{month}-{day}"
    full_path = os.path.join(base_path, date_folder)
    
    # Create folders if they don't exist
    os.makedirs(full_path, exist_ok=True)
    
    return full_path

def merge_era5_data(data_path, year, month, day, hour):
    """Create new dataset in GraphCast format with single timestep"""
    print("\nMerging ERA5 data files...")
    try:
        # Load source files
        date_hour_prefix = f"{year}-{month}-{day}-{hour}"
        surface_ds = xr.open_dataset(os.path.join(data_path, f'{date_hour_prefix}-surface.nc'))
        toa_ds = xr.open_dataset(os.path.join(data_path, f'{date_hour_prefix}-toa.nc'))
        precip_ds = xr.open_dataset(os.path.join(data_path, f'{date_hour_prefix}-precip.nc'))
        pressure_ds = xr.open_dataset(os.path.join(data_path, f'{date_hour_prefix}-pressure.nc'))

        # Create target grid
        lon_target = np.arange(0, 360, 0.25, dtype=np.float32)
        lat_target = np.arange(-90, 90.25, 0.25, dtype=np.float32)
        
        # Define pressure levels explicitly
        pressure_levels = np.array([1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 
                                  225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 
                                  800, 825, 850, 875, 900, 925, 950, 975, 1000], dtype=np.int32)
        
        def process_to_array(data, target_shape):
            """Convert data to numpy array with correct shape and interpolation"""
            if 'valid_time' in data.dims:
                data = data.isel(valid_time=0)
            
            # Interpolate to target grid
            data = data.interp(
                longitude=lon_target,
                latitude=lat_target,
                method='linear'
            )
            
            # Convert to numpy and reshape
            arr = data.values.astype(np.float32)
            return arr.reshape(target_shape)

        # Create new dataset
        ds = xr.Dataset()
        
        # Update the coordinates section
        ds.coords['lon'] = ('lon', lon_target)
        ds.coords['lat'] = ('lat', lat_target)
        ds.coords['level'] = ('level', pressure_levels)
        
        # Create time and datetime coordinates properly
        time_value = pd.Timestamp(f"{year}-{month}-{day}T{hour}:00:00")
        ds.coords['time'] = ('time', [time_value])
        ds.coords['datetime'] = ('time', [time_value])
        ds.coords['batch'] = ('batch', [0])

        # Add progress coordinate (required for GraphCast)
        progress = np.array([0.0], dtype=np.float32)  # Single timestep
        ds.coords['progress'] = ('time', progress)

        # Process 2D variables
        for var_name, (data, long_name, units) in var_4d_mapping.items():
            arr = process_to_array(data, (721, 1440))
            ds[var_name] = xr.DataArray(
                arr[np.newaxis, np.newaxis, :, :],  # [batch, time, lat, lon]
                dims=['batch', 'time', 'lat', 'lon'],
                attrs={'long_name': long_name, 'units': units}
            )

        # Process 5D variables (pressure level variables)
        for var_name, (data, long_name, units) in var_5d_mapping.items():
            arr = np.stack([
                process_to_array(data.sel(pressure_level=lev), (721, 1440))
                for lev in pressure_levels
            ])
            ds[var_name] = xr.DataArray(
                arr[np.newaxis, np.newaxis, :, :, :],  # [batch, time, level, lat, lon]
                dims=['batch', 'time', 'level', 'lat', 'lon'],
                attrs={'long_name': long_name, 'units': units}
            )

        # Ensure consistent dimension order
        ds = ds.transpose('batch', 'time', 'level', 'lat', 'lon')

        # Save with GraphCast naming convention
        output_file = os.path.join(data_path, f'source-era5_date-{year}-{month}-{day}_res-0.25_levels-37_steps-01.nc')
        ds.to_netcdf(output_file)
        print(f"Data merged and saved in GraphCast format: {output_file}")

        # Close input datasets
        surface_ds.close()
        toa_ds.close()
        precip_ds.close()
        pressure_ds.close()

        return output_file

    except Exception as e:
        print(f"Error merging data: {e}")
        print("\nPressure dataset info:")
        if 'pressure_ds' in locals():
            print("Dimensions:", pressure_ds.dims)
            print("Coordinates:", pressure_ds.coords)
            print("Data variables:", list(pressure_ds.data_vars))
        return None

def get_era5_data():
    # Calculate the date 6 days ago (for final release data)
    current_date = datetime.utcnow() - timedelta(days=6)
    year = str(current_date.year)
    month = str(current_date.month).zfill(2)
    day = str(current_date.day).zfill(2)
    hour = str(current_date.hour).zfill(2)

    print(f"Requesting data for: {year}-{month}-{day} {hour}:00")
    
    # Create folders for this date
    data_path = create_data_folders(year, month, day)
    
    # Define file paths with new naming convention
    date_hour_prefix = f"{year}-{month}-{day}-{hour}"
    surface_file = os.path.join(data_path, f'{date_hour_prefix}-surface.nc')
    toa_file = os.path.join(data_path, f'{date_hour_prefix}-toa.nc')
    precip_file = os.path.join(data_path, f'{date_hour_prefix}-precip.nc')
    pressure_file = os.path.join(data_path, f'{date_hour_prefix}-pressure.nc')

    client = cdsapi.Client()
    max_retries = 3
    wait_time = 30
    
    # First request: surface-level variables
    print("Downloading surface data...")
    for attempt in range(max_retries):
        try:
            surface_request = {
                "product_type": ["reanalysis"],
                "variable": [
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                    "2m_temperature",
                    "mean_sea_level_pressure",
                    "geopotential",
                    "land_sea_mask"
                ],
                "year": [year],
                "month": [month],
                "day": [day],
                "time": [hour],
                "data_format": "netcdf",
                "download_format": "unarchived"
            }
            client.retrieve("reanalysis-era5-single-levels", surface_request, surface_file)
            print("Surface data downloaded successfully")
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Error downloading surface data after {max_retries} attempts: {e}")
                return None, None

    # Separate request for TOA radiation
    print("Downloading TOA radiation data...")
    for attempt in range(max_retries):
        try:
            toa_request = {
                "product_type": ["reanalysis"],
                "variable": ["toa_incident_solar_radiation"],
                "year": [year],
                "month": [month],
                "day": [day],
                "time": [hour],
                "data_format": "netcdf",
                "download_format": "unarchived"
            }
            client.retrieve("reanalysis-era5-single-levels", toa_request, toa_file)
            print("TOA radiation data downloaded successfully")
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Error downloading TOA data after {max_retries} attempts: {e}")
                return None, None

    # Separate request for 6-hour precipitation
    print("Downloading 6-hour precipitation data...")
    for attempt in range(max_retries):
        try:
            precip_time = datetime.strptime(f"{year}-{month}-{day} {hour}", "%Y-%m-%d %H") - timedelta(hours=5)
            precip_hours = [f"{h:02d}:00" for h in range(precip_time.hour, precip_time.hour + 6)]
            
            precip_request = {
                "product_type": ["reanalysis"],
                "variable": ["total_precipitation"],
                "year": [year],
                "month": [month],
                "day": [day],
                "time": precip_hours,
                "data_format": "netcdf",
                "download_format": "unarchived"
            }
            
            # Download to a temporary file first
            temp_precip_file = os.path.join(data_path, 'temp_precip.nc')
            client.retrieve("reanalysis-era5-single-levels", precip_request, temp_precip_file)
            
            # Process the data and save to final location
            with xr.open_dataset(temp_precip_file) as ds:
                precip_6hr = ds['tp'].isel(valid_time=slice(-6, None)).sum('valid_time')
                new_ds = xr.Dataset({'total_precipitation_6hr': precip_6hr})
                new_ds.to_netcdf(precip_file)
            
            if os.path.exists(temp_precip_file):
                os.remove(temp_precip_file)
                
            print("Precipitation data downloaded and processed successfully")
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if os.path.exists(temp_precip_file):
                os.remove(temp_precip_file)
            if attempt < max_retries - 1:
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Error downloading precipitation data after {max_retries} attempts: {e}")
                return None, None

    # Pressure level request
    print("Downloading pressure level data...")
    for attempt in range(max_retries):
        try:
            pressure_request = {
                "product_type": ["reanalysis"],
                "variable": [
                    "temperature",
                    "geopotential",
                    "u_component_of_wind",
                    "v_component_of_wind",
                    "vertical_velocity",
                    "specific_humidity"
                ],
                "pressure_level": [
                    "1", "2", "3", "5", "7", "10", "20", "30", "50", "70",
                    "100", "125", "150", "175", "200", "225", "250", "300", "350",
                    "400", "450", "500", "550", "600", "650", "700", "750", "775",
                    "800", "825", "850", "875", "900", "925", "950", "975", "1000"
                ],
                "year": [year],
                "month": [month],
                "day": [day],
                "time": [hour],
                "data_format": "netcdf",
                "download_format": "unarchived"
            }
            client.retrieve("reanalysis-era5-pressure-levels", pressure_request, pressure_file)
            print("Pressure level data downloaded successfully")
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Error downloading pressure level data after {max_retries} attempts: {e}")
                return None, None

    print(f"\nAll data downloaded to: {data_path}")
    
    # Merge the data after downloading
    merged_file = merge_era5_data(data_path, year, month, day, hour)
    if merged_file:
        print(f"Data successfully merged to: {merged_file}")
    
    return data_path, hour

def job():
    print(f"\nStarting data collection at {datetime.utcnow()}")
    data_path, hour = get_era5_data()
    # Extract year, month, day from data_path
    date_folder = os.path.basename(data_path)  # Gets YYYY-MM-DD
    print(f"Data collection completed for {date_folder} {hour}:00")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ERA5 Data Collector')
    parser.add_argument('--schedule', action='store_true', help='Run in scheduled mode')
    args = parser.parse_args()
    
    if args.schedule:
        # Schedule the job to run every hour
        schedule.every().hour.at(":00").do(job)
        print("Scheduler started. Will run every hour at :00")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    else:
        # Run once immediately
        data_path, hour = get_era5_data()
        date_folder = os.path.basename(data_path)  # Gets YYYY-MM-DD
        print(f"Data collection completed for {date_folder} {hour}:00")
