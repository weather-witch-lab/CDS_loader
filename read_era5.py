import xarray as xr
import numpy as np

def analyze_graphcast_file(file_path):
    """Detailed analysis of a GraphCast-formatted netCDF file."""
    print(f"\nReading file: {file_path}\n")
    
    # Load the dataset
    ds = xr.open_dataset(file_path)
    
    # Basic dataset info
    print("Dataset info:")
    print(ds.info())
    print("\n")
    
    # List all variables
    print("Variables:", list(ds.variables))
    print("\n")
    
    # List all dimensions
    print("Dimensions:", ds.dims)
    print("\n")
    
    # List all coordinates
    print("Coordinates:", list(ds.coords))
    print("\n")
    
    # Detailed analysis of each variable
    print("Detailed variable analysis:")
    for var_name in ds.variables:
        var = ds[var_name]
        print(f"\nVariable: {var_name}")
        print(f"Dimensions: {var.dims}")
        print(f"Shape: {var.shape}")
        print(f"Dtype: {var.dtype}")
        print("Attributes:", var.attrs)
        
        # Print some sample values
        if var.size > 0:
            print("First few values:", var.values.flatten()[:3])
            print("Min value:", float(var.min()))
            print("Max value:", float(var.max()))
    
    # Time coordinate analysis
    if 'time' in ds.coords:
        print("\nTime coordinate analysis:")
        print(f"Time values: {ds.time.values}")
        print(f"Time dtype: {ds.time.dtype}")
    
    if 'datetime' in ds.coords:
        print("\nDatetime coordinate analysis:")
        print(f"Datetime values: {ds.datetime.values}")
        print(f"Datetime dtype: {ds.datetime.dtype}")
        print(f"Datetime dims: {ds.datetime.dims}")
    
    # Level coordinate analysis
    if 'level' in ds.coords:
        print("\nPressure levels:")
        print(ds.level.values)
    
    # Check for any _FillValue or missing values
    print("\nMissing value analysis:")
    for var_name in ds.variables:
        if '_FillValue' in ds[var_name].encoding:
            print(f"{var_name} _FillValue:", ds[var_name].encoding['_FillValue'])
        if np.any(np.isnan(ds[var_name].values)):
            print(f"{var_name} has NaN values")

if __name__ == "__main__":
    file_path = "source-era5_date-2022-01-01_res-0.25_levels-37_steps-01.nc"
    analyze_graphcast_file(file_path)