# Data processing libs.
import numpy as np
import xarray as xr

# Visualisation libs.
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Set parameters.
# 🔴Critical: Error handling when opening a file.
try:
    # Parent coordinate file.
    pcf = xr.open_dataset('/mnt/localssd/Data_nemo/Meshes_domains/Coordinates/Global/ORCA_R36_coord_new.nc').squeeze()
except FileNotFoundError:
    print("Error: Parent coordinate file not found.")
    exit(1)
except Exception as e:
    print(f"Error loading parent coordinate file: {e}")
    exit(1)

x_middle = int(pcf['x'].size / 2)

# Enter Pacific and Atlantic y-indicies (latitude-like).
pac_first_yind = 7550
pac_last_yind = -2
atl_first_yind = 7350
atl_last_yind = -1

# Enter first and last x-indicies (latitude-like). It must be less than 1/2 x-dimention size!
pac_first_xind = 1500
pac_last_xind = 5900
atl_first_xind = pac_last_xind + (x_middle - pac_last_xind) * 2 + 1
atl_last_xind = pcf['x'].size - pac_first_xind + 1

# Saving properties
target_path = '/mnt/localssd/Data_nemo/Meshes_domains/Coordinates/Regional'
target_name = 'arct_cutorca36_coord.nc'


# Patch processing
def grid_selector(pcf, var, extent, pac_patch=False):
    '''
    Functon that select and cut 2D arrays from parent global ORCA coordinate file.

    Parameters
    ----------
    pcf : xarray Dataset
        Parent global ORCA Coordinate File.
    var : str
        Grid variable name from pcf.
    extent : list
        List with indices to cut [y_min, y_max, x_min, x_max].
    pac_patch : bool  # 🔴Critical: The docString was indicated by atl_patch, but the code uses pac_patch.
        Pacific (True) and Atlantic (False) switch (default is False)

    Returns
    -------
    grid_array : ndarray
        Ndarray to put in patch dataset.
    '''

    # grid type lists
    t_vars = ['nav_lon', 'nav_lat', 'glamt', 'gphit', 'e1t', 'e2t']
    u_vars = ['glamu', 'gphiu', 'e1u', 'e2u']
    v_vars = ['glamv', 'gphiv', 'e1v', 'e2v']
    f_vars = ['glamf', 'gphif', 'e1f', 'e2f']

    grid_array = None

    # 🔴Critical: Error handling if var not found in parent coordinate file.
    try:
        if pac_patch:  # Pacific patch selection
            if var in t_vars:
                grid_array = np.flip(pcf[var].sel(y=slice(extent[0], extent[1]), x=slice(extent[2], extent[3])).values)
            elif var in u_vars:
                grid_array = np.flip(
                    pcf[var].sel(y=slice(extent[0], extent[1]), x=slice(extent[2] - 1, extent[3] - 1)).values)
            elif var in v_vars:
                grid_array = np.flip(
                    pcf[var].sel(y=slice(extent[0] - 1, extent[1] - 1), x=slice(extent[2], extent[3])).values)
            elif var in f_vars:
                grid_array = np.flip(
                    pcf[var].sel(y=slice(extent[0] - 1, extent[1] - 1), x=slice(extent[2] - 1, extent[3] - 1)).values)
        else:  # Atlantic patch selection
            grid_array = pcf[var].sel(y=slice(extent[0], extent[1]), x=slice(extent[2], extent[3])).values
    except KeyError:
        print(f"Error: Variable {var} not found in parent coordinate file.")
        return None

    return grid_array



# Dataset creation
# TODO: dataset generator. I don't like this wet shit.
# Atlantic patch as xarray Dataset
atl_extent = [atl_first_yind, atl_last_yind, atl_first_xind, atl_last_xind]
atl_dataset = xr.Dataset(
    data_vars=dict(
        nav_lon=(["y", "x"], grid_selector(pcf, 'nav_lon', atl_extent)),
        nav_lat=(["y", "x"], grid_selector(pcf, 'nav_lat', atl_extent)),
        glamt=(["y", "x"], grid_selector(pcf, 'glamt', atl_extent)),
        glamu=(["y", "x"], grid_selector(pcf, 'glamu', atl_extent)),
        glamv=(["y", "x"], grid_selector(pcf, 'glamv', atl_extent)),
        glamf=(["y", "x"], grid_selector(pcf, 'glamf', atl_extent)),
        gphit=(["y", "x"], grid_selector(pcf, 'gphit', atl_extent)),
        gphiu=(["y", "x"], grid_selector(pcf, 'gphiu', atl_extent)),
        gphiv=(["y", "x"], grid_selector(pcf, 'gphiv', atl_extent)),
        gphif=(["y", "x"], grid_selector(pcf, 'gphif', atl_extent)),
        e1t=(["y", "x"], grid_selector(pcf, 'e1t', atl_extent)),
        e1u=(["y", "x"], grid_selector(pcf, 'e1u', atl_extent)),
        e1v=(["y", "x"], grid_selector(pcf, 'e1v', atl_extent)),
        e1f=(["y", "x"], grid_selector(pcf, 'e1f', atl_extent)),
        e2t=(["y", "x"], grid_selector(pcf, 'e2t', atl_extent)),
        e2u=(["y", "x"], grid_selector(pcf, 'e2u', atl_extent)),
        e2v=(["y", "x"], grid_selector(pcf, 'e2v', atl_extent)),
        e2f=(["y", "x"], grid_selector(pcf, 'e2f', atl_extent))
    )
)

# Pacific patch as xarray Dataset
pac_extent = [pac_first_yind, pac_last_yind, pac_first_xind, pac_last_xind]
pac_dataset = xr.Dataset(
    data_vars=dict(
        nav_lon=(["y", "x"], grid_selector(pcf, 'nav_lon', pac_extent, pac_patch=True)),
        nav_lat=(["y", "x"], grid_selector(pcf, 'nav_lat', pac_extent, pac_patch=True)),
        glamt=(["y", "x"], grid_selector(pcf, 'glamt', pac_extent, pac_patch=True)),
        glamu=(["y", "x"], grid_selector(pcf, 'glamu', pac_extent, pac_patch=True)),
        glamv=(["y", "x"], grid_selector(pcf, 'glamv', pac_extent, pac_patch=True)),
        glamf=(["y", "x"], grid_selector(pcf, 'glamf', pac_extent, pac_patch=True)),
        gphit=(["y", "x"], grid_selector(pcf, 'gphit', pac_extent, pac_patch=True)),
        gphiu=(["y", "x"], grid_selector(pcf, 'gphiu', pac_extent, pac_patch=True)),
        gphiv=(["y", "x"], grid_selector(pcf, 'gphiv', pac_extent, pac_patch=True)),
        gphif=(["y", "x"], grid_selector(pcf, 'gphif', pac_extent, pac_patch=True)),
        e1t=(["y", "x"], grid_selector(pcf, 'e1t', pac_extent, pac_patch=True)),
        e1u=(["y", "x"], grid_selector(pcf, 'e1u', pac_extent, pac_patch=True)),
        e1v=(["y", "x"], grid_selector(pcf, 'e1v', pac_extent, pac_patch=True)),
        e1f=(["y", "x"], grid_selector(pcf, 'e1f', pac_extent, pac_patch=True)),
        e2t=(["y", "x"], grid_selector(pcf, 'e2t', pac_extent, pac_patch=True)),
        e2u=(["y", "x"], grid_selector(pcf, 'e2u', pac_extent, pac_patch=True)),
        e2v=(["y", "x"], grid_selector(pcf, 'e2v', pac_extent, pac_patch=True)),
        e2f=(["y", "x"], grid_selector(pcf, 'e2f', pac_extent, pac_patch=True))
    ),
)

whole_dataset = xr.concat([atl_dataset, pac_dataset], dim='y')

# 🔴Critical: Error handling when saving.
# Save dataset
try:
    whole_dataset.to_netcdf(f'{target_path}/{target_name}')
    print(f"Dataset saved to {target_path}/{target_name}")
except Exception as e:
    print(f"Error saving dataset: {e}")

# Visualization
try:
    plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    whole_dataset.nav_lon.plot(ax=ax, transform=ccrs.PlateCarree())
    plt.show()
except Exception as e:
    print(f"Error during visualization: {e}")