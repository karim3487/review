# Data processing libs.
from typing import Optional

import numpy as np
import xarray as xr

# Visualisation libs.
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Logging and configuration
import logging
from pathlib import Path
from environs import Env
from xarray import Dataset

# 🟡Preferably: Load environment variables for configuration.
env = Env()
env.read_env()

# 🟡Preferably: Setup logging to capture runtime information and errors.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# 🟡Preferably: Encapsulate configuration in a dedicated class.
class Config:
    def __init__(
            self,
            parent_file: str,
            target_path: str,
            target_name: str,
            pac_extent: list,
            atl_extent: list,
            x_middle: Optional[int] = None,
    ) -> None:
        self.parent_file = Path(parent_file)
        self.target_path = Path(target_path)
        self.target_name = target_name
        self.x_middle = x_middle
        self.pac_extent = pac_extent
        self.atl_extent = atl_extent

    def load(self) -> None:
        """Load and validate configuration."""
        if not self.parent_file.is_file():
            logging.error(f"Critical: Parent file not found at {self.parent_file}")
            raise FileNotFoundError
        if not self.target_path.exists():
            self.target_path.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created target directory: {self.target_path}")

        # Calculate Atlantic extent based on Pacific extent
        pcf = xr.open_dataset(self.parent_file).squeeze()
        self.x_middle = int(pcf["x"].size / 2)
        self.atl_extent[2] = (
                self.pac_extent[3] + (self.x_middle - self.pac_extent[3]) * 2 + 1
        )
        self.atl_extent[3] = pcf["x"].size - self.pac_extent[2] + 1


# Patch processing
def grid_selector(pcf: Dataset, var: str, extent: list, pac_patch: bool = False) -> np.ndarray[tuple[int, ...]] | None:
    """
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
    """

    # 🟡Preferably: Using valid_vars in the form of a dict is preferable than four separate lists.
    valid_vars = {
        "t": ["nav_lon", "nav_lat", "glamt", "gphit", "e1t", "e2t"],
        "u": ["glamu", "gphiu", "e1u", "e2u"],
        "v": ["glamv", "gphiv", "e1v", "e2v"],
        "f": ["glamf", "gphif", "e1f", "e2f"]
    }

    if var not in sum(valid_vars.values(), []):
        logging.error(f"Variable {var} not found in valid variables.")
        return None

    # grid type lists
    grid_type = next((key for key, vars in valid_vars.items() if var in vars), None)

    if grid_type == "t":
        y_slice, x_slice = slice(extent[0], extent[1]), slice(extent[2], extent[3])
    elif grid_type == "u":
        y_slice, x_slice = slice(extent[0], extent[1]), slice(extent[2] - 1, extent[3] - 1)
    elif grid_type == "v":
        y_slice, x_slice = slice(extent[0] - 1, extent[1] - 1), slice(extent[2], extent[3])
    elif grid_type == "f":
        y_slice, x_slice = slice(extent[0] - 1, extent[1] - 1), slice(extent[2] - 1, extent[3] - 1)

    # 🔴Critical: Error handling if var not found in parent coordinate file.
    try:
        grid_array = np.flip(pcf[var].sel(y=y_slice, x=x_slice).values)
    except IndexError:
        logging.error(f"Index out of bounds for variable {var}.")
        return None

    return grid_array


# Dataset creation
def create_dataset(pcf: Dataset, extent: list, pac_patch: bool = False) -> Dataset:
    """
    Create an xarray.Dataset for a given extent.

    Parameters
    ----------
    pcf : xarray Dataset
        Parent global ORCA Coordinate File.
    extent : list
        List with indices to cut [y_min, y_max, x_min, x_max].
    pac_patch : bool
        Pacific (True) and Atlantic (False) switch (default is False).

    Returns
    -------
    dataset : xarray.Dataset
        Created dataset.
    """
    variables = [
        "nav_lon", "nav_lat", "glamt", "glamu", "glamv", "glamf",
        "gphit", "gphiu", "gphiv", "gphif", "e1t", "e1u", "e1v", "e1f",
        "e2t", "e2u", "e2v", "e2f"
    ]

    # 🟡Preferably: Use dictionary comprehension to build data_vars concisely.
    data_vars = {
        var: (["y", "x"], grid_selector(pcf, var, extent, pac_patch))
        for var in variables
    }

    return xr.Dataset(data_vars)


def visualize_dataset(dataset: Dataset):
    """
    Visualize the dataset using Cartopy.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset to visualize.
    """
    if "nav_lon" not in dataset:
        logging.error("Critical: 'nav_lon' not found in dataset. Skipping visualization.")
        return

    try:
        plt.figure(figsize=(10, 5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()
        dataset.nav_lon.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="coolwarm")
        plt.title("Longitude Visualization")
        plt.show()
    except Exception as e:
        logging.error(f"Critical: Error during visualization: {e}")


def main():
    # 🟡Preferably: Use a main function to encapsulate the script execution.
    config = Config(
        env.str("parent_file"),
        env.str("target_path"),
        env.str("target_name"),
        [7550, -2, 1500, 5900],
        [7350, -1, None, None])
    config.load()

    try:
        pcf = xr.open_dataset(config.parent_file).squeeze()
        logging.info("Parent coordinate file loaded successfully.")

        # Create datasets
        atl_dataset = create_dataset(pcf, config.atl_extent)
        pac_dataset = create_dataset(pcf, config.pac_extent, pac_patch=True)

        # Concatenate datasets
        whole_dataset = xr.concat([atl_dataset, pac_dataset], dim="y")
        logging.info("Datasets concatenated successfully.")

        # Save dataset
        target_file = config.target_path / config.target_name
        whole_dataset.to_netcdf(target_file)
        logging.info(f"Dataset saved to {target_file}")

        # Visualize dataset
        visualize_dataset(whole_dataset)

    except Exception as e:
        logging.error(f"Critical: An error occurred: {e}")


if __name__ == "__main__":
    # 🟡Preferably: Ensure main() is executed only when running the script directly.
    main()
