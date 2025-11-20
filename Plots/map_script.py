#!/usr/bin/env python3
"""
Function to generate the Cariaco Basin study site map using Cartopy and PyGMT data,
designed to be integrated as a panel in a larger Matplotlib figure.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import pygmt 

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# --- Fixed Parameters ---
lon_min, lon_max = -67.0, -63.5
lat_min, lat_max = 9.5, 12.0
region = [lon_min, lon_max, lat_min, lat_max] 
mooring_lon = -64.57
mooring_lat = 10.5

# --- PyGMT Data Loading ---
# Load high-resolution bathymetry once globally or inside the function
# We'll load it inside the function to ensure modularity if called multiple times, 
# but it could be loaded globally for slight speed up if the script is run many times.
def load_bathymetry(region):
    """Loads bathymetry data using PyGMT."""
    grid = pygmt.datasets.load_earth_relief(resolution="01m", region=region)
    grid = pygmt.grdclip(grid, above=[0, 0]) # Clip land elevations to 0
    return grid

# --- The Main Function ---

def draw_cariaco_map_with_inset(ax, fig, grid):
    """
    Draws the full Cariaco Basin map, including inset and colorbar, onto 
    a provided Cartopy axis (ax) and figure (fig).
    
    Args:
        ax (matplotlib.axes._subplots.AxesSubplot): The Cartopy axis to draw the main map on.
        fig (matplotlib.figure.Figure): The parent figure object for adding the colorbar.
        grid (xarray.DataArray): The PyGMT bathymetry data.
    """
    # --- Fixed Colormap ---
    cmap = plt.get_cmap("Blues_r")
    norm = mcolors.Normalize(vmin=-2000, vmax=0)

    # 1. Setup Main Map
    # The axis projection (ccrs.PlateCarree()) must be set when the figure/subplot is created 
    # in the notebook (Step 3). We just set the extent here.
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # --- Plot PyGMT Grid using Cartopy/Xarray ---
    grid.plot.pcolormesh(
        ax=ax,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        zorder=0,
        add_colorbar=False
    )

    # --- Cartopy Styling ---

    # Add land feature
    ax.add_feature(cfeature.LAND, color='antiquewhite', zorder=1)
    
    # Add gridlines with labels
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                      alpha=0.5, linestyle='--', zorder=3)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 11}
    gl.ylabel_style = {'size': 11}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # Add cities and islands
    cities = {'Puerto La Cruz': (-64.63, 10.21), 'Barcelona': (-64.70, 10.13)}
    islands = {'Isla de Margarita': (-63.95, 10.95), 'Isla la Tortuga': (-65.30, 10.93)}
    
    for label, (clon, clat) in {**cities, **islands}.items():
        ax.plot(clon, clat, 'o', color='black', markersize=5,
                transform=ccrs.PlateCarree(), zorder=5)
        ax.text(clon + 0.05, clat, label, fontsize=9, color='black',
                transform=ccrs.PlateCarree(), zorder=5)

    # Add Venezuela label
    ax.text(-64.5, 9.7, 'Venezuela', fontsize=16, color='black',
            weight='bold', ha='center', alpha=0.8,
            transform=ccrs.PlateCarree(), zorder=5)

    # Add CARIACO mooring
    ax.plot(mooring_lon, mooring_lat, 'o', color='red', markersize=10,
            markeredgecolor='darkred', markeredgewidth=1.5,
            transform=ccrs.PlateCarree(), zorder=10)
    ax.annotate('CARIACO Mooring',
                xy=(mooring_lon, mooring_lat),
                xytext=(mooring_lon, mooring_lat + 0.6),
                fontsize=11, color='red',
                ha='center', va='bottom',
                arrowprops=dict(arrowstyle='-|>', color='red', lw=2, relpos=(0.5, 0.)),
                transform=ccrs.PlateCarree(), zorder=10)

    # Add scale bar and North arrow (Code kept brief for modularity)
    # ... (rest of your scale bar and arrow code, ensured to use ax.plot/ax.text)
    
    # -------------------
    # 2. Inset Map (Requires a parent axis for transAxes coordinates)
    # -------------------
    
    inset_x, inset_y, inset_w, inset_h_total = 0.0, 1.0 - 0.30, 0.25, 0.30
    rect_patch = mpatches.Rectangle((inset_x, inset_y), inset_w, inset_h_total,
                                     facecolor='white', edgecolor='black', linewidth=1.0,
                                     transform=ax.transAxes, zorder=11)
    ax.add_patch(rect_patch)

    map_h = inset_w
    map_y = inset_y + (inset_h_total - map_h) - 0.02
    
    ax_inset = ax.inset_axes([inset_x, map_y, inset_w, map_h],
                             projection=ccrs.Orthographic(
                                 central_longitude=mooring_lon, central_latitude=mooring_lat),
                             zorder=12)

    ax_inset.set_global()
    ax_inset.add_feature(cfeature.OCEAN, color='lightblue', zorder=0)
    ax_inset.add_feature(cfeature.LAND, color='antiquewhite', zorder=1)
    ax_inset.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='grey', zorder=2)
    ax_inset.plot(mooring_lon, mooring_lat, 'o', color='red', markersize=5,
                  markeredgecolor='darkred', markeredgewidth=1.0,
                  transform=ccrs.PlateCarree(), zorder=13)
    
    # Inset Label
    ax.text(inset_x + inset_w / 2, inset_y + 0.0, "Location on World Map",
            ha='center', va='bottom', fontsize=9,
            transform=ax.transAxes, zorder=13)
            
    # -------------------
    # 3. Colorbar (Requires the figure object to be positioned relative to 'ax')
    # -------------------
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical',
                        shrink=0.7, pad=0.05, extend='min')
    cbar.set_label('Water Depth (m)', fontsize=10, weight='bold')
    
    # Final cleanup (optional: set title here or in notebook)
    ax.set_title('(a) Cariaco Basin Study Site', fontsize=14, weight='bold', pad=20, loc='left')


# --- Standalone Execution for Testing ---
if __name__ == "__main__":
    # Create the data needed for standalone run
    bathy_grid = load_bathymetry(region)
    
    fig = plt.figure(figsize=(10, 8))
    # Create the axis with the required Cartopy projection
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree()) 
    
    draw_cariaco_map_with_inset(ax, fig, bathy_grid)
    
    plt.tight_layout()
    plt.savefig('cariaco_basin_map_standalone.pdf', dpi=300, bbox_inches='tight')
    plt.show()