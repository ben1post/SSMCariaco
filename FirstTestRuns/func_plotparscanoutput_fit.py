import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import xarray as xr

import matplotlib.colors as mcolors

# --- Parameters ---
target_composition = np.array([0.308, 0.290, 0.402])  # Replace with your desired target
# target_composition = np.array([0.261, 0.199, 0.539])  # Replace with your desired target
start_time = 365 * 9
end_time = 365 * 10

# --- Slice and average over the last year ---
biomass = KsZ_mort_scan_X.Phytoplankton__biomass.isel(time=slice(start_time, end_time))
biomass_mean = biomass.mean(dim='time')

# Transpose so dimensions are (HigherOrderMortality__rate, GGE__gge, phyto)
biomass_mean = biomass_mean.transpose('HigherOrderMortality__rate', 'N0__value', 'phyto')

# --- Normalize each (i,j) point to relative composition ---
total_biomass = biomass_mean.sum(dim='phyto')
relative_composition = biomass_mean / total_biomass
relative_composition = relative_composition.fillna(0)

# --- Compute distance to target composition ---
rel_np = relative_composition.values  # shape: (rate, gge, phyto)
dist = np.linalg.norm(rel_np - target_composition, axis=2)  # shape: (rate, gge)
print(dist.min())
# Normalize and compute brightness
dist_max = np.percentile(dist, 95)
dist_clipped = np.clip(dist, 0, dist_max)
dist_norm = dist_clipped / dist_max
brightness = 1 - dist_norm

# Apply gamma correction
gamma = 0.5
brightness_gamma = brightness ** gamma
rgb = rel_np * brightness_gamma[..., np.newaxis]

# Clean up RGB array for imshow
rgb = np.nan_to_num(rgb, nan=0.0)
rgb = np.clip(rgb, 0, 1)

# --- Prepare coordinate grids for contour ---
x_vals = biomass_mean.N0__value.values  # Horizontal axis
y_vals = biomass_mean.HigherOrderMortality__rate.values  # Vertical axis
X, Y = np.meshgrid(x_vals, y_vals)

# --- Plot ---
plt.figure(figsize=(10, 6))
plt.imshow(rgb, origin='lower', aspect='auto',
           extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()])
plt.xlabel('N0__value')
plt.ylabel('HigherOrderMortality__rate')
plt.title('Phytoplankton Composition Fit\nRGB = Composition, Brightness = Fit Quality')
plt.grid(False)

# Overlay contour for perfect match:
# Create your contour levels
levels = [0.02, 0.05, 0.1, 0.2]

# Define a colormap that goes from grey to white
cmap = mcolors.LinearSegmentedColormap.from_list("grey_white", ["grey", "white"], N=256)

# Create the contour plot
cs = plt.contour(X, Y, dist, levels=levels, colors=cmap(np.linspace(1, 0, len(levels))), linewidths=1.5)

# Optionally, add labels or a colorbar to the plot
#plt.colorbar(cs)



legend_elements = [
    Patch(facecolor='red', label='Phyto 1 dominates'),
    Patch(facecolor='green', label='Phyto 2 dominates'),
    Patch(facecolor='blue', label='Phyto 3 dominates'),
    Patch(facecolor='white', label='Contour shows good match to target'),
    Patch(facecolor='black', label='Poor match or unstable')
]
plt.legend(handles=legend_elements, loc='upper left')

plt.savefig("ParScan_09_c2_detailed.pdf", format='pdf', bbox_inches='tight')
plt.show()
