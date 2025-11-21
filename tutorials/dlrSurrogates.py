import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps

# Add the FuelLib directory to the Python path
FUELLIB_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(FUELLIB_DIR)
from paths import *
import FuelLib as fl
import colormap as cscmap

# Save the figures?
saveFig = False
figname = "density_viscosity_vs_DLR.png"

# The following property data is available:
prop_name = ["Viscosity", "Density"]

# Standard temp and pressure
T_stp = 298.15  # K
T_rho = 288.15  # K (15°C)
T_nu = 253.15  # K (-20°C)
p_stp = 101325  # Pa or 1 atm

# Plotting parameters
line_w = 1.5  # Adjustable line width
box_size = 10  # Adjustable box size for marker size
hash_length = 0.045  # Adjust this to control the length of the hashes

# File names
decompFileName = "DLR_SurrogateCompounds"
propsFileName = "DLR_Surrogate.csv"


# Read the property data, ignore row with units
df_props = pd.read_csv(
    os.path.join(FUELDATA_PROPS_DIR, propsFileName), index_col=False, skiprows=[1]
)
df_props.fillna(np.nan, inplace=True)

# Get names of the surrogate fuels and the number of fuels
fuel_names = df_props.Fuel.to_list()
num_fuels = len(fuel_names)

# Get GCM predictions for density and viscosity
gcm_density = []
gcm_viscosity = []
drop = {}
drop["d_0"] = 100 * 1e-6  # initial droplet diameter (m)
drop["r_0"] = drop["d_0"] / 2.0  # initial droplet radius (m)
drop["V_0"] = 4.0 / 3.0 * np.pi * drop["r_0"] ** 3  # initial droplet volume


def drop_mass(fuel, Yi, T):
    return drop["V_0"] / (fuel.molar_liquid_vol(T) @ Yi) * Yi * fuel.MW  # (kg)


for name in fuel_names:
    fuel = fl.fuel("DLR_" + name, decompFileName)
    rho = fuel.mixture_density(fuel.Y_0, T_rho)
    gcm_density.append(rho)

    mu = fuel.mixture_dynamic_viscosity(fuel.Y_0, T_nu)
    nu = mu / rho * 1e6

    mass = drop_mass(fuel, fuel.Y_0, T_nu)
    nu = fuel.mixture_kinematic_viscosity(mass, T_nu)
    nu *= 1e6  # Convert from m^2/s to mm^2/s
    gcm_viscosity.append(nu)

# Set up xticks and labels
xticks = [k for k in range(num_fuels)]
yticks_density = np.linspace(675, 925, 6)
yticks_viscosity = np.linspace(0, 16, 5)

# Plot data
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
ax[0].plot(
    xticks,
    df_props.Density_Model.to_list(),
    "o",
    markersize=box_size,
    color="black",
    label="DLR Model",
)
ax[0].plot(
    xticks,
    df_props.Density_Experiment,
    "s",
    markersize=box_size,
    color="indianred",
    label="Experiment",
)
ax[1].plot(xticks, df_props.Viscosity_Model, "o", markersize=box_size, color="black")
ax[1].plot(
    xticks, df_props.Viscosity_Experiment, "s", markersize=box_size, color="indianred"
)

# Plot Predictions
ax[0].plot(xticks, gcm_density, "*", markersize=box_size, color="teal", label="FuelLib")
ax[1].plot(xticks, gcm_viscosity, "*", markersize=box_size, color="teal")

# Labels, legend, axes etc.
ax[0].set_xticks(xticks, labels=[])
ax[0].set_yticks(ticks=yticks_density)
ax[0].tick_params(axis="y", labelsize=18)
ax[0].set_ylabel(r"kg/m$^3$", fontsize=18)
ax[0].set_title("Density at 15°C", fontsize=18)
ax[1].set_xticks(ticks=xticks)
ax[1].set_xticklabels(fuel_names, fontsize=18, rotation=290, ha="center")
ax[1].set_ylim([0, yticks_viscosity.max()])
ax[1].set_yticks(ticks=yticks_viscosity)
ax[1].set_ylabel(r"mm$^2$/s", fontsize=18)
ax[1].tick_params(axis="y", labelsize=18)
ax[1].set_title("Viscosity at -20°C", fontsize=18)
plt.tight_layout()
plt.subplots_adjust(bottom=0.25)
fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0), ncols=3, fontsize=18)

if saveFig:
    FIG_PATH = os.path.join(FUELLIB_DIR, "figures")
    plt.savefig(os.path.join(FIG_PATH, figname))
plt.show()
