import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
import re

# Add the FuelLib directory to the Python path
FUELLIB_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(FUELLIB_DIR)
from paths import *
import FuelLib as fl
import colormap as cscmap

# Fuel name
fuel_name = "DLR_SurrogateCompounds"

# Save the figures?
saveFig = False
normalize = False

# The following property data is available:
# density, boiling_point, vapor_pressure, enthalpy_of_vaporization, viscosity, molar_volume
# append '_NIST' or '_CS' to switch between NIST data or ChemSpider preditions (ACD/Labs - PhysChem)
prop_name = [
    "boiling_point_NIST",
    "critical_temp_NIST",
    "critical_pressure_NIST",
    "vapor_pressure_NIST",
    "density_NIST",
    "enthalpy_of_vaporization_NIST",
    "viscosity_NIST",
]

# Standard temp and pressure
T_stp = 298.15  # K
T_mu = 298.15  # K
p_stp = 101325  # Pa or 1 atm

# Plotting parameters
line_w = 1.5  # Adjustable line width
box_size = 6  # Adjustable box size for marker size
hash_length = 0.045  # Adjust this to control the length of the hashes

# Initialize GCM object
cmps = fl.fuel(fuel_name)

# NIST Data for each compound
propsDataFile = os.path.join(FUELDATA_PROPS_DIR, f"{fuel_name}.csv")

# Read the property data, ignore row with units
df_props = pd.read_csv(propsDataFile, skiprows=[1])
df_props.fillna(np.nan, inplace=True)  # Replace missing data with nan's

# Get number of compounds and initialize lists for properties
num_compounds = cmps.num_compounds

# Calculate properties for each compound
T_b = cmps.Tb  # K
T_c = cmps.Tc  # K
p_c = cmps.Pc  # Pa
p_v = cmps.psat(T_stp)  # Pa
rho = cmps.density(T_stp)  # kg/m^3
Hv_stp = cmps.Hv_stp  # J/mol
Vm = cmps.molar_liquid_vol(T_stp)  # m^3/mol
mu = cmps.viscosity_dynamic(T_mu)  # Pa*s


# Plot predicted versus data
def make_plot(compounds, compound_props, estimate, prop_name, xlab, ylab, scale=0):
    # x-locations for each compound in plot
    x = [k for k in range(compounds.num_compounds)]

    uncertainty = prop_name + "_uncertainty"

    # Define a colormap
    cmap = colormaps.get_cmap("tab10")
    cmap = cscmap.get_pastel()

    if normalize:
        # Set normalizing constant to the median of the property data
        norm_const = compound_props[prop_name].median()

    # Plot exact with uncertainty error bars
    for k in range(compounds.num_compounds):
        color_k = cmap(k % cmap.N)  # Get a consistent color for each compound
        prop_val = compound_props[prop_name][k]
        uncertainty_val = compound_props[uncertainty][k]
        prediction = estimate[k]

        # Normalize the data?
        if normalize:
            prop_val /= norm_const
            uncertainty_val /= norm_const
            prediction /= norm_const

        # If uncertainty is precent plot bars
        if not np.isnan(uncertainty_val):
            hi = prop_val + uncertainty_val
            lo = prop_val - uncertainty_val

            plt.plot([x[k], x[k]], [lo, hi], linewidth=line_w, color=color_k)

        # Plot data
        plt.plot(x[k], prop_val, marker="D", markersize=box_size, color=color_k)

        # Plot estimate from GCM
        plt.scatter(
            x[k],
            prediction,
            label=compound_props.name[k],
            color=color_k,
            facecolor="none",
            linewidth=line_w,
        )

    if normalize:
        ylab = re.sub(r"\[.*?\]", "", ylab).strip()
        ylab = f"Normalized {ylab}"
    plt.ylabel(ylab, fontsize=18)
    if scale == 1:
        plt.yscale("log")
    title_text = "GCM Predictions"
    if "_NIST" in prop_name:
        title_text += " vs. NIST Predictions"
    elif "_CS" in prop_name:
        title_text += " vs. Measurement data from ChemSpider"
    plt.legend(
        title=title_text,
        bbox_to_anchor=(1.05, 1.0),
        loc="upper left",
        fontsize=12,
        title_fontsize=12,
    )
    plt.tight_layout()


# Specify variable and y_label for plotting
prop_name_2_plots = {
    "boiling_point_NIST": (T_b, "$T_b$ [K]"),
    "boiling_point_CS": (fl.K2C(T_b), "$T_b$ [°C]"),
    "critical_temp_NIST": (T_c, "$T_c$ [K]"),
    "critical_pressure_NIST": (p_c * 1e-3, "$p_c$ [kPa]"),
    "vapor_pressure_CS": (p_v / 133.322, "$p_v$ at 25°C [mmHg]"),
    "vapor_pressure_NIST": (p_v * 1e-3, "$p_v$ at 25°C [kPa]"),
    "density_NIST": (rho, r"$\rho$ at 25°C [kg/m$^3$]"),
    "density_CS": (rho * 1e-3, r"$\rho$ at 25°C [g/cm$^3$]"),
    "enthalpy_of_vaporization_NIST": (Hv_stp * 1e-3, r"$H_{v}$ at 25°C [kJ/mol]"),
    "enthalpy_of_vaporization_CS": (Hv_stp * 1e-3, r"$H_{v}$ at 25°C [kJ/mol]"),
    "molar_volume_CS": (Vm * 1e6, "$V_m$ [cm$^3$/mol]"),
    "viscosity_NIST": (mu, r"$\mu$ at 25°C [Pa$\cdot$s]"),
}

# Set up xticks and labels
xticks = [k for k in range(cmps.num_compounds)]
xlab = []
for k in range(len(df_props.name)):
    xlab.append(df_props.name[k].split()[0])

# Make plots for each property name listed in prop_name
for k in range(len(prop_name)):
    prop, ylab = prop_name_2_plots[prop_name[k]]
    scale = 0
    plt.figure(figsize=(12, 7))
    make_plot(cmps, df_props, prop, prop_name[k], xlab, ylab, scale)
    plt.yticks(fontsize=18)
    plt.xticks(ticks=xticks, labels=xlab, fontsize=18, rotation=290, ha="center")
    plt.tight_layout()
    if saveFig:
        figName = ""
        figDir = os.path.join(FUELLIB_DIR, f"figures/{fuel_name}")
        if normalize:
            figDir = f"{figDir}/normalized_plots"
            figName = "normalized_"
        figName += f"{prop_name[k]}.png"
        if not os.path.exists(figDir):
            os.makedirs(figDir)
        plt.savefig(os.path.join(figDir, figName))

plt.show()
