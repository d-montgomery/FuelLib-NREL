import os
import sys
import numpy as np

# Add the FuelLib directory to the Python path
fuellib_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(fuellib_dir)
import FuelLib as fl

# Create a groupContribution object for the fuel "heptane-decane"
fuel = fl.groupContribution("heptane-decane")

# Display fuel name, components, initial composition, and critical temperature
print(f"Fuel name: {fuel.name}")
print(f"Fuel components: {fuel.compounds}")
print(f"Initial composition: {fuel.Y_0}")
print(f"Critical temperature: {fuel.Tc} K")

# Calculate the saturated vapor pressure at 320 K
T = 320  # K
p_sat_i = fuel.psat(T)
p_sat_mix = fuel.mixture_vapor_pressure(fuel.Y_0, T)
print(f"Saturated vapor pressure at {T} K: {p_sat_i} Pa")
print(f"Mixture saturated vapor pressure at {T} K: {p_sat_mix:.2f} Pa")
