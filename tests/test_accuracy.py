import os
import numpy as np
import pandas as pd
import test_functions as fxns
import unittest


class CompTestCase(unittest.TestCase):
    """Test that accuracy is preserved"""

    def test_accuracy(self):
        """Does the PR impact the accuracy of single component fuel predictions?"""
        test_dir = os.path.dirname(__file__)
        baseline_dir = os.path.join(test_dir, "baselinePredictions")

        # Set the percentage for variation in max error
        max_error_diff = 1e-6

        # Fuels to test
        fuel_names = [
            "heptane",
            "decane",
            "dodecane",
            "posf10264",
            "posf10325",
            "posf10289",
        ]

        # Properties to test
        prop_names = [
            "Density",
            "Viscosity",
            "VaporPressure",
            "SurfaceTension",
            "ThermalConductivity",
        ]

        # Compare to NIST predictions and previous model predictions
        for fuel_name in fuel_names:

            baseline_file = os.path.join(baseline_dir, f"{fuel_name}.csv")
            df_base = pd.read_csv(baseline_file, skiprows=[1])

            for prop in prop_names:
                # Baseline error
                err_base = df_base[f"Error_{prop}"].dropna().to_numpy()

                # Cumulative baseline error
                sum_err_base = np.sum(err_base)

                # Allow cumulative error within max_error_diff
                sum_err_base += sum_err_base * max_error_diff

                # Get predictions for current model
                T, data, pred = fxns.getPredAndData(fuel_name, prop)
                err = np.abs(data - pred)
                sum_err = np.sum(err)

                np.testing.assert_array_less(
                    sum_err,
                    sum_err_base,
                    err_msg=f"Error: Prediction of {prop} for {fuel_name} is less accurate than previous model!",
                )


if __name__ == "__main__":
    unittest.main()
