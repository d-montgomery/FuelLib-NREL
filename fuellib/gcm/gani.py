"""Constantinou-Gani group contribution method implementation."""

import os
from typing import ClassVar

import numpy as np
import pandas as pd

from .base import GCMMethod
from .registry import register_gcm


class GaniGCM(GCMMethod):
    """
    Constantinou-Gani group contribution method.

    Computes critical/thermodynamic properties (MW, Tc, Pc, Vc, Tb, Tm, Hf, Gf,
    Hv_stp, Lv_stp, Cp_stp, Cp_B, Cp_C, Vm_stp, omega) and hydrocarbon
    classification (hc_type, fam, nC, nH) from a functional group decomposition
    matrix, using group contribution coefficients and metadata from the
    built-in gani.csv table.
    """

    name = "gani"
    provided_properties: ClassVar[list] = [
        "MW",
        "Tc",
        "Pc",
        "Vc",
        "Tb",
        "Tm",
        "Hf",
        "Gf",
        "Hv_stp",
        "Lv_stp",
        "Cp_stp",
        "Cp_B",
        "Cp_C",
        "Vm_stp",
        "omega",
        "hc_type",
        "fam",
        "nC",
        "nH",
    ]

    #: Numeric family code lookup keyed by hc_type; anything not listed -> 0
    _FAM_MAP: ClassVar[dict] = {
        "n-alkane": 0,
        "aromatic": 1,
        "cyclo-alkane": 2,
        "alkene": 3,
    }

    def __init__(self, fuelDataDecompDir, decompName, gcmtable_dir):
        """
        Load the Constantinou-Gani group table and compute properties.

        :param fuelDataDecompDir: Directory containing functional group decomposition files.
        :type fuelDataDecompDir: str
        :param decompName: Name of the group decomposition file, without extension.
        :type decompName: str
        :param gcmtable_dir: Directory containing the gani.csv GCM table.
        :type gcmtable_dir: str
        """
        self.gcmTableFile = os.path.join(gcmtable_dir, "gani.csv")
        df_table = pd.read_csv(self.gcmTableFile)
        df_table = df_table.drop(columns=["Units"])

        # Extract the "order" and "type" metadata rows (describe each group
        # column) before dropping them from the numeric coefficient table.
        order_row = df_table[df_table["Property"] == "order"].iloc[0, 1:]
        type_row = df_table[df_table["Property"] == "type"].iloc[0, 1:]
        self._order_row = order_row
        self._type_row = type_row.fillna("")

        # Number of first- and second-order groups, derived from the "order"
        # row instead of hardcoded constants.
        order_numeric = pd.to_numeric(order_row, errors="coerce")
        self.N_g1 = int((order_numeric == 1).sum())
        self.N_g2 = int((order_numeric == 2).sum())

        # Exclude the metadata rows and coerce the remaining coefficient
        # columns to numeric. This is necessary because pandas infers dtype
        # per-column: a metadata row's string values would otherwise force an
        # entire column to object/string dtype.
        metadata_properties = ["order", "type"]
        df_table = df_table[~df_table["Property"].isin(metadata_properties)]
        coefficient_columns = df_table.columns[1:]
        df_table[coefficient_columns] = df_table[coefficient_columns].apply(
            pd.to_numeric
        )
        self._df_table = df_table

        super().__init__(fuelDataDecompDir, decompName)

    def _get_row(self, property_name: str) -> np.ndarray:
        """
        Get a coefficient row from the GCM table.

        :param property_name: Name of the property to retrieve.
        :type property_name: str
        :return: Property values for all functional groups.
        :rtype: np.ndarray
        :raises ValueError: If property not found in GCM table.
        """
        row = self._df_table[self._df_table["Property"] == property_name]
        if row.empty:
            raise ValueError(f"Property '{property_name}' not found in GCM table.")
        return row.iloc[:, 1:].to_numpy().flatten()

    def compute(self, Nij) -> dict:
        """
        Compute all Constantinou-Gani properties for the given decomposition matrix.

        :param Nij: Functional group decomposition matrix, shape (num_compounds, num_groups).
        :type Nij: np.ndarray
        :return: Dictionary mapping property name to computed np.ndarray.
        :rtype: dict
        """
        num_compounds = Nij.shape[0]

        # Table data for functional groups (num_compounds,)
        Tck = self._get_row("tck")  # critical temperature (1)
        Pck = self._get_row("pck")  # critical pressure (bar)
        Vck = self._get_row("vck")  # critical volume (m^3/kmol)
        Tbk = self._get_row("tbk")  # boiling temperature (1)
        Tmk = self._get_row("tmk")  # melting point temperature (1)
        hfk = self._get_row("hfk")  # enthalpy of formation, (kJ/mol)
        gfk = self._get_row("gfk")  # Gibbs energy (kJ/mol)
        hvk = self._get_row("hvk")  # latent heat of vaporization (kJ/mol)
        wk = self._get_row("wk")  # accentric factor (1)
        Vmk = self._get_row("vmk")  # liquid molar volume fraction (m^3/kmol)
        cpak = self._get_row("CpAk")  # specific heat values (J/mol/K)
        cpbk = self._get_row("CpBk")  # specific heat values (J/mol/K)
        cpck = self._get_row("CpCk")  # specific heat values (J/mol/K)
        mwk = self._get_row("MW")  # molecular weights (g/mol)

        # --- Compute critical properties at standard temp (num_compounds,)
        # Molecular weights
        MW = np.matmul(Nij, mwk)  # g/mol
        MW *= 1e-3  # Convert to kg/mol

        # T_c (critical temperature)
        Tc = 181.128 * np.log(np.matmul(Nij, Tck))  # K

        # p_c (critical pressure)
        Pc = 1.3705 + (np.matmul(Nij, Pck) + 0.10022) ** (-2)  # bar
        Pc *= 1e5  # Convert to Pa from bar

        # V_c (critical volume)
        Vc = -0.00435 + (np.matmul(Nij, Vck))  # m^3/kmol
        Vc *= 1e-3  # Convert to m^3/mol

        # T_b (boiling temperature)
        Tb = 204.359 * np.log(np.matmul(Nij, Tbk))  # K

        # T_m (melting temperature)
        Tm = 102.425 * np.log(np.matmul(Nij, Tmk))  # K

        # H_f (enthalpy of formation)
        Hf = 10.835 + np.matmul(Nij, hfk)  # kJ/mol
        Hf *= 1e3  # Convert to J/mol

        # G_f (Gibbs free energy)
        Gf = -14.828 + np.matmul(Nij, gfk)  # kJ/mol
        Gf *= 1e3  # Convert to J/mol

        # H_v,stp (enthalpy of vaporization at 298 K)
        Hv_stp = 6.829 + (np.matmul(Nij, hvk))  # kJ/mol
        Hv_stp *= 1e3  # Convert to J/mol

        # omega (accentric factor)
        omega = 0.4085 * np.log(np.matmul(Nij, wk) + 1.1507) ** (1.0 / 0.5050)

        # V_m (molar liquid volume at 298 K)
        Vm_stp = 0.01211 + np.matmul(Nij, Vmk)  # m^3/kmol
        Vm_stp *= 1e-3  # Convert to m^3/mol

        # C_p,stp (molar specific heat at 298 K)
        Cp_stp = np.matmul(Nij, cpak) - 19.7779  # J/mol/K

        # Temperature corrections for C_p
        Cp_B = np.matmul(Nij, cpbk)
        Cp_C = np.matmul(Nij, cpck)

        # L_v,stp (latent heat of vaporization at 298 K)
        Lv_stp = Hv_stp / MW  # J/kg

        hc_type, fam = self._classify_hydrocarbons(Nij, num_compounds)
        nC, nH = self._compute_carbon_hydrogen_counts(Nij, num_compounds)

        return {
            "MW": MW,
            "Tc": Tc,
            "Pc": Pc,
            "Vc": Vc,
            "Tb": Tb,
            "Tm": Tm,
            "Hf": Hf,
            "Gf": Gf,
            "Hv_stp": Hv_stp,
            "Lv_stp": Lv_stp,
            "Cp_stp": Cp_stp,
            "Cp_B": Cp_B,
            "Cp_C": Cp_C,
            "Vm_stp": Vm_stp,
            "omega": omega,
            "hc_type": hc_type,
            "fam": fam,
            "nC": nC,
            "nH": nH,
        }

    def _classify_hydrocarbons(self, Nij, num_compounds):
        """
        Classify each compound's hydrocarbon type and family code.

        Uses the "type" metadata row from gani.csv to build boolean masks over
        the group columns (aromatic, cycloalkane, alkene, branched, alkane),
        rather than hardcoded group index ranges. A compound with none of
        these group types present is classified as "not-hydrocarbon".

        :param Nij: Functional group decomposition matrix.
        :type Nij: np.ndarray
        :param num_compounds: Number of compounds.
        :type num_compounds: int
        :return: (hc_type, fam) arrays.
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        type_values = self._type_row.to_numpy()
        aromatic_mask = type_values == "aromatic"
        cycloalkane_mask = type_values == "cycloalkane"
        alkene_mask = type_values == "alkene"
        branched_mask = type_values == "branched"
        alkane_mask = type_values == "alkane"

        hc_type = np.array([""] * num_compounds, dtype=object)
        fam = np.zeros(num_compounds, dtype=int)

        for i in range(num_compounds):
            # Check if aromatic: does it contain AC's?
            if np.sum(Nij[i, aromatic_mask]) > 0:
                hc_type[i] = "aromatic"
            # Check if cycloparaffin: does it contain rings?
            elif np.sum(Nij[i, cycloalkane_mask]) > 0:
                hc_type[i] = "cyclo-alkane"
            # Check if olefin: does it contain double bonds?
            elif np.sum(Nij[i, alkene_mask]) > 0:
                hc_type[i] = "alkene"
            # Check for branching groups (CH, C quaternary carbons)
            elif np.sum(Nij[i, branched_mask]) > 0:
                hc_type[i] = "iso-alkane"
            # Only CH3 and CH2 -> n-alkane (linear)
            elif np.sum(Nij[i, alkane_mask]) > 0:
                hc_type[i] = "n-alkane"
            else:
                hc_type[i] = "not-hydrocarbon"
            fam[i] = self._FAM_MAP.get(hc_type[i], 0)

        return hc_type, fam

    def _compute_carbon_hydrogen_counts(self, Nij, num_compounds):
        """
        Compute carbon and hydrogen counts from first-order group decomposition.

        For jet fuels, use only alkyl (groups 0-3) and aromatic (groups 10-14)
        groups. Alkyl: CH3=1C,3H; CH2=1C,2H; CH=1C,1H; C=1C,0H. Aromatic:
        ACH=1C,1H; AC=1C,0H; ACCH3=2C,3H; ACCH2=2C,2H; ACCH=2C,1H.

        :param Nij: Functional group decomposition matrix.
        :type Nij: np.ndarray
        :param num_compounds: Number of compounds.
        :type num_compounds: int
        :return: (nC, nH) arrays.
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        alkyl_carbons = np.array([1, 1, 1, 1])  # groups 0-3
        alkyl_hydrogens = np.array([3, 2, 1, 0])
        # Olefinic: group 4 appears to represent 2 carbons with 3 hydrogens in UNIFAC-based system
        olefinic_carbons = np.array([2, 1, 1, 0, 0, 0])  # groups 4-9
        olefinic_hydrogens = np.array([3, 1, 0, 0, 0, 0])
        aromatic_carbons = np.array([1, 1, 2, 2, 2])  # groups 10-14
        aromatic_hydrogens = np.array([1, 0, 3, 2, 1])

        nC = np.zeros(num_compounds, dtype=float)
        nH = np.zeros(num_compounds, dtype=float)
        for i in range(num_compounds):
            # Alkyl contribution (groups 0-3)
            nC[i] = np.dot(Nij[i, 0:4], alkyl_carbons)
            nH[i] = np.dot(Nij[i, 0:4], alkyl_hydrogens)
            # Olefinic contribution (groups 4-9)
            nC[i] += np.dot(Nij[i, 4:10], olefinic_carbons)
            nH[i] += np.dot(Nij[i, 4:10], olefinic_hydrogens)
            # Aromatic contribution (groups 10-14)
            nC[i] += np.dot(Nij[i, 10:15], aromatic_carbons)
            nH[i] += np.dot(Nij[i, 10:15], aromatic_hydrogens)

        return nC, nH


register_gcm("gani", GaniGCM)

__all__ = ["GaniGCM"]
