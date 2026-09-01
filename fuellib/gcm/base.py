"""Abstract base class for Group Contribution Method (GCM) implementations."""

import os
from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
import pandas as pd


class GCMMethod(ABC):
    """
    Abstract base class for group contribution method (GCM) implementations.

    Each concrete GCM (e.g. Constantinou-Gani, UNIFAC) loads its own functional
    group decomposition matrix (Nij) from a group decomposition file and
    computes a distinct set of per-compound properties from it. Subclasses
    declare which properties they provide via ``provided_properties`` and
    implement :meth:`compute` to calculate them. Computed properties are
    accessible as attributes on the instance (e.g. ``gani.Tc``) or via
    :meth:`get`.
    """

    #: Unique registry name for this GCM (e.g. "gani")
    name: ClassVar[str]

    #: Names of properties this GCM computes (e.g. ["Tc", "Pc", ...])
    provided_properties: ClassVar[list]

    def __init__(self, fuelDataDecompDir, decompName):
        """
        Load the functional group decomposition matrix and compute properties.

        :param fuelDataDecompDir: Directory containing functional group decomposition files.
        :type fuelDataDecompDir: str
        :param decompName: Name of the group decomposition file, without extension.
        :type decompName: str
        """
        self.groupDecompFile = os.path.join(fuelDataDecompDir, f"{decompName}.csv")
        df_Nij = pd.read_csv(self.groupDecompFile)
        self.Nij = df_Nij.iloc[:, 1:].to_numpy()
        self.num_compounds = self.Nij.shape[0]
        self.num_groups = self.Nij.shape[1]
        self._properties = self.compute(self.Nij)

    @abstractmethod
    def compute(self, Nij) -> dict:
        """
        Compute all properties provided by this GCM.

        :param Nij: Functional group decomposition matrix, shape (num_compounds, num_groups).
        :type Nij: np.ndarray
        :return: Dictionary mapping property name to computed np.ndarray.
        :rtype: dict
        """
        raise NotImplementedError

    def get(self, property_name: str) -> np.ndarray:
        """
        Retrieve a computed property by name.

        :param property_name: Name of the property to retrieve.
        :type property_name: str
        :return: Computed property array.
        :rtype: np.ndarray
        :raises KeyError: If this GCM does not provide the requested property.
        """
        if property_name not in self._properties:
            raise KeyError(
                f"GCM '{self.name}' does not provide property '{property_name}'."
            )
        return self._properties[property_name]

    def __getattr__(self, item):
        """Allow attribute-style access to computed properties, e.g. ``gani.Tc``."""
        properties = self.__dict__.get("_properties", {})
        if item in properties:
            return properties[item]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")


__all__ = ["GCMMethod"]
