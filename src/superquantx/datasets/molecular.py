"""Molecular datasets for quantum chemistry and simulation.

This module provides molecular datasets and utilities for quantum chemistry
applications, including common molecules used in quantum simulation benchmarks.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class Molecule:
    """Represents a molecule for quantum simulation.
    
    Attributes:
        name: Molecule name
        geometry: List of (atom, coordinates) tuples
        charge: Molecular charge
        multiplicity: Spin multiplicity
        basis: Basis set for quantum chemistry calculations
        hamiltonian: Molecular Hamiltonian (if computed)
        n_qubits: Number of qubits needed for simulation
        n_electrons: Number of electrons

    """

    def __init__(
        self,
        name: str,
        geometry: List[Tuple[str, Tuple[float, float, float]]],
        charge: int = 0,
        multiplicity: int = 1,
        basis: str = 'sto-3g'
    ):
        self.name = name
        self.geometry = geometry
        self.charge = charge
        self.multiplicity = multiplicity
        self.basis = basis
        self.hamiltonian = None
        self.n_qubits = None
        self.n_electrons = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert molecule to dictionary representation."""
        return {
            'name': self.name,
            'geometry': self.geometry,
            'charge': self.charge,
            'multiplicity': self.multiplicity,
            'basis': self.basis,
            'n_qubits': self.n_qubits,
            'n_electrons': self.n_electrons
        }


def load_molecule(
    name: str,
    bond_length: Optional[float] = None,
    basis: str = 'sto-3g'
) -> Tuple[Molecule, Dict[str, Any]]:
    """Load a predefined molecule for quantum simulation.
    
    Args:
        name: Molecule name ('H2', 'LiH', 'BeH2', 'H2O', 'NH3', 'CH4')
        bond_length: Custom bond length (if applicable)
        basis: Basis set for quantum chemistry calculations
        
    Returns:
        Tuple of (molecule, metadata)

    """
    molecules = {
        'H2': _create_h2_molecule,
        'LiH': _create_lih_molecule,
        'BeH2': _create_beh2_molecule,
        'H2O': _create_h2o_molecule,
        'NH3': _create_nh3_molecule,
        'CH4': _create_ch4_molecule
    }

    if name not in molecules:
        raise ValueError(f"Unknown molecule: {name}. Available: {list(molecules.keys())}")

    molecule, metadata = molecules[name](bond_length, basis)
    return molecule, metadata


def load_h2_molecule(
    bond_length: float = 0.735,
    basis: str = 'sto-3g'
) -> Tuple[Molecule, Dict[str, Any]]:
    """Load H2 molecule with specified bond length.
    
    Args:
        bond_length: H-H bond length in Angstroms
        basis: Basis set
        
    Returns:
        Tuple of (molecule, metadata)

    """
    return _create_h2_molecule(bond_length, basis)


def load_lih_molecule(
    bond_length: float = 1.595,
    basis: str = 'sto-3g'
) -> Tuple[Molecule, Dict[str, Any]]:
    """Load LiH molecule with specified bond length.
    
    Args:
        bond_length: Li-H bond length in Angstroms
        basis: Basis set
        
    Returns:
        Tuple of (molecule, metadata)

    """
    return _create_lih_molecule(bond_length, basis)


def load_beh2_molecule(
    bond_length: float = 1.326,
    basis: str = 'sto-3g'
) -> Tuple[Molecule, Dict[str, Any]]:
    """Load BeH2 molecule with specified bond length.
    
    Args:
        bond_length: Be-H bond length in Angstroms
        basis: Basis set
        
    Returns:
        Tuple of (molecule, metadata)

    """
    return _create_beh2_molecule(bond_length, basis)


def _create_h2_molecule(bond_length: Optional[float], basis: str) -> Tuple[Molecule, Dict[str, Any]]:
    """Create H2 molecule."""
    if bond_length is None:
        bond_length = 0.735

    geometry = [
        ('H', (0.0, 0.0, 0.0)),
        ('H', (0.0, 0.0, bond_length))
    ]

    molecule = Molecule('H2', geometry, charge=0, multiplicity=1, basis=basis)
    molecule.n_electrons = 2
    molecule.n_qubits = 4  # Minimal basis

    metadata = {
        'molecule_type': 'diatomic',
        'bond_length': bond_length,
        'ground_state_energy': -1.137,  # Approximate
        'typical_applications': ['VQE benchmark', 'quantum simulation'],
        'difficulty': 'easy'
    }

    return molecule, metadata


def _create_lih_molecule(bond_length: Optional[float], basis: str) -> Tuple[Molecule, Dict[str, Any]]:
    """Create LiH molecule."""
    if bond_length is None:
        bond_length = 1.595

    geometry = [
        ('Li', (0.0, 0.0, 0.0)),
        ('H', (0.0, 0.0, bond_length))
    ]

    molecule = Molecule('LiH', geometry, charge=0, multiplicity=1, basis=basis)
    molecule.n_electrons = 4
    molecule.n_qubits = 12  # Typical for STO-3G

    metadata = {
        'molecule_type': 'diatomic',
        'bond_length': bond_length,
        'ground_state_energy': -8.027,  # Approximate
        'typical_applications': ['VQE', 'quantum chemistry'],
        'difficulty': 'medium'
    }

    return molecule, metadata


def _create_beh2_molecule(bond_length: Optional[float], basis: str) -> Tuple[Molecule, Dict[str, Any]]:
    """Create BeH2 molecule."""
    if bond_length is None:
        bond_length = 1.326

    geometry = [
        ('Be', (0.0, 0.0, 0.0)),
        ('H', (0.0, 0.0, bond_length)),
        ('H', (0.0, 0.0, -bond_length))
    ]

    molecule = Molecule('BeH2', geometry, charge=0, multiplicity=1, basis=basis)
    molecule.n_electrons = 6
    molecule.n_qubits = 14  # Typical for STO-3G

    metadata = {
        'molecule_type': 'triatomic',
        'bond_length': bond_length,
        'ground_state_energy': -15.77,  # Approximate
        'typical_applications': ['quantum chemistry', 'benchmarking'],
        'difficulty': 'medium'
    }

    return molecule, metadata


def _create_h2o_molecule(bond_length: Optional[float], basis: str) -> Tuple[Molecule, Dict[str, Any]]:
    """Create H2O molecule."""
    if bond_length is None:
        bond_length = 0.958

    # Water geometry (bent structure)
    angle = 104.5 * np.pi / 180  # HOH angle in radians

    geometry = [
        ('O', (0.0, 0.0, 0.0)),
        ('H', (bond_length * np.sin(angle/2), 0.0, bond_length * np.cos(angle/2))),
        ('H', (-bond_length * np.sin(angle/2), 0.0, bond_length * np.cos(angle/2)))
    ]

    molecule = Molecule('H2O', geometry, charge=0, multiplicity=1, basis=basis)
    molecule.n_electrons = 10
    molecule.n_qubits = 20  # Approximate

    metadata = {
        'molecule_type': 'triatomic',
        'bond_length': bond_length,
        'bond_angle': 104.5,
        'ground_state_energy': -76.0,  # Approximate
        'typical_applications': ['quantum chemistry', 'environmental modeling'],
        'difficulty': 'hard'
    }

    return molecule, metadata


def _create_nh3_molecule(bond_length: Optional[float], basis: str) -> Tuple[Molecule, Dict[str, Any]]:
    """Create NH3 molecule."""
    if bond_length is None:
        bond_length = 1.012

    # Ammonia geometry (trigonal pyramid)
    angle = 106.67 * np.pi / 180  # HNH angle

    geometry = [
        ('N', (0.0, 0.0, 0.0)),
        ('H', (bond_length, 0.0, 0.0)),
        ('H', (-bond_length * np.cos(angle), bond_length * np.sin(angle), 0.0)),
        ('H', (-bond_length * np.cos(angle), -bond_length * np.sin(angle), 0.0))
    ]

    molecule = Molecule('NH3', geometry, charge=0, multiplicity=1, basis=basis)
    molecule.n_electrons = 10
    molecule.n_qubits = 22  # Approximate

    metadata = {
        'molecule_type': 'tetrahedral',
        'bond_length': bond_length,
        'ground_state_energy': -56.2,  # Approximate
        'typical_applications': ['quantum chemistry', 'catalysis modeling'],
        'difficulty': 'hard'
    }

    return molecule, metadata


def _create_ch4_molecule(bond_length: Optional[float], basis: str) -> Tuple[Molecule, Dict[str, Any]]:
    """Create CH4 molecule."""
    if bond_length is None:
        bond_length = 1.087

    # Methane geometry (tetrahedral)
    # Tetrahedral coordinates
    tet_coords = [
        (1, 1, 1),
        (1, -1, -1),
        (-1, 1, -1),
        (-1, -1, 1)
    ]

    geometry = [('C', (0.0, 0.0, 0.0))]
    for x, y, z in tet_coords:
        coord = np.array([x, y, z]) * bond_length / np.sqrt(3)
        geometry.append(('H', tuple(coord)))

    molecule = Molecule('CH4', geometry, charge=0, multiplicity=1, basis=basis)
    molecule.n_electrons = 10
    molecule.n_qubits = 24  # Approximate

    metadata = {
        'molecule_type': 'tetrahedral',
        'bond_length': bond_length,
        'ground_state_energy': -40.2,  # Approximate
        'typical_applications': ['quantum chemistry', 'combustion modeling'],
        'difficulty': 'very_hard'
    }

    return molecule, metadata
