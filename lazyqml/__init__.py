"""Top-level package for lazyqml."""

__author__ = """Diego García Vega, Fernando Álvaro Plou Llorente, Alejandro Leal Castaño"""
__email__ = "garciavdiego@uniovi.es, ploufernando@uniovi.es, lealcalejandro@uniovi.es"
__version__ = "0.0.8"

import lazyqml

_simulation = "statevector"

def set_simulation_type(sim):
    try:
        assert sim == "statevector" or sim == "tensor"
        lazyqml._simulation = sim

    except Exception as e:
        raise ValueError(f"Simulation type must be \"statevector\" or \"tensor\". Got \"{sim}\"")
    

def get_simulation_type():
    return lazyqml._simulation

# Max bond dimension getter/setter
_max_bond_dim = 64

def set_max_bond_dim(dim: int):
    lazyqml._max_bond_dim = dim

def get_max_bond_dim():
    return lazyqml._max_bond_dim