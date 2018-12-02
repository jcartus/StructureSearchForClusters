"""This module contains function to calculate an "energy" for a 
molecule. This may be any kind of target function.

Author:
    Johannes Cartus, TU Graz
"""

import numpy as np

from pyscf import scf, dft



def rhf_energy(molecule):
    """Perform unrestricted HF calculation to evalute the energy of a molecule.
    Args:
     - molecule <pyscf.gto.Mole>: molecule to be evaluated.

    Return:
     - <float>: energy of the molecule.
    """

    mf = scf.RHF(molecule)
    mf.verbose = 0
    E = mf.scf()

    return E

def uhf_energy(molecule):
    """Perform unrestricted HF calculation to evalute the energy of a molecule.
    Args:
     - molecule <pyscf.gto.Mole>: molecule to be evaluated.

    Returns:
     - <float>: energy of the molecule.
    """

    mf = scf.UHF(molecule)
    mf.verbose = 0
    E = mf.scf()

    return E

def rks_energy(molecule, xc="LDA"):
    """Perform KS dft with LDA functional to evalute the energy of a molecule.
    
    Args:
     - molecule <pyscf.gto.Mole>: molecule to be evaluated.

    Returns:
     - <float>: energy of the molecule.
    """
    mf = dft.RKS(molecule)
    mf.xc = xc
    E = mf.kernel()

    return E

def uks_energy(molecule, xc="LDA"):
    """Perform unrestricted KS dft with LDA functional to evalute the energy 
    of a molecule.

    Args:
     - molecule <pyscf.gto.Mole>: molecule to be evaluated.

    Returns:
     - <float>: energy of the molecule.
    """
    mf = dft.UKS(molecule)
    mf.xc = xc
    E = mf.kernel()

    return E

def lennard_jones(r, sigma, epsilon):
    """LJ Potential. https://en.wikipedia.org/wiki/Lennard-Jones_potential"""

    V = 4 * epsilon * (
        (sigma / r)**12 - (sigma / r)**6
    )

    return V

def lennard_jones_energy(molecule, sigma, epsilon):
    """Calculate energy of lennard jones potential
    Args:
     - molecule <pyscf.gto.Mole>: molecule to be evaluate.
     - sigma <float>: sigma param of LJ-potential.
     - epsilon <float>: epsilon param of LJ-potential.

    Returns:
     - <float>: energy of molecule
    """

    E = 0
    
    for i in range(molecule.natm):
        for j in range(molecule.natm):

            if j >= i:
                continue
            
            r = np.sqrt(np.sum(
                (
                    np.array(molecule.atom_coord(i)) - \
                    np.array(molecule.atom_coord(j))
                )**2
            ))

            E += lennard_jones(r, sigma, epsilon)

    return E / molecule.natm