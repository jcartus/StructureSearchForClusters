import numpy as np
import matplotlib.pyplot as plt

from pyscf import gto, scf
import random

plt.style.use("seaborn")



class MoleculeMetaData(object):
    def __init__(self, zMatrix):
        self.first_atom = None
        self.second_atom = None
        self.third_atom = None
        
        self.species = []
        self.genome = []

        self.calculate_genome(zMatrix)
            
    def calculate_genome(self, matrix_str):
        
        lines =[line for line in matrix_str.split("\n") if line]

        if len(lines) < 4:
            raise ValueError("Molecule must have 4 atoms at least!")
        
        # first line
        self.first_atom = lines[0].split()[0]
        
        # second line
        splits = lines[1].split()
        self.second_atom = splits[0]
        self.genome.append(float(splits[2]))
        
        # third line
        splits = lines[2].split()
        self.third_atom = splits[0]
        self.genome.append(float(splits[2]))
        self.genome.append(float(splits[4]))


        for line in lines[3:]:
            split = line.split()

            self.species.append((split[0], [split[1], split[3], split[5]]))
            self.genome += [float(split[2]), float(split[4]), float(split[6])]


def create_z_matrix(first_atom, second_atom, third_atom, species, genome):
    """Create the z-matrix string for a genome"""
    matrix_str = first_atom + "\n"
    matrix_str += second_atom + " 1 " + str(genome[0]) + "\n"
    matrix_str += third_atom + " 1 " + str(genome[1]) + " 2 " + str(genome[2]) + "\n"
    
    i = 3
    for (species, reference) in species:
        
        matrix_str += " ".join(
            [
                species, 
                reference[0], 
                str(genome[i]), 
                reference[1], 
                str(genome[i + 1]), 
                reference[2], 
                str(genome[i + 2])
            ]
        ) + "\n"
        
        i += 3
        
    return matrix_str

def build_molecule_from_genome(genome, meta):
    """From the genome create a pyscf molecule for energy calculation"""
    
    mol = gto.Mole()
    mol.atom = create_z_matrix(
        meta.first_atom,
        meta.second_atom,
        meta.third_atom,
        meta.species,
        genome
    )
    
    mol.build()
    
    return mol

def init_individual(meta):
    """Initialize an individual of the population by adding noise to the 
    genome of the guess structure"""
    return [gene + random.gauss(0, 0.1 * gene) for gene in meta.genome]


def evaluateFitness(individual, meta):
    """Calculate the energy of an electron. This will be used as fitness
    function for the GA."""

    mol = build_molecule_from_genome(individual, meta)
    
    mf = scf.UHF(mol)
    mf.verbose = 0
    E = mf.scf()
    
    # has to be a tuple (because of syntax)
    return E,

def calculate_fitness_distribution(energies, bins):
    """Calculate how often energies appear (histogram)"""

    fitness_distribution = np.histogram(
        np.array(energies).flatten(), 
        bins=bins, 
    )[0]

    return fitness_distribution


def plot_data(energies, bins):
    bins = np.asarray(bins)
    x = (bins[:-1] + bins[1:]) / 2
    #x = bins[:-1]

    plt.bar(x, energies, width=0.5)
    plt.xlabel("Energies / E_h")
    plt.ylabel("Absolute frequency / 1")
