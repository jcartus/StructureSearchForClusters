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


class MinEnergyStateCounter(object):

    def __init__(self, delta):

        self.delta = delta
        self.E_min = 1e10
        self.count = 0

    def update(self, energies):

        if np.abs(np.min(energies) - self.E_min) < self.delta:
            self.count += 1
        else:
            self.E_min = np.min(energies)
            self.count = 1


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


def plot_data(distribution):
    bins = np.asarray(distribution.bins)
    x = (bins[:-1] + bins[1:]) / 2
    #x = bins[:-1]

    plt.bar(x, distribution.energies, width=0.8*distribution.energy_diff)
    plt.xlabel("Energies / E_h")
    plt.ylabel("Absolute frequency / 1")

class Distribution(object):

    def __init__(self, energies, number_on_bins):

        self.offset = np.min(energies)

        self.energies, self.bins = self.initial_histogram(
            energies - self.offset,
            number_on_bins
        )

        self.energy_diff = np.mean(np.diff(self.bins - self.offset))

    def initial_histogram(self, energies, number_on_bins):

        return np.histogram(
            np.array(energies).flatten(), 
            bins=number_on_bins, 
        )

    def update(self, energies):

        for e in (np.array(energies) - self.offset):
            
            #--- addd missing bins ---
            if e < self.bins[0]:

                n_missing_bins = np.ceil((self.bins[0] - e) / self.energy_diff)

                missing_bins = \
                    np.arange(n_missing_bins, 0, -1) * self.energy_diff + self.bins[0]

                self.bins = np.concatenate([missing_bins, self.bins])
                self.energies = np.concatenate([self.energies, np.zeros(len(missing_bins))])

            elif e > self.bins[-1]:

                n_missing_bins = np.ceil((-self.bins[-1] + e) / self.energy_diff)

                missing_bins = \
                    np.arange(n_missing_bins + 1) * self.energy_diff + self.bins[-1]

                self.bins = np.concatenate([missing_bins, self.bins])
                self.energies = np.concatenate([np.zeros(len(missing_bins)), self.energies])

            #---


            # if in nth bin, n bin boundaries will be smaller, thus index=n-1
            self.energies[np.sum(e > self.bins) - 1] += 1
            



    