import numpy as np
import matplotlib.pyplot as plt

from pyscf import gto, scf
import random, time
from datetime import datetime   

from deap import tools

plt.style.use("seaborn")



class MoleculeMetaData(object):
    def __init__(self, basis, zMatrix):
        
        self.basis = basis

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

class Logger(object):
    """Class that displays user massages in a defined format."""

    markers = {
        1: "[ ]",
        2: "[-]",
        3: "[+]"
    }
    indentation_marker = " " * 25

    log_level = 3

    @staticmethod
    def _time():
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": "

    @classmethod
    def _prefix(cls, level):
        return cls.markers[level] + " " + cls._time()

    @classmethod
    def parse(cls, message):
        """Make sure line breaks etc. have proper indentation"""
        
        return message.replace(
            "\n", 
            "\n" + cls.indentation_marker 
        )

    @classmethod
    def log(cls, message, level=1):
        if cls.log_level == 3 or (cls.log_level == 2 and level >= 1) or \
            (cls.log_level == 1 and level >= 2):
            print(cls._prefix(level) + cls.parse(message))

    @classmethod
    def log_zmatrix(cls, comment, meta, genome, level=1):
        """Used to disp a genome in z-matrix form"""
        
        message = comment + "\n\n"
        message += str(create_z_matrix(
            meta.first_atom,
            meta.second_atom,
            meta.third_atom,
            meta.species,
            genome
        ))

        cls.log(message, level)

    @classmethod
    def log_carthesian(cls, comment, geometry, level=1):
        """Used to disp a genome in z-matrix form"""
        
        message = comment + "\n\n"
        for key, positions in geometry.items():
            for pos in positions:
                message += "{:2s} {:3.3f} {:3.3f} {:3.3f}\n".format(
                    key, 
                    *pos
                )
        
        cls.log(message, level)

    @classmethod
    def log_params(cls, comment, params, level):
        msg = comment + "\n\n"
        for key, value in params.__dict__.items():
            msg += "{0}: {1}\n".format(key, value)

        cls.log(msg, level)

class Result(object):

    def __init__(self, delta):

        self.delta = delta
        self.E_min = 1e10
        self.count = 0

        self.best_genome = None

    def update(self, energies, population):

        if np.abs(np.min(energies) - self.E_min) < self.delta:
            self.count += 1
        else:
            E_old = self.E_min
            self.E_min = np.min(energies)
            self.count = 1
            self.best_genome = tools.selBest(population, 1)[0]


            Logger.log("New Minimum found: {:3.5f} (Diff.: {:1.2e})".format(
                self.E_min,
                E_old - self.E_min
            ), 2)


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

    mol.unit = "Angstrom"
    mol.basis = meta.basis
    
    mol.build()
    
    return mol

def mutate_individual(genome, noise=0.3, gene_mutation_probability=0.3):
    """Add some noise to a genome (and then take the absolute value) 
    to yield a mutated individual. Absolute value is taken to ensure the 
    resulting values are valid for z-matrix.
    Used for mutation and initialisation.
    
    Args:
     - genome <list<float>>: list of genes to be mutated.
     - noise <float>: fraction of gene to used a variance for the noise 
        that is added as mutation.
     - gene_mutation_probability <float>: probability for each individual gene 
        to be mutated.

    Returns:
     - <list<float>>: mutated genome.
    """
    
    mutated_genome = []
    for gene in genome:
        
        if random.random() < gene_mutation_probability: 
            mutated_gene = abs(gene + random.gauss(0, noise * gene))
        else:
            mutated_gene = gene
        
        mutated_genome.append(mutated_gene)

    return mutated_genome

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


def evaluateFitness(individual, meta, fitness_callback=uhf_energy):
    """Calculate the energy of an electron. This will be used as fitness
    function for the GA."""

    mol = build_molecule_from_genome(individual, meta)
    
    try:
        E = fitness_callback(mol)
    except Exception as ex:
        print("Problem during SCF calculation: " +  str(ex))
        E = 1e10


    # has to be a tuple (because of syntax of library)
    return E,

def calculate_fitness_distribution(energies, bins):
    """Calculate how often energies appear (histogram)"""

    fitness_distribution = np.histogram(
        np.array(energies).flatten(), 
        bins=bins, 
    )[0]

    return fitness_distribution

def plot_genome(genome, molecule_meta):
    from mpl_toolkits.mplot3d import Axes3D
    
    #--- calculate carthsian coordinates of final genome ---
    mol = build_molecule_from_genome(genome, molecule_meta)

    # sort atoms by species
    geometries = {}
    for i in range(mol.natm):
        species = mol.atom_pure_symbol(i)
        pos = mol.atom_coord(i)
        
        stored = positions = geometries.get(species, [])
        stored.append(list(pos))
        geometries.update({species: stored})
    #---

    Logger.log_carthesian("Carthesian coordinates:", geometries)
     
    #--- plot ---
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # for python 2 & 3 compatibility
    for (species, positions) in geometries.items():
        
        positions = np.asarray(positions)

        ax.scatter(
            positions[:, 0], 
            positions[:, 1], 
            positions[:, 2], 
            label=species,
            marker="o",
            s=200
        )

    ax.set_xlabel("x axis / bohr")
    ax.set_ylabel("y axis / bohr")
    ax.set_zlabel("z axis / bohr")

    ax.legend()
    ax.set_title("Minimum Geometry")
    #---

def plot_data(distribution):
    bins = np.asarray(distribution.bins)
    x = (bins[:-1] + bins[1:]) / 2
    #x = bins[:-1]

    plt.bar(x, distribution.energies, width=0.8*distribution.energy_diff)
    plt.xlabel("Energies / E_h")
    plt.ylabel("Absolute frequency / 1")


class Timer(object):
    """This class is used to log measure the time it takes the algorithm to 
    finish
    
    Properties:
     - start_time <float>: time in ms since begin of the epoch until the start.
    """

    def __init__(self):

        self.start_time = None

    def start(self):
        """Start the timer"""
        self.start_time = time.time()

    def stop(self):
        """Stop timer and return elapsed time"""
        stop = time.time()
        
        time_elapsed = stop - self.start_time

        self.start_time = None

        return time_elapsed


