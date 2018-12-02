"""This module contains helper functions and utilities, to 
encode and later decode and display molecular information.

Author:
 - Johannes Cartus, TU Graz
"""

import numpy as np
import matplotlib.pyplot as plt

from pyscf import gto
import random, time
from datetime import datetime   

from deap import tools

plt.style.use("seaborn")



class MoleculeMetaData(object):
    """This class is used to encode information extracted during 
    the creation of the first genome and needed to set other genomes 
    back together to a zmatrix.
    """

    def __init__(self, basis, zMatrix):
        
        self.basis = basis

        self.first_atom = None
        self.second_atom = None
        self.third_atom = None
        
        self.species = []
        self.genome = []

        self.calculate_genome(zMatrix)
            
    def calculate_genome(self, matrix_str):
        """Extracts a genome from a z-matrix string"""


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
        """Write a user message to console."""
        if cls.log_level == 3 or (cls.log_level == 2 and level >= 1) or \
            (cls.log_level == 1 and level >= 2):
            print(cls._prefix(level) + cls.parse(message))

    @classmethod
    def log_zmatrix(cls, comment, meta, genome, level=1):
        """Used to disp a genome in z-matrix form."""
        
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
        """Used to disp a genome in carthesian coordinates."""
        
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
        """Used to disp a GA.Parameters object to console."""
        msg = comment + "\n\n"
        for key, value in params.__dict__.items():
            msg += "{0}: {1}\n".format(key, value)

        cls.log(msg, level)

class Result(object):
    """This class is sued to scan the states found during structure search 
    and to record the best of them and how often they were found.

    Attributes:
        - best_genome <list>: best state found so far
        - E_min <float>: fitness of this best state
        - delta <float>: tolerance, how close state fintess values can be 
        and not be recognized as differing.
    """

    def __init__(self, delta):

        self.delta = delta
        self.E_min = 1e10
        self.count = 0

        self.best_genome = None

    def update(self, energies, population):
        """Checks the population and their fitness values (energies) for a 
        new minimum state. If one is found it is stored
        """

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

def plot_genome(genome, molecule_meta):
    """Display the final results, by logging the coordinates and 
    creating a 3D plot of the result.
    
    Args:
        - genome <list>: found best state.
        - molecule_meta <utilities.MoleMoleculeMetaData>: meta information
    """
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


