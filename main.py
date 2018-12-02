"""This is the main script, used to start and controll the algorithm. 

Author:
 - Johannes Cartus, TU Graz
"""

import utilities, energy
import argparse
import molecules as database
import matplotlib.pyplot as plt

import GA as ga

# TODO commandline argument for molecules



def main(data):

    molecule_meta = utilities.MoleculeMetaData(*data)

    
    params = ga.Params(
        number_of_generations=30,
        size_of_population=300,
        noise_initial_population=2.0,
        probability_crossing=0.5,
        probability_mutation=0.5,
        individual_gene_probability_mutation=0.5,
        noise_mutation=1.0,
        fitness_callback=lambda x: energy.lennard_jones_energy(x, 5, 1.1)
        #fitness_callback=energy.rhf_energy
        #fitness_callback=energy.uhf_energy
    )

    algo = ga.GAStructureOptimisation(params)

    counter = algo.run(molecule_meta)

    utilities.plot_genome(counter.best_genome, molecule_meta)
    #plt.show()
    
if __name__ == '__main__':

    main(database.Be13)

