import utilities
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
        noise_initial_population=0.8,
        probability_mutation=0.5,
        individual_gene_probability_mutation=0.1,
        noise_mutation=0.2,
        #fitness_callback=lambda x: utilities.lennard_jones_energy(x, 5, 1.5)
        #fitness_callback=utilities.rhf_energy
        fitness_callback=utilities.uhf_energy
    )

    algo = ga.GAStructureOptimisation(params)

    counter = algo.run(molecule_meta)

    utilities.plot_genome(counter.best_genome, molecule_meta)
    #plt.show()
    
if __name__ == '__main__':

    main(database.Ethen)

