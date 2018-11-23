import utilities
import argparse
import molecules as database
import matplotlib.pyplot as plt

import GA as ga

# TODO commandline argument for molecules



def main(data):

    molecule_meta = utilities.MoleculeMetaData(*data)

    print("--> Initial Geomety:\n")
    print(utilities.create_z_matrix(
        molecule_meta.first_atom,
        molecule_meta.second_atom,
        molecule_meta.third_atom,
        molecule_meta.species,
        molecule_meta.genome
    ))

    params = {
        "n_population": 250,
        "noise_initial_population": 0.8,
        "threshold_energy_difference": 1e-8,
        "n_generations": 300,
        "tournament_size": 5,
        "probability_crossing": 0.3,
        "probability_mutation": 0.2,
        "individual_gene_probability_mutation": 0.1,
        "noise_mutation": 0.5,
        "fitness_callback": lambda x: utilities.lennard_jones_energy(x, 1, 1)
    }

    print("--> Starting Optimization:\n")
    counter = ga.find_best_geometry(molecule_meta, params)

    print("--> E_min = {0} found {1} times".format(counter.E_min, counter.count))

    print("--> Final geometry: \n")
    print("zMatrix:")
    print(utilities.create_z_matrix(
        molecule_meta.first_atom,
        molecule_meta.second_atom,
        molecule_meta.third_atom,
        molecule_meta.species,
        counter.best_genome
    ))
    print("\nCarthesian:\n")

    utilities.plot_genome(counter.best_genome, molecule_meta)
    plt.show()
    
if __name__ == '__main__':

    main(database.Be13)

