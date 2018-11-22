import utilities
import argparse
import molecules as database
import matplotlib.pyplot as plt

import GA as ga

# TODO commandline argument for molecules



def main(data):

    molecule_meta = utilities.MoleculeMetaData(*data)

    print(utilities.create_z_matrix(
        molecule_meta.first_atom,
        molecule_meta.second_atom,
        molecule_meta.third_atom,
        molecule_meta.species,
        molecule_meta.genome
    ))

    params = {
        "n_population": 200,
        "noise_initial_population": 0.5,
        "threshold_energy_difference": 1e-8,
        "n_generations": 1000,
        "probability_crossing": 0.3,
        "probability_mutation": 0.2,
    }

    counter = ga.find_best_geometry(molecule_meta, params)

    print("--> E_min = {0} found {1} times".format(counter.E_min, counter.count))

    print("--> Final geometry: \n")
    print(utilities.create_z_matrix(
        molecule_meta.first_atom,
        molecule_meta.second_atom,
        molecule_meta.third_atom,
        molecule_meta.species,
        counter.best_genome
    ))


    utilities.plot_genome(counter.best_genome, molecule_meta)

    plt.savefig("result.pdf")
    plt.show()
    
if __name__ == '__main__':

    main(database.Ethen)

