import utilities
import argparse
import matplotlib.pyplot as plt

import GA as ga

# TODO commandline argument for molecules

Na2 = """
Na
Na 1 2.00000
"""

CH4 = """C
H   1 1.089000
H   1 1.089000  2  109.4710
H   1 1.089000  2  109.4710  3  120.0000
H   1 1.089000  2  109.4710  3 -120.0000"""

H2O = """
O
H 1 1.08
H 1 1.08 2 109.0
"""

def main(zMatrix):

    molecule_meta = utilities.MoleculeMetaData(zMatrix)

    params = {
        "n_population": 10,
        "n_fitness_bins": 50,
        "n_generations": 10,
        "probability_crossing": 0.5,
        "probability_mutation": 0.2,

    }

    energies, bins = ga.find_best_geometry(molecule_meta, params)

    utilities.plot_data(energies, bins)

    plt.show()



    
if __name__ == '__main__':
    main(CH4)

