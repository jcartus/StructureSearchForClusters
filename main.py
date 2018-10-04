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
        "threshold_energy_difference": 1e-8,
        "n_generations": 10,
        "probability_crossing": 0.5,
        "probability_mutation": 0.2,
    }

    counter = ga.find_best_geometry(molecule_meta, params)

    print("### E_min = {0} found {1} times".format(counter.E_min, counter.count))



    
if __name__ == '__main__':
    main(CH4)

