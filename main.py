import utilities
import argparse
import matplotlib.pyplot as plt

import GA as ga

# TODO commandline argument for molecules

Na2 = """
Na
Na 1 2.00000
"""

CH4Optimal = (
    "sto-3g",
    """C
H   1 1.089000
H   1 1.089000  2  109.4710
H   1 1.089000  2  109.4710  3  120.0000
H   1 1.089000  2  109.4710  3 -120.0000"""
) 

CH4 = (
    "sto-3g",
    """C
H       1        0.97277
H       1        1.83709     2       75.22245
H       1        1.51072     2       68.33162     3      229.76379
H       1        1.46114     2      132.61576     3      144.53074"""
)

Be5 = (
    "sto-3g",
    """Be 
Be      1        1.34938
Be      1        1.33697     2       76.21984
Be      3        1.09958     1       58.28930     2       52.72671
Be      3        0.86090     1       55.17264     2      340.85847"""
)

Au11 = (
    "aug-cc-pVDZ-pp",
    """Au 
Au      1        2.94793
Au      2        1.42838     1       52.10134
Au      2        1.57412     1      127.90754     3       71.31932
Au      3        1.16311     2      154.31965     1      285.70814
Au      1        1.79223     2       39.34704     3      180.39389
Au      2        1.96822     1      129.17899     3      251.31932
Au      7        2.24173     2       71.58992     1        0.02562
Au      2        1.46600     1       68.73655     3       71.31932
Au      8        1.85913     7      126.76067     2       64.18444
Au      2        1.25557     1      117.48985     3      178.69855
Au      3        1.06216     2      122.67485     1       45.25976
Au      9        1.46831     2       67.78086     1       83.97701
Au      3        1.53839     2       66.24055     1      230.68731
"""
)


Ag11 = (
    "aug-cc-pVDZ-pp",
    """Ag 
Ag      1        2.05582
Ag      1        5.50766     2       97.05418
Ag      3        4.50769     1       68.36745     2       68.87680
Ag      3        2.65294     1      122.71947     2      102.78170
Ag      5        1.17093     3       79.86863     1       20.00155
Ag      4        2.43944     3       93.43852     1      72.30375
Ag      2        2.36688     1       79.52714     3      293.30386
Ag      1        4.06750     2      115.92736     3      28.97667
Ag      3        2.35778     1       49.18152     2      280.46291
Ag      4        3.14174     3      120.99307     1      140.54695
"""
)


Ag4 = (
    "aug-cc-pVDZ-pp",
    """Ag 
Ag      1        2.94793
Ag      2        1.42838     1       52.10134
Ag      2        1.57412     1      127.90754     3       71.31932
"""
)

def main(data):

    molecule_meta = utilities.MoleculeMetaData(*data)

    params = {
        "n_population": 100,
        "threshold_energy_difference": 1e-8,
        "n_generations": 100,
        "probability_crossing": 0.5,
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
    main(Au11)

