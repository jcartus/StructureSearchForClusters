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

Be5 = """Be 
Be      1        1.34938
Be      1        1.33697     2       76.21984
Be      3        1.09958     1       58.28930     2       52.72671
Be      3        0.86090     1       55.17264     2      340.85847
"""

Au13 = """Au 
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
Au      1        1.30229     2       64.62637     3      108.68068
Au      8        1.64300     7       78.05606     2      273.65350
"""

def main(zMatrix):

    molecule_meta = utilities.MoleculeMetaData(zMatrix)

    params = {
        "n_population": 100,
        "threshold_energy_difference": 1e-8,
        "n_generations": 100,
        "probability_crossing": 0.5,
        "probability_mutation": 0.2,
    }

    counter = ga.find_best_geometry(molecule_meta, params)

    print("--> E_min = {0} found {1} times".format(counter.E_min, counter.count))



    
if __name__ == '__main__':
    main(Be5)

