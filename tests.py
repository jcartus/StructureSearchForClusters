"""This module contains unit tests."""

import utilities
import molecules as database

import unittest
import numpy as np


class TestGAUtils(unittest.TestCase):

    def test_mutation_deactivated(self):
        """Mutation probability zero means it genome remains unchanged."""
        
        genome = np.random.rand(10)

        # turn off noise for sample to mutate
        genome_no_mutation = utilities.mutate_individual(genome, 0.0, 1.0)
        np.testing.assert_array_equal(
            genome,
            genome_no_mutation
        )

        # turn off individual probability for gene mutation
        genome_no_gene_mutation = utilities.mutate_individual(
            genome, 
            np.random.rand(),
            0.0
        )
        np.testing.assert_array_equal(
            genome,
            genome_no_gene_mutation
        )


    def test_mutation_values_positive(self):
        """Test if the mutated values fulfill required properties. """

        genome = \
            [np.random.rand() + np.random.randint(1, 100) for i in range(10)]

        # mutate all genes
        noise = np.random.rand()
        indiv_prob = 1
        mutated_genome = utilities.mutate_individual(genome, noise, indiv_prob)

        #--- analysis ---
        # make sure all genes stay positive
        np.testing.assert_array_less(
            np.zeros(len(genome)),
            mutated_genome
        ) 
        #---

    def test_mutation_values_distribution(self):
        """Test if the mutated values fulfill required properties. """

        mu = np.random.randint(10)
        genome = [mu] * int(1e4) 

        # mutate all genes
        noise = np.random.rand()
        indiv_prob = 1
        mutated_genome = utilities.mutate_individual(genome, noise, indiv_prob)

        #--- analysis ---
        # make sure mean is still the old genome
        np.testing.assert_allclose(
            mu,
            np.mean(mutated_genome),
            atol=5e-1
        ) 

        # make sure the scatter as planned
        np.testing.assert_allclose(
            noise * mu,
            np.std(mutated_genome),
            atol=5e-1
        )
        #---






class TestMoleculeMetaData(unittest.TestCase):

    def setUp(self):

        self.basis = database.EthenOptimum[0]
        self.matrixstr = database.EthenOptimum[1]

        self.first_atom = "C"
        self.second_atom = "C"
        self.third_atom = "H"

        self.species = [
            ("H", ['1', '2', '3']), 
            ("H", ['2', '1', '3']),
            ("H", ['2', '1', '3']),
        ]

        self.genome = [
            1.339, 
            1.087, 121.3, 
            1.087, 121.3, 180.0,
            1.087, 121.3, 0.0,
            1.087, 121.3, 180.0
        ]

    def test_calculate_genome(self):

        meta = utilities.MoleculeMetaData(self.basis, self.matrixstr)


        self.assertEqual(
            meta.first_atom,
            self.first_atom
        )
        
        self.assertEqual(
            meta.second_atom,
            self.second_atom
        )

        self.assertEqual(
            meta.third_atom,
            self.third_atom
        )
            
        self.assertEqual(
            meta.species,
            self.species
        )

        self.assertEqual(
            meta.genome,
            self.genome
        )


class TestCreateZMatrix(unittest.TestCase):

    def test_ethen(self):

        meta = utilities.MoleculeMetaData(*database.EthenOptimum)

        actual = utilities.create_z_matrix(
            meta.first_atom,
            meta.second_atom,
            meta.third_atom,
            meta.species,
            meta.genome
        ).split()



        expected = database.EthenOptimum[1].split()

        self.assertEqual(
            actual,
            expected
        )

if __name__ == '__main__':
    unittest.main()