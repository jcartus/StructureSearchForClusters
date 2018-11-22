"""This module contains unit tests."""

import utilities
import molecules as database

import unittest

class MoleculeMetaData(unittest.TestCase):

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