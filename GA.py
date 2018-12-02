"""This module contains the components of the genetic algorithm. 

Author:
 - Johannes Cartus, TU Grazs
"""

import numpy as np

from deap import base
from deap import creator
from deap import tools

import utilities, energy
from utilities import Logger
import random

class Params(object):
    """Class that stores the parameters used in the algorithm.
    
    Attributes:
     - size_of_population <int>: number individuals in one generation.
     - number_of_generations <int>: number of generations to run the 
        algorithm for.
     - threshold_energy_difference <float>: to check if the energy of a new 
        geometry is lower than that of other the difference must be greater than
        this threshold.
     - tournament_size <int>: for next generation out of three tournament_size
        individuals the best will be picked.
     - probability_crossing <float>: the probability for an individual to be 
        mated with another (random) individual of the population.
     - probability_mutation <float>: the probability for an individual to 
        be selected for mutation.
     - individual_gene_probability_mutation <float>: the probability for 
        each gene to be mutated, once an individual was selected for mutation.
     - noise_initial_population <float>: the fraction of the value of a gene
        used as standard deviation for the distribution from which the 
        individuals for the initial population are drawn.
     - noise_mutation <float>: the fraction of the value of a gene used as 
        standard deviation for the normal distribution from which deviantion 
        to the original values are drawn in order to mutate the gene.
     - fitness_callback <function>: function that takes a molecule and returns
        an energy value for it (e.g. via DFT calculation).
    """

    def __init__(
        self, 
        size_of_population=200, 
        number_of_generations=50,
        threshold_energy_difference=1e-8,
        tournament_size=3,
        probability_crossing=0.3,
        probability_mutation=0.5,
        individual_gene_probability_mutation=0.1,
        noise_initial_population=0.8,
        noise_mutation=0.2,
        fitness_callback=energy.uhf_energy
    ):
        
        self.size_of_population = size_of_population
        self.number_of_generations = number_of_generations
        self.threshold_energy_difference = threshold_energy_difference
        self.tournament_size = tournament_size
        self.probability_crossing = probability_crossing
        self.probability_mutation = probability_mutation
        self.individual_gene_probability_mutation = \
            individual_gene_probability_mutation
        self.noise_mutation = noise_mutation
        self.noise_initial_population=noise_initial_population
        self.fitness_callback = fitness_callback


            

class GAStructureOptimisation(object):
    """This class contains the Genetic Algorithm that optimizes a molecule
    structure. The (meta-)information of this molecule must be passed 
    into the run method."""


    def __init__(self, params):

        self._params = params
        self._timer = utilities.Timer()
    
        Logger.log_params("Algorithm Parameters: ", params, 2)

    def run(self, meta):
        """Start the optimisation.

        Args:
         - meta <utilities.MoleculeMetaData>: meta information of the molecule
            structure to be optimized.
        
        Returns: 
         - <utilities.Result> counter object which has 
            information on the result of the optimisation
        """
        
        Logger.log("Optimisation started.", 3)
        Logger.log_zmatrix("Initial geometry:", meta, meta.genome)

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self._timer.start()

        toolbox = base.Toolbox()

        #--- register initialisation ---
        toolbox.register(
            "init_individual", 
            tools.initIterate, 
            creator.Individual, 
            lambda: self.mutate_individual(meta.genome)
        )

        toolbox.register(
            "init_population", 
            tools.initRepeat, 
            list, 
            toolbox.init_individual
        )
        #---

        # register fitness function
        toolbox.register(
            "evaluate", 
            self.evaluateFitness, 
            meta=meta    
        )
        
        #--- register genetic functions ---
        toolbox.register("mate", tools.cxTwoPoint)

        # alter gene with 0.05 % probability (w/ noise)
        toolbox.register("mutate", self.mutate_individual)
        toolbox.register(
            "select", 
            tools.selTournament, 
            tournsize=self._params.tournament_size
        )
        #---

        #--- create initial population and calculate fitness ---
        population = toolbox.init_population(n=self._params.size_of_population)

        fitness_values = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitness_values):
            ind.fitness.values = fit

        Logger.log(
            "Initial population     E_min = {:3.5f} / Hartree".format(
                tools.selBest(population, 1)[0].fitness.values[0]
            )
        )

        counter = utilities.Result(self._params.threshold_energy_difference)
        counter.update(fitness_values, population)
        #---

        

        #--- execute algorithm ---

        # Begin the evolution
        for i in range(self._params.number_of_generations):

            # Select next generation (via turnament)
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))
            
            # shuffle the elements, so cross over is between random individuals
            random.shuffle(offspring)

            # do cross over (cross neighbours)
            count_cross_over = 0
            for child1, child2 in zip(offspring[::2], offspring[1::2]):

                if random.random() < self._params.probability_crossing:
                    toolbox.mate(child1, child2)

                    del child1.fitness.values
                    del child2.fitness.values

                    count_cross_over += 1
                    
            # do mutation
            count_mutation_candidates = 0
            for mutant in offspring:
                if random.random() < self._params.probability_mutation: 
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            
                    count_mutation_candidates += 1

            # log number of mutations and matings
            Logger.log(

                "Cross over candidates: {:3d}\n".format(count_cross_over) + \
                "Mutation candidates:   {:3d}".format(count_mutation_candidates)
            )

            # recalculate fitness values of mates and mutants
            invalid_individuals = \
                [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_individuals)
            for ind, fit in zip(invalid_individuals, fitnesses):
                ind.fitness.values = fit
            
            population[:] = offspring
            
            # update list of fitness value
            fitness_values = [ind.fitness.values[0] for ind in population]
            

            
            #--- print out information on the energy ---
            Logger.log(
                "Generation {:3d} finished. E_mean = {:3.5f} / Hartree.".format(
                    i+1, 
                    np.mean(fitness_values)
                )
            )
            #---

            # updates fitness distributions
            counter.update(fitness_values, population)
        #---

        Logger.log("Optimisation finished.", 3)
        Logger.log(
            "Operation elapsed in {:2.4f} ms.".format(self._timer.stop()),
            2
        )
        Logger.log("E_min = {:4.5f} found {:3d} times".format(
            counter.E_min, 
            counter.count
        ), 2)
        Logger.log_zmatrix("Best Geometry:", meta, counter.best_genome, 2)

        return counter


    def mutate_individual(self, genome):
        """Add some noise to a genome (and then take the absolute value) 
        to yield a mutated individual. Absolute value is taken to ensure the 
        resulting values are valid for z-matrix.
        Used for mutation and initialisation.
        
        Args:
        - genome <list<float>>: list of genes to be mutated.
        
        Returns:
        - <list<float>>: mutated genome.
        """
        
        mutated_genome = []
        for gene in genome:
            
            if random.random() < \
                self._params.individual_gene_probability_mutation: 
                mutated_gene = abs(
                    gene + \
                    random.gauss(0, self._params.noise_mutation * gene)
                )
            else:
                mutated_gene = gene
            
            mutated_genome.append(mutated_gene)

        return mutated_genome


    def evaluateFitness(self, individual, meta):
        """Calculate the energy of an electron. This will be used as fitness
        function for the GA."""

        mol = utilities.build_molecule_from_genome(individual, meta)
        
        try:
            E = self._params.fitness_callback(mol)
        except Exception as ex:
            print("Problem during SCF calculation: " +  str(ex))
            E = 1e10


        # has to be a tuple (because of syntax of DEAP library)
        return E,
