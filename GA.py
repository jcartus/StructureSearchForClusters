import numpy as np

from deap import base
from deap import creator
from deap import tools

import utilities
from utilities import Logger
import random

class Params(object):
    """Class that stores the parameters used in the algorithm.
    
    Attributes:
     - size_of_population <int>: number individuals in one generation.
     - number_of_generations <int>:
    
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
        fitness_callback=utilities.uhf_energy
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
            lambda: utilities.mutate_individual(
                meta.genome, 
                self._params.noise_initial_population
            )
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
            utilities.evaluateFitness, 
            meta=meta,
            fitness_callback=self._params.fitness_callback    
        )
        
        #--- register genetic functions ---
        toolbox.register("mate", tools.cxTwoPoint)

        # alter gene with 0.05 % probability (w/ noise)
        toolbox.register(
            "mutate", 
            utilities.mutate_individual, 
            noise=self._params.noise_mutation, 
            gene_mutation_probability=\
                self._params.individual_gene_probability_mutation
        )
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

            # Select next generation
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            # do cross over
            for child1, child2 in zip(offspring[::2], offspring[1::2]):

                if random.random() < self._params.probability_crossing:
                    toolbox.mate(child1, child2)

                    del child1.fitness.values
                    del child2.fitness.values
                    
            # do mutation
            for mutant in offspring:
                if random.random() < self._params.probability_mutation: 
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            
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
                "Generation {:3i} finished. E_mean = {:3.5f} / Hartree.".format(
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
        Logger.log("E_min = {:4.5f} found {:4i} times".format(
            counter.E_min, 
            counter.count
        ), 2)
        Logger.log_zmatrix("Best Geometry:", meta, counter.best_genome, 2)

        return counter


