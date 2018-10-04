import numpy as np

from deap import base
from deap import creator
from deap import tools

import utilities
import random


def find_best_geometry(molecule_meta, params):
    """
    molecule_meta utities. MoleculeMetaData
    params dict<str,??>

    return energy values distribution
    """

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    #--- register initialisation ---
    toolbox.register(
        "init_individual", 
        tools.initIterate, 
        creator.Individual, 
        lambda: utilities.init_individual(molecule_meta)
    )

    toolbox.register(
        "init_population", 
        tools.initRepeat, 
        list, 
        toolbox.init_individual
    )
    #---

    # register fitness function
    toolbox.register("evaluate", utilities.evaluateFitness, meta=molecule_meta)
    
    #--- register genetic functions ---
    toolbox.register("mate", tools.cxTwoPoint)

    # alter gene with 0.05 % probability (w/ noise)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.05) # TODO mutation noise should be proportional to gene value!
    toolbox.register("select", tools.selTournament, tournsize=3)
    #---

    #--- create initial population and calculate fitness ---
    population = toolbox.init_population(n=params["n_population"])

    fitness_values = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitness_values):
        ind.fitness.values = fit

    fitness_distribution = utilities.Distribution(
        fitness_values, 
        params["n_fitness_bins"]
    )
    #---

    #--- Print information ---
    #TODO print infos about initial population
    print(
        "Initial population minimal energy: {0}".format(
            tools.selBest(population, 1)[0].fitness.values
        )
    )
    #---

    #--- execute algorithm ---

    # Begin the evolution
    for i in range(params["n_generations"]):

        # Select next generation
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # do cross over
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            if random.random() < params["probability_crossing"]:
                toolbox.mate(child1, child2)

                del child1.fitness.values
                del child2.fitness.values
                
        # do mutation
        for mutant in offspring:
            if random.random() < params["probability_mutation"]:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # recalculate fitness values of mates and mutants
        invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_individuals)
        for ind, fit in zip(invalid_individuals, fitnesses):
            ind.fitness.values = fit
        
        population[:] = offspring
        
        # update list of fitness value
        fitness_values = [ind.fitness.values[0] for ind in population]
        
        # updates fitness distributions
        fitness_distribution.update(fitness_values)
        
        #--- print out information on the energy ---
        print("Generation {0} finished.\n - E_mean = {1}\n".format(
            i+1, 
            np.mean(fitness_values)
        ))
        #---
        
        # TODO convergence criterium
        #if np.abs(E - E_old) < CONVERGENCE_THRESHOLD:
        #    print("\n\nCONVERGED!\n")
        #    break
        #
        #else:
        #    E_old = E

    #---

    return fitness_distribution


