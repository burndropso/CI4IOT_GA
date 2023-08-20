import random
import sys
import time

import pandas as pd
from deap import base, creator, tools

# Load the distance matrix from an Excel file
dist_df = pd.read_excel("Project3_DistancesMatrix.xlsx", sheet_name="Sheet1", index_col=0)
# dist_df = pd.read_csv("Project3_DistancesMatrix.csv")

distance_matrix = dist_df.to_numpy()

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize the fitness value
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# Attribute generator
toolbox.register("indices", random.sample, range(0, len(distance_matrix)), len(distance_matrix))
# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# Define the fitness function
def fitness(path):
    # Remove 0 from middle of path
    a = list(path)
    a.remove(0)
    # Calculate the sum of distances between consecutive positions
    dist = 0
    for i in range(len(a) - 1):
        dist += distance_matrix[a[i]][a[i + 1]]

    dist += distance_matrix[0][a[0]] + distance_matrix[a[-1]][0]
    return dist,


toolbox.register("evaluate", fitness)
toolbox.register("mate", tools.cxOrdered)  # Ordered crossover between two individuals
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.3)  # Probability of shuffing a index of a individual
toolbox.register("select", tools.selTournament, tournsize=3)  # Select the 3 bests individuals

if __name__ == "__main__":
    # Record the start time
    start_time = time.perf_counter()

    # Generate random seed
    seed = random.randrange(sys.maxsize)
    # seed = 7352728248864557841
    random.seed(seed)

    # Initialize the population
    popula = 400
    pop = toolbox.population(n=popula)  # n individuals in a population

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    # NGEN is the number of generations
    CXPB, MUTPB, NGEN = 0.8, 0.2, 10000

    # Begin the evolution
    for g in range(NGEN):
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness(not evaluated yet)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        print("Last fitness value calculated:", fit)

        # Replace the old population with the offspring
        pop[:] = offspring

    # Sort the population by fitness
    pop = sorted(pop, key=lambda ind: ind.fitness.values)

    best_individual = [0] + [item for item in pop[0] if item != 0] + [0]
    # Record the end time
    end_time = time.perf_counter()
    # Print the best individual (path)
    print("-- Best individual --")
    print("Path: ", best_individual)
    print("Fitness: ", pop[0].fitness.values[0])
    print("Random Seed: ", seed)
    print("Execution time: ", end_time - start_time, "seconds")
    print("Params: ", CXPB, "(CXPB);  ", MUTPB, "(MUTPB);  ", NGEN, "(NGEN);  ", popula, "(Population)")


# -- Best individual --
# Path:  [0, 72, 76, 77, 27, 79, 18, 55, 23, 28, 44, 69, 33, 21, 80, 62, 64, 82, 81, 85, 96, 52, 4, 39, 51, 22, 88, 32, 78, 13, 71, 91, 8, 24, 31, 50, 70, 36, 97, 99, 56, 74, 16, 2, 92, 35, 42, 37, 38, 40, 5, 93, 83, 95, 94, 86, 73, 75, 53, 12, 30, 9, 87, 54, 58, 57, 45, 60, 1, 17, 89, 43, 34, 7, 46, 65, 10, 11, 19, 26, 48, 15, 61, 47, 29, 49, 68, 98, 25, 3, 90, 84, 59, 67, 6, 14, 41, 63, 66, 20, 0]
# Fitness:  42.9
# Random Seed:  7352728248864557841
# Execution time:  397.6317148 seconds
# Params:  0.8 (CXPB);   0.2 (MUTPB);   10000 (NGEN);   400 (Population)
