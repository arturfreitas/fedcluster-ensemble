import random
from deap import base, creator, tools, algorithms
from typing import List, Tuple
from niid_bench.main import main_wrapper
import logging
from contextlib import contextmanager
# Define the multi-objective problem: maximize accuracy (positive) and minimize communication cost (negative)
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Number of rounds in the main_wrapper function
num_rounds = 2

# DEAP setup
toolbox = base.Toolbox()

# Attribute generator: random float between 0 and 1 for each round
toolbox.register("attr_float", random.uniform, 0.0, 1.0)

# Structure initializer: creates individuals (lists) of length num_rounds
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, num_rounds)

# Population initializer
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


@contextmanager
def suppress_logs():
    # Save the current logging level
    previous_level = logging.getLogger().level
    # Set the logging level to CRITICAL to suppress lower-level logs
    logging.getLogger().setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        # Restore the previous logging level
        logging.getLogger().setLevel(previous_level)


# Evaluation function
def evaluate(individual):
    # Run the main_wrapper function with the individual's values as the fraction_fit_lst
    with suppress_logs():
        accuracy, cost = main_wrapper(list(individual))
    return accuracy, cost

# Register the evaluation function
toolbox.register("evaluate", evaluate)

# Genetic operators: mating (crossover) and mutation
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0.5, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

def main():
    # Set up the population
    population = toolbox.population(n=10)

    # Probability of mating two individuals
    cxpb = 0.5
    # Probability of mutating an individual
    mutpb = 0.2
    # Number of generations
    ngen = 40

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    print("Initial population:")
    for i, ind in enumerate(population):
        print(f"Individual {i}: {ind}, Fitness: {ind.fitness.values}")

    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace the old population with the offspring
        population[:] = offspring

        # Print the individuals for the current generation
        print(f"\nGeneration {gen}:")
        for i, ind in enumerate(population):
            print(f"Individual {i}: {ind}, Fitness: {ind.fitness.values}")

    # Get the best individuals based on Pareto front (multi-objective optimization)
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    
    # Print the individuals in the Pareto front
    print("\nPareto front solutions:")
    for ind in pareto_front:
        print(f"Individual: {ind}, Fitness: {evaluate(ind)}")

if __name__ == "__main__":
    main()
