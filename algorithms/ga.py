import time
import random
import numpy as np
import globals
from utils import evaluar_tour


class GeneticAlgorithm:

    def __init__(self, n, distancias,
                 pop_size=100,
                 crossover_rate=0.9,
                 mutation_rate=0.05,
                 elite_size=5):

        self.n = n
        self.distancias = distancias
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size

    # =====================================================
    # RUN
    # =====================================================
    def run(self):

        start_time = time.time()
        history = []

        # =====================================================
        # INIT POPULATION
        # =====================================================
        def init_population():
            population = []
            for _ in range(self.pop_size):
                ind = list(range(self.n))
                random.shuffle(ind)
                population.append(ind)
            return population

        # =====================================================
        # FITNESS
        # =====================================================
        def fitness(ind):
            return evaluar_tour(ind, self.distancias)

        # =====================================================
        # SELECTION (Torneo)
        # =====================================================
        def selection(population):
            k = 3
            candidates = random.sample(population, k)
            candidates.sort(key=fitness)
            return candidates[0]

        # =====================================================
        # CROSSOVER (Order Crossover)
        # =====================================================
        def crossover(p1, p2):
            a, b = sorted(random.sample(range(self.n), 2))
            child = [None] * self.n
            child[a:b] = p1[a:b]

            fill = [c for c in p2 if c not in child]
            idx = 0
            for i in range(self.n):
                if child[i] is None:
                    child[i] = fill[idx]
                    idx += 1

            return child

        # =====================================================
        # MUTATION (swap)
        # =====================================================
        def mutation(ind):
            i, j = random.sample(range(self.n), 2)
            ind[i], ind[j] = ind[j], ind[i]
            return ind

        # =====================================================
        # ELITISM
        # =====================================================
        def elitism(population):
            population.sort(key=fitness)
            return population[:self.elite_size]

        # =====================================================
        # MAIN LOOP
        # =====================================================
        population = init_population()

        best = None
        best_cost = float("inf")

        while time.time() - start_time < globals.MAX_TIME:

            new_population = elitism(population)

            while len(new_population) < self.pop_size:

                parent1 = selection(population)
                parent2 = selection(population)

                if random.random() < self.crossover_rate:
                    child = crossover(parent1, parent2)
                else:
                    child = parent1[:]

                if random.random() < self.mutation_rate:
                    child = mutation(child)

                new_population.append(child)

            population = new_population

            # Evaluar mejor
            population.sort(key=fitness)
            current_best = population[0]
            current_cost = fitness(current_best)

            if current_cost < best_cost:
                best_cost = current_cost
                best = current_best

            history.append(best_cost)

        return best, best_cost, history