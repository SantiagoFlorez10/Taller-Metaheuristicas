import time
import random
import numpy as np
import globals
from utils import evaluar_tour


class CellularGeneticAlgorithm:

    def __init__(self,
                 n,
                 distancias,
                 pop_size=100,
                 mutation_rate=0.2,
                 diversity_factor=0.05,
                 replacement_strategy="worst",
                 two_opt_mode="off"):

        self.n = n
        self.distancias = distancias
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.diversity_factor = diversity_factor
        self.replacement_strategy = replacement_strategy
        self.two_opt_mode = two_opt_mode

    # =====================================================
    # RUN
    # =====================================================
    def run(self):

        start_time = time.time()
        history = []

        DIVERSITY_THRESHOLD = int(self.n * self.diversity_factor)

        # =====================================================
        # UTILIDADES
        # =====================================================
        def fitness(ind):
            return evaluar_tour(ind, self.distancias)

        def hash_tour(tour):
            return tuple(tour)

        def hamming_distance(ind1, ind2):
            return sum(1 for i in range(self.n) if ind1[i] != ind2[i])

        # =====================================================
        # 2-OPT
        # =====================================================
        def two_opt(tour):

            best = tour[:]
            best_cost = fitness(best)
            improved = True

            while time.time() - start_time < globals.MAX_TIME and improved:

                improved = False

                for i in range(1, self.n - 2):
                    for j in range(i + 1, self.n):

                        if j - i == 1:
                            continue

                        new_tour = best[:]
                        new_tour[i:j] = reversed(best[i:j])
                        new_cost = fitness(new_tour)

                        if new_cost < best_cost:
                            best = new_tour
                            best_cost = new_cost
                            improved = True

                    if time.time() - start_time >= globals.MAX_TIME:
                        break

            return best

        # =====================================================
        # INICIALIZACIÓN
        # =====================================================
        population = []
        fitness_values = []
        hash_set = set()

        while len(population) < self.pop_size:
            ind = list(range(self.n))
            random.shuffle(ind)
            h = hash_tour(ind)

            if h not in hash_set:
                population.append(ind)
                fitness_values.append(fitness(ind))
                hash_set.add(h)

        best_idx = np.argmin(fitness_values)
        best = population[best_idx]
        best_cost = fitness_values[best_idx]

        # =====================================================
        # OPERADORES
        # =====================================================
        def selection():
            k = 3
            indices = random.sample(range(self.pop_size), k)
            best_local = min(indices, key=lambda i: fitness_values[i])
            return population[best_local]

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

        def mutation(ind):
            i, j = random.sample(range(self.n), 2)
            ind[i], ind[j] = ind[j], ind[i]
            return ind

        def choose_replacement(child):

            if self.replacement_strategy == "worst":
                return np.argmax(fitness_values)

            elif self.replacement_strategy == "most_similar":
                distances = [hamming_distance(child, ind) for ind in population]
                return np.argmin(distances)

            elif self.replacement_strategy == "worst_similar":
                distances = [hamming_distance(child, ind) for ind in population]
                closest = np.argsort(distances)[:10]
                return max(closest, key=lambda i: fitness_values[i])

            else:
                raise ValueError("Estrategia no válida")

        # =====================================================
        # LOOP PRINCIPAL
        # =====================================================
        while time.time() - start_time < globals.MAX_TIME:

            p1 = selection()
            p2 = selection()

            child = crossover(p1, p2)

            if random.random() < self.mutation_rate:
                child = mutation(child)

            # -------------------------------------------------
            # CONTROL DUPLICADOS
            # -------------------------------------------------
            h_child = hash_tour(child)
            if h_child in hash_set:
                history.append(best_cost)
                continue

            # -------------------------------------------------
            # CONTROL DIVERSIDAD
            # -------------------------------------------------
            min_div = min(hamming_distance(child, ind) for ind in population)
            if min_div < DIVERSITY_THRESHOLD:
                history.append(best_cost)
                continue

            child_cost = fitness(child)

            # -------------------------------------------------
            # DECISIÓN 2-OPT
            # -------------------------------------------------
            if self.two_opt_mode != "off":

                temp_fitness = fitness_values + [child_cost]
                ranking = np.argsort(temp_fitness)
                rank_child = np.where(ranking == len(temp_fitness) - 1)[0][0]

                aplicar = False

                if self.two_opt_mode == "top-5" and rank_child < 5:
                    aplicar = True

                if self.two_opt_mode == "top-10" and rank_child < 10:
                    aplicar = True

                if aplicar:
                    child = two_opt(child)
                    child_cost = fitness(child)

            # -------------------------------------------------
            # REEMPLAZO
            # -------------------------------------------------
            idx_replace = choose_replacement(child)

            if child_cost < fitness_values[idx_replace]:

                old_hash = hash_tour(population[idx_replace])
                hash_set.discard(old_hash)

                population[idx_replace] = child
                fitness_values[idx_replace] = child_cost
                hash_set.add(hash_tour(child))

                if child_cost < best_cost:
                    best_cost = child_cost
                    best = child

            history.append(best_cost)

        return best, best_cost, history