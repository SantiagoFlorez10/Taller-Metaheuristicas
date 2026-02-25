import time
import random
import numpy as np
import globals
from utils import evaluar_tour


class AntColony:

    def __init__(self,
                 distancias,
                 num_ants=20,
                 alpha=1.0,
                 beta=2.0,
                 rho=0.1,
                 Q=100):

        self.distancias = distancias
        self.n = len(distancias)

        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q

    # =====================================================
    # RUN
    # =====================================================
    def run(self):

        start_time = time.time()
        history = []

        # Inicializar feromonas
        tau0 = 1.0
        pheromones = np.full((self.n, self.n), tau0)

        best_tour = None
        best_cost = float("inf")

        # =====================================================
        # ELEGIR SIGUIENTE CIUDAD
        # =====================================================
        def elegir_siguiente_ciudad(current, visited):

            probabilities = []
            total = 0.0

            for j in range(self.n):
                if not visited[j]:
                    tau = pheromones[current][j] ** self.alpha
                    eta = (1.0 / (self.distancias[current][j] + 1e-10)) ** self.beta
                    value = tau * eta
                    probabilities.append((j, value))
                    total += value

            r = random.random() * total
            cumulative = 0.0

            for city, prob in probabilities:
                cumulative += prob
                if cumulative >= r:
                    return city

            return probabilities[-1][0]

        # =====================================================
        # CONSTRUIR SOLUCIÓN
        # =====================================================
        def construir_solucion():

            visited = [False] * self.n
            tour = []

            current = random.randint(0, self.n - 1)
            tour.append(current)
            visited[current] = True

            for _ in range(self.n - 1):
                next_city = elegir_siguiente_ciudad(current, visited)
                tour.append(next_city)
                visited[next_city] = True
                current = next_city

            return tour

        # =====================================================
        # ACTUALIZAR FEROMONAS
        # =====================================================
        def actualizar_feromonas(all_tours):

            # Evaporación
            pheromones[:] = (1 - self.rho) * pheromones

            # Depósito
            for tour in all_tours:
                cost = evaluar_tour(tour, self.distancias)
                deposit = self.Q / cost

                for i in range(self.n - 1):
                    a, b = tour[i], tour[i + 1]
                    pheromones[a][b] += deposit
                    pheromones[b][a] += deposit

                # cerrar ciclo
                pheromones[tour[-1]][tour[0]] += deposit
                pheromones[tour[0]][tour[-1]] += deposit

            pheromones[:] = np.clip(pheromones, 1e-6, 1e6)

        # =====================================================
        # BUCLE PRINCIPAL
        # =====================================================
        while time.time() - start_time < globals.MAX_TIME:

            all_tours = []

            for _ in range(self.num_ants):
                tour = construir_solucion()
                all_tours.append(tour)

                cost = evaluar_tour(tour, self.distancias)
                if cost < best_cost:
                    best_cost = cost
                    best_tour = tour

            actualizar_feromonas(all_tours)

            history.append(best_cost)

        return best_tour, best_cost, history