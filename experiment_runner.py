import random
import numpy as np
import time
from algorithms.ga import GeneticAlgorithm
from algorithms.aco import AntColony
from algorithms.cbga import CellularGeneticAlgorithm

# Se mantiene global porque tus algoritmos lo usan
import globals


class ExperimentRunner:

    def __init__(self, budget):
        self.budget = budget

    def run(self, algoritmo, instancia, seed, config=None):
        
        random.seed(seed)
        np.random.seed(seed)

        globals.MAX_TIME = self.budget

        inicio = time.time()

        if algoritmo == "GA":

            if config is None:
                config = {}

            model = GeneticAlgorithm(
                instancia.dimension,
                instancia.distancias,
                pop_size=config.get("pop_size", 100),
                crossover_rate=config.get("crossover_rate", 0.9),
                mutation_rate=config.get("mutation_rate", 0.05),
                elite_size=config.get("elite_size", 5)
            )

            _, best, history = model.run()

        elif algoritmo == "ACO":
            model = AntColony(instancia.distancias)
            _, best, history = model.run()

        elif algoritmo == "CBGA":

            if config is None:
                config = {}

            model = CellularGeneticAlgorithm(
                instancia.dimension,
                instancia.distancias,
                pop_size=config.get("pop_size", 100),
                mutation_rate=config.get("mutation_rate", 0.2),
                diversity_factor=config.get("diversity_factor", 0.05),
                replacement_strategy=config.get("replacement_strategy", "worst"),
                two_opt_mode=config.get("two_opt", "off")
            )

            _, best, history = model.run()
        else:
            raise ValueError("Algoritmo no reconocido")

        tiempo = time.time() - inicio
        optimo = instancia.optimo
        if optimo is not None:
            gap = ((best - optimo) / optimo) * 100
        else:
            gap = np.nan

        result = {
            "instancia": instancia.nombre,
            "algoritmo": algoritmo,
            "seed": seed,
            "best": best,
            "tiempo": tiempo,
            "gap_%": gap,
            "history": history
        }

        if config:
            for k, v in config.items():
                result[k] = v

        return result