import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from experiment_runner import ExperimentRunner
from utils import cargar_tsp, matriz_distancias
from tsp_instance import TSPInstance


class TuningPipeline:

    def __init__(self, tsp_files, n_seeds, budget):
        self.tsp_files = tsp_files
        self.n_seeds = n_seeds
        self.budget = budget
        self.instancias = []
        self.runner = ExperimentRunner(budget)

    # =====================================================
    # CARGAR INSTANCIAS
    # =====================================================
    def cargar_instancias(self):
        for ruta in self.tsp_files:
            data = cargar_tsp(ruta)

            instancia = TSPInstance(
                nombre=data["nombre"],
                dimension=data["dimension"],
                coordenadas=data["coordenadas"],
                optimo=data.get("optimo")
            )

            instancia.distancias = matriz_distancias(data["coordenadas"])
            self.instancias.append(instancia)

    # =====================================================
    # EJECUTAR EXPERIMENTO
    # =====================================================
    def run_experiment(self, algoritmo, instancia, seed, config=None):
        return self.runner.run(algoritmo, instancia, seed, config)

    # =====================================================
    # TUNING UNA VARIABLE
    # =====================================================
    def tuning_un_variable(self, algoritmo, baseline, param_name, valores):

        results = []
        seeds = range(self.n_seeds)

        for val in valores:

            config = baseline.copy()
            config[param_name] = val

            for instancia in self.instancias:
                for seed in seeds:

                    r = self.run_experiment(
                        algoritmo,
                        instancia,
                        seed,
                        config
                    )

                    results.append(r)

        df = pd.DataFrame(results)

        resumen = df.groupby(["instancia", param_name]).agg(
            promedio=("best", "mean"),
            std=("best", "std"),
            mejor=("best", "min"),
            gap_promedio=("gap_%", "mean"),
            tiempo_promedio=("tiempo", "mean")
        )

        return df, resumen

    # =====================================================
    # FASE BASELINE
    # =====================================================
    def run_baseline(self, baselines):

        all_results = []
        seeds = range(self.n_seeds)

        for algoritmo, baseline in baselines.items():
            for instancia in self.instancias:
                for seed in seeds:

                    r = self.run_experiment(
                        algoritmo,
                        instancia,
                        seed,
                        baseline
                    )

                    all_results.append(r)

        df = pd.DataFrame(all_results)
        df.to_csv("baseline.csv", index=False)

        return df

    # =====================================================
    # FASE FINAL
    # =====================================================
    def run_final_comparison(self, best_configs):

        all_results = []
        seeds = range(self.n_seeds)

        for instancia in self.instancias:

            for algoritmo, config in best_configs.items():

                for seed in seeds:

                    r = self.run_experiment(
                        algoritmo,
                        instancia,
                        seed,
                        config
                    )

                    all_results.append(r)

        df = pd.DataFrame(all_results)
        df.to_csv("resultados_finales.csv", index=False)

        return df
    
    def generar_tabla_resumen(self, df):

        tabla_resumen = df.groupby(["instancia", "algoritmo"]).agg(
            promedio=("best", "mean"),
            std=("best", "std"),
            mejor=("best", "min"),
            peor=("best", "max"),
            gap_promedio=("gap_%", "mean"),
            tiempo_promedio=("tiempo", "mean")
        ).reset_index()

        print("\n========= TABLA RESUMEN FINAL =========")
        print(tabla_resumen)

        return tabla_resumen
    def graficar_boxplots(self, df):

        for instancia in df["instancia"].unique():

            plt.figure()

            data = df[df["instancia"] == instancia]

            sns.boxplot(x="algoritmo", y="best", data=data)

            plt.title(f"Boxplot - {instancia}")
            plt.ylabel("Mejor distancia")
            plt.xlabel("Algoritmo")

            plt.show()

    def graficar_curvas_convergencia(self, df):

        for instancia in df["instancia"].unique():

            plt.figure()

            for algoritmo in df["algoritmo"].unique():

                subset = df[
                    (df["instancia"] == instancia) &
                    (df["algoritmo"] == algoritmo)
                ]

                histories = subset["history"].tolist()

                if len(histories) == 0:
                    continue

                min_len = min(len(h) for h in histories)
                histories = [h[:min_len] for h in histories]

                histories = np.array(histories)

                mean_curve = histories.mean(axis=0)
                std_curve = histories.std(axis=0)

                x = range(len(mean_curve))

                plt.plot(x, mean_curve, label=algoritmo)

                plt.fill_between(
                    x,
                    mean_curve - std_curve,
                    mean_curve + std_curve,
                    alpha=0.2
                )

            plt.title(f"Convergencia Promedio - {instancia}")
            plt.xlabel("Iteración")
            plt.ylabel("Mejor distancia")
            plt.legend()
            plt.show()   