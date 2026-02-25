import pandas as pd
from tsp_instance import TSPInstance
from experiment_runner import ExperimentRunner
from tuning_pipeline import TuningPipeline   
from utils import cargar_tsp
import numpy as np
import matplotlib.pyplot as plt



# =====================================================
# CONFIGURACIÓN
# =====================================================

N_SEEDS = 2
BUDGET = 3

TSP_FILES = [
    "data/berlin52.tsp",
    "data/st70.tsp"
]


# =====================================================
# CURVA PROMEDIO
# =====================================================

def curva_promedio(df, instancia, algoritmo):

    subset = df[(df["instancia"] == instancia) &
                (df["algoritmo"] == algoritmo)]

    histories = subset["history"].tolist()

    if len(histories) == 0:
        return None, None

    min_len = min(len(h) for h in histories)
    histories = [h[:min_len] for h in histories]

    promedio = np.mean(histories, axis=0)
    std = np.std(histories, axis=0)

    return promedio, std


# =====================================================
# GRAFICAR
# =====================================================

def graficar_curva(df, instancia):

    plt.figure(figsize=(10, 6))

    for algoritmo in ["GA", "ACO", "CBGA"]:

        promedio, std = curva_promedio(df, instancia, algoritmo)

        if promedio is None:
            continue

        x = range(len(promedio))

        plt.plot(x, promedio, label=algoritmo)
        plt.fill_between(x, promedio - std, promedio + std, alpha=0.2)

    plt.xlabel("Iteraciones")
    plt.ylabel("Mejor costo acumulado")
    plt.title(f"Curva de Convergencia Promedio - {instancia}")
    plt.legend()
    plt.grid(True)
    plt.show()


# =====================================================
# MAIN
# =====================================================

def main():

    # ===============================
    # TU PARTE ORIGINAL (NO CAMBIA)
    # ===============================

    instancias = []

    for ruta in TSP_FILES:
        data = cargar_tsp(ruta)

        instancia = TSPInstance(
            nombre=data["nombre"],
            dimension=data["dimension"],
            coordenadas=data["coordenadas"],
            optimo=data.get("optimo")
        )

        instancias.append(instancia)

    runner = ExperimentRunner(BUDGET)

    all_results = []

    for instancia in instancias:
        for algoritmo in ["GA", "ACO", "CBGA"]:
            for seed in range(N_SEEDS):

                result = runner.run(algoritmo, instancia, seed)
                all_results.append(result)

                print(
                    f"{instancia.nombre} | "
                    f"{algoritmo} | "
                    f"Seed {seed} → {result['best']:.2f}"
                )

    df = pd.DataFrame(all_results)

    df.to_csv("resultados_experimento.csv", index=False)

    print("\nExperimento finalizado correctamente.")

    # =================================================
    # RESUMEN ESTADÍSTICO
    # =================================================

    resumen = df.groupby(["instancia", "algoritmo"]).agg(
        mejor_global=("best", "min"),
        promedio=("best", "mean"),
        std=("best", "std"),
        tiempo_promedio=("tiempo", "mean"),
        gap_promedio=("gap_%", "mean")
    )

    print("\n=========== RESUMEN FINAL ===========")
    print(resumen)

    # =================================================
    # GRAFICAR CURVAS
    # =================================================

    for instancia in df["instancia"].unique():
        graficar_curva(df, instancia)

    
    GA_best = {
        "pop_size": 100,
        "mutation_rate": 0.10,
        "crossover_rate": 0.95,
        "elite_size": 2
    }

    ACO_best = {
        "num_ants": 20,
        "alpha": 1.0,
        "beta": 5,
        "rho": 0.3
    }

    CBGA_best = {
        "diversity_factor": 0.05,
        "replacement_strategy": "worst",
        "two_opt": "top-10"
    }

    best_configs = {
        "GA": GA_best,
        "ACO": ACO_best,
        "CBGA": CBGA_best
    }

    pipeline = TuningPipeline(
        tsp_files=TSP_FILES,
        n_seeds=N_SEEDS,
        budget=BUDGET
    )

    pipeline.cargar_instancias()

    
    df_fase3 = pipeline.run_final_comparison(best_configs)
    pipeline.generar_tabla_resumen(df_fase3)
    pipeline.graficar_boxplots(df_fase3)
    pipeline.graficar_curvas_convergencia(df_fase3)


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    main()