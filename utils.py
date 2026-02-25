import numpy as np
import os
import math

OPTIMOS_CONOCIDOS = {
    "berlin52.tsp": 7542,
    "st70.tsp": 675,
}


# =====================================================
# CARGA INSTANCIA TSP (formato TSPLIB)
# =====================================================

def cargar_tsp(ruta):
    coordenadas = []
    dimension = None
    leyendo = False

    with open(ruta, 'r') as f:
        for linea in f:
            linea = linea.strip()

            if linea.startswith("DIMENSION"):
                dimension = int(linea.split(":")[1])

            elif linea == "NODE_COORD_SECTION":
                leyendo = True
                continue

            elif linea == "EOF":
                break

            elif leyendo:
                partes = linea.split()
                if len(partes) >= 3:
                    _, x, y = partes[:3]
                    coordenadas.append((float(x), float(y)))

    coordenadas = np.array(coordenadas)

    if dimension is None:
        dimension = len(coordenadas)

    nombre = os.path.basename(ruta)
    optimo = OPTIMOS_CONOCIDOS.get(nombre, None)

    return {
        "nombre": nombre,
        "dimension": dimension,
        "optimo": optimo,
        "coordenadas": coordenadas
    }


# =====================================================
# MATRIZ DE DISTANCIAS EUCLIDEA
# =====================================================

def matriz_distancias(coordenadas):
    n = len(coordenadas)
    dist = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i][j] = math.sqrt(
                    (coordenadas[i][0] - coordenadas[j][0])**2 +
                    (coordenadas[i][1] - coordenadas[j][1])**2
                )
    return dist

# =====================================================
# EVALUAR TOUR
# =====================================================

def evaluar_tour(tour, distancias):
    total = 0
    n = len(tour)
    for i in range(n - 1):
        total += distancias[tour[i]][tour[i+1]]
    total += distancias[tour[-1]][tour[0]]
    return total