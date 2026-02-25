from utils import matriz_distancias


class TSPInstance:
    def __init__(self, nombre, dimension, coordenadas, optimo=None):
        self.nombre = nombre
        self.dimension = dimension
        self.coordenadas = coordenadas
        self.optimo = optimo
        self.distancias = self._calcular_distancias()

    def _calcular_distancias(self):
        return matriz_distancias(self.coordenadas)