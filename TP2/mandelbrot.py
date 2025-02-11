# Calcul de l'ensemble de Mandelbrot en python
import numpy as np
from dataclasses import dataclass
from PIL import Image
from math import log
from time import time
import matplotlib.cm
from mpi4py import MPI


@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius:  float = 2.0

    def __contains__(self, c: complex) -> bool:
        return self.stability(c) == 1

    def convergence(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.count_iterations(c, smooth)/self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value

    def count_iterations(self, c: complex,  smooth=False) -> int | float:
        z:    complex
        iter: int

        # On vérifie dans un premier temps si le complexe
        # n'appartient pas à une zone de convergence connue :
        #   1. Appartenance aux disques  C0{(0,0),1/4} et C1{(-1,0),1/4}
        if c.real*c.real+c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (c.real+1)*(c.real+1)+c.imag*c.imag < 0.0625:
            return self.max_iterations
        #  2.  Appartenance à la cardioïde {(1/4,0),1/2(1-cos(theta))}
        if (c.real > -0.75) and (c.real < 0.5):
            ct = c.real-0.25 + 1.j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5*(1-ct.real/max(ctnrm2, 1.E-14)):
                return self.max_iterations
        # Sinon on itère
        z = 0
        for iter in range(self.max_iterations):
            z = z*z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return iter + 1 - log(log(abs(z)))/log(2)
                return iter
        return self.max_iterations


# MPI initialisation
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# On peut changer les paramètres des deux prochaines lignes
mandelbrot_set = MandelbrotSet(max_iterations=2000, escape_radius=10)
width, height = 1024, 1024

scaleX = 3./width
scaleY = 2.25/height

# Gérer les colonnes
columns_per_process = width // size
start_col = rank * columns_per_process
end_col = (rank + 1) * columns_per_process if rank != size - 1 else width


# Calcul de l'ensemble de mandelbrot :

local_convergence = np.empty((end_col - start_col, height), dtype=np.double)
deb = time()
for y in range(height):
    local_x = 0  # Index local démarrant à 0 pour chaque processus
    for x in range(start_col, end_col):
        c = complex(-2. + scaleX*x, -1.125 + scaleY * y)
        local_convergence[local_x, y] = mandelbrot_set.convergence(c, smooth=True)
        local_x += 1  # Incrémenter l'indice local
calcul = time() - deb

# Rassembler les résultats
global_convergence = None
if rank == 0:
    global_convergence = np.empty((width, height), dtype=np.double)

rassembler_deb = time()
comm.Gather(local_convergence, global_convergence, root=0)
rassembler_fin = time() - rassembler_deb

if rank == 0:
    # Créer l'image
    image_deb = time()
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(global_convergence.T) * 255))
    image_fin = time()
    image.show()
    print(f"Temps total de calcul (parallèle) : {calcul}")
    print(f"Temps total pour rassembler les calculs : {rassembler_fin}")
    print(f"Temps du calcul de l'image : {image_fin - image_deb}")




