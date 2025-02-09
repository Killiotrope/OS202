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

mandelbrot_set = MandelbrotSet(max_iterations=200, escape_radius=10)
width, height = 1024, 1024
scaleX = 3.0 / width
scaleY = 2.25 / height

if rank == 0:
    # Maître
    start_time = time()  # Début du chrono global

    global_convergence = np.empty((width, height), dtype=np.double)
    cols_restantes = list(range(width))
    esclaves_actifs = size - 1

    # Réception initiale des "prêts" (cad tag=1)
    for _ in range(size - 1):
        comm.recv(source=MPI.ANY_SOURCE, tag=1)

    # Distribution initiale du travail
    for esclave in range(1, size):
        if cols_restantes:
            col = cols_restantes.pop(0)
            comm.send(col, dest=esclave, tag=0)
        else:
            comm.send(-1, dest=esclave, tag=0)
            esclaves_actifs -= 1

    # Réception des messages
    while cols_restantes or esclaves_actifs > 0:
        stat = MPI.Status()
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=stat)
        source = stat.Get_source()
        tag = stat.Get_tag()

        if tag == 1:  # Esclave prêt
            if cols_restantes:
                col = cols_restantes.pop(0)
                comm.send(col, dest=source, tag=0)
            else:
                comm.send(-1, dest=source, tag=0)
                esclaves_actifs -= 1
        elif tag == 0:  # Données reçues
            # On ne peut pas calculer le temps de rassembler les calculs car cela se fait petit à petit
            col, col_data = data
            global_convergence[col, :] = col_data

    # On collecte les temps de calcul
    total_time = time() - start_time  # Fin du chrono global

    compute_times = []
    for esclave in range(1, size):
        comm.send(None, dest=esclave, tag=3)  # Demande d'envoi du temps
        compute_time = comm.recv(source=esclave, tag=2)
        compute_times.append(compute_time)
    
    total_compute_time = sum(compute_times)

    # Génération de l'image
    image_deb = time()
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(global_convergence.T) * 255))
    image_fin = time()
    image.show()
    print(f"Temps total de calcul (de chaque esclaves) : {total_compute_time}s")
    print(f"Temps total de calcul (paralelle) : {total_time}s")
    print(f"Temps du calcul de l'image : {image_fin - image_deb}")

else:
    # Esclave
    compute_time = 0.0

    while True:
        comm.send(None, dest=0, tag=1)  # Signal "prêt"
        col = comm.recv(source=0, tag=0)
        
        if col == -1:
            break

        # Calcul de la colonne
        start_time = time()
        col_data = np.empty(height, dtype=np.double)
        for y in range(height):
            c = complex(-2.0 + scaleX * col, -1.125 + scaleY * y)
            col_data[y] = mandelbrot_set.convergence(c, smooth=True)
        compute_time += time() - start_time

        comm.send((col, col_data), dest=0, tag=0)

    # Attente de la demande du maître pour envoyer le temps
    comm.recv(source=0, tag=3)
    comm.send(compute_time, dest=0, tag=2)