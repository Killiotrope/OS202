# Produit matrice-vecteur v = A.u
import numpy as np
from mpi4py import MPI
from time import time

# Dimension du problème (peut-être changé)
dim = 120
# Initialisation de la matrice
A = np.array([[(i+j) % dim+1. for i in range(dim)] for j in range(dim)])
print(f"A = {A}")

# Initialisation du vecteur u
u = np.array([i+1. for i in range(dim)])
print(f"u = {u}")

# MPI initialisation
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# N_loc et start_row
N_loc = dim // size
start_row = rank * N_loc

# Génère des matrices A_local qui sont des portions de N_loc lignes de A
A_local = np.array([[(j + (start_row + i)) % dim + 1 for j in range(dim)] for i in range(N_loc)], dtype=np.float64)


# Faire les produits
debut = time()
local_v = A_local.dot(u)

# Allgather pour que tout le monde est les infos
global_v = np.zeros(dim, dtype=np.float64)
comm.Allgather(local_v, global_v)
fin = time() - debut

# Ecriture des résultats pour un seul process
if rank == 0:
    print(f"v = {global_v}")
    print(f"Temps de calcul : {fin}")

