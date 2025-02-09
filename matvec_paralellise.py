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

# N_loc et start_col
N_loc = dim // size
start_col = rank * N_loc

# Génère des matrices A_local qui sont des portions de N_loc colonnes de A
A_local = np.array([[(i + (start_col + j)) % dim + 1.0 for j in range(N_loc)] for i in range(dim)], dtype=np.float64)

# Génère des matrices u_local qui sont des portions de N_loc lignes de u
u_local = np.array([(start_col + j + 1.0) for j in range(N_loc)], dtype=np.float64)

# Faire les produits
debut = time()
local_v = A_local.dot(u_local)

# Allreduce pour sommer chaque local_v dans global_v
global_v = np.zeros(dim, dtype=np.float64)
comm.Allreduce(local_v, global_v, op=MPI.SUM)
fin = time() - debut

# Ecriture des résultats pour un seul process
if rank == 0:
    print(f"v = {global_v}")
    print(f"Temps de calcul : {fin}")