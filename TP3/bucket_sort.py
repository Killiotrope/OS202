from mpi4py import MPI
import numpy as np

# MPI initialisation
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

dim = 1000
max =100
r = int(max/dim)

X = np.random.randint(1, max, dim)

X.sort()
# print(X)
quantils = np.quantile(X, np.linspace(0, 1, size + 1))
# print(f"Quantils : {quantils}")

quantil_global = np.zeros((size+1)*size, dtype="float")
comm.Allgather(quantils, quantil_global)
quantil_global.sort()
quantils_all = np.quantile(quantil_global, np.linspace(0, 1, size + 1))
# print(quantils_all)


buckets = []
for i in range(size):
    buckets.append(X[(X >= quantils_all[i]) & (X <= quantils_all[i+1])])

# Affichage des résultats
# print(f"Rank {rank} buckets:")
# for i, bucket in enumerate(buckets):
   #  print(f"Bucket {i}: {bucket}")

buckets_all = comm.alltoall(buckets)
buckets_all = np.concatenate(buckets_all)
buckets_all.sort()
# print(f"Bucket_all {rank}: {buckets_all}")


Final = comm.gather(buckets_all, root=0)

# Le process 0 affiche le résultat
if rank == 0:
    Final = np.concatenate(Final)
    print("Tableau trié", Final)