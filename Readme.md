# TD n° 2 - 27 Janvier 2025

## Machine utilisée :

La commande hwloc-ls me donne :
Machine (3535MB total)
  Package L#0
    NUMANode L#0 (P#0 3535MB)
    L2 L#0 (4096KB)
      L1d L#0 (64KB) + L1i L#0 (128KB) + Core L#0 + PU L#0 (P#0)
      L1d L#1 (64KB) + L1i L#1 (128KB) + Core L#1 + PU L#1 (P#1)
      L1d L#2 (64KB) + L1i L#2 (128KB) + Core L#2 + PU L#2 (P#2)
      L1d L#3 (64KB) + L1i L#3 (128KB) + Core L#3 + PU L#3 (P#3)
    L2 L#1 (16MB)
      L1d L#4 (128KB) + L1i L#4 (192KB) + Core L#4 + PU L#4 (P#4)
      L1d L#5 (128KB) + L1i L#5 (192KB) + Core L#5 + PU L#5 (P#5)
      L1d L#6 (128KB) + L1i L#6 (192KB) + Core L#6 + PU L#6 (P#6)
      L1d L#7 (128KB) + L1i L#7 (192KB) + Core L#7 + PU L#7 (P#7)
  CoProc(OpenCL) "opencl0d0"

Avec un processeur M2. J'ai donc 4 coeurs d'économie de batterie et 4 coeurs de performance.


##  1. Parallélisation ensemble de Mandelbrot

L'ensensemble de Mandebrot est un ensemble fractal inventé par Benoit Mandelbrot permettant d'étudier la convergence ou la rapidité de divergence dans le plan complexe de la suite récursive suivante :
$$
\left\{
\begin{array}{l}
    c\,\,\textrm{valeurs\,\,complexe\,\,donnée}\\
    z_{0} = 0 \\
    z_{n+1} = z_{n}^{2} + c
\end{array}
\right.
$$
dépendant du paramètre $c$.

Il est facile de montrer que si il existe un $N$ tel que $\mid z_{N} \mid > 2$, alors la suite $z_{n}$ diverge. Cette propriété est très utile pour arrêter le calcul de la suite puisqu'on aura détecter que la suite a divergé. La rapidité de divergence est le plus petit $N$ trouvé pour la suite tel que $\mid z_{N} \mid > 2$.

On fixe un nombre d'itérations maximal $N_{\textrm{max}}$. Si jusqu'à cette itération, aucune valeur de $z_{N}$ ne dépasse en module 2, on considère que la suite converge.

L'ensemble de Mandelbrot sur le plan complexe est l'ensemble des valeurs de $c$ pour lesquels la suite converge.

Pour l'affichage de cette suite, on calcule une image de $W\times H$ pixels telle qu'à chaque pixel $(p_{i},p_{j})$, de l'espace image, on associe une valeur complexe  $c = x_{min} + p_{i}.\frac{x_{\textrm{max}}-x_{\textrm{min}}}{W} + i.\left(y_{\textrm{min}} + p_{j}.\frac{y_{\textrm{max}}-y_{\textrm{min}}}{H}\right)$. Pour chacune des valeurs $c$ associées à chaque pixel, on teste si la suite converge ou diverge.

- Si la suite converge, on affiche le pixel correspondant en noir
- Si la suite diverge, on affiche le pixel avec une couleur correspondant à la rapidité de divergence.

1. À partir du code séquentiel `mandelbrot.py`, faire une partition équitable par bloc suivant les lignes de l'image pour distribuer le calcul sur `nbp` processus  puis rassembler l'image sur le processus zéro pour la sauvegarder. Calculer le temps d'exécution pour différents nombre de tâches et calculer le speedup. Comment interpréter les résultats obtenus ?
2. Réfléchissez à une meilleur répartition statique des lignes au vu de l'ensemble obtenu sur notre exemple et mettez la en œuvre. Calculer le temps d'exécution pour différents nombre de tâches et calculer le speedup et comparez avec l'ancienne répartition. Quel problème pourrait se poser avec une telle stratégie ?
3. Mettre en œuvre une stratégie maître-esclave pour distribuer les différentes lignes de l'image à calculer. Calculer le speedup avec cette approche et comparez  avec les solutions différentes. Qu'en concluez-vous ?

## Réponse :

1) En exécutant mandelbrot_vec.py, j'obtiens :
Temps du calcul de l'ensemble de Mandelbrot : 2.2320263385772705
Temps de constitution de l'image : 0.01720714569091797

En exécutant mandelbrot.py avec 8 coeurs, j'obtiens :
Temps total de calcul (parallèle) : 0.2878708839416504
Temps total pour rassembler les calculs : 1.133652925491333
Temps du calcul de l'image : 0.02663111686706543

Soit un spedup de 1.57 sur les calculs et leur rassemblement. Je considère la différence de temps sur le calcul de l'image comme normalmenent inchangé car non paralélisé.

On voit donc que c'est bien plus rapide que le programme de base. Cependant on perd beaucoup de temps lorsqu'on rassemble les calculs. On va alors essayer de faire une attribution cyclique des colonnes.


2) Il est plus long de calculé une colonne avec plus de contraste. Une attribution cyclique des colonnes de l'image permet de répartir la charge de travail (load balancing)

En exécutant mandelbrot_cyclique.py, j'obtiens : 
Temps total de calcul (parallèle) : 0.8331449031829834
Temps total pour rassembler les calculs : 0.1098020076751709
Temps ordonnée l'image : 0.0013997554779052734
Temps du calcul de l'image : 0.029397964477539062

Soit un spedup de 2.36 C'est en effet plus rapide que la version de base. Cependant, une telle stratégie peut vite devenir coûteuse si on a de grandes images à cause de la phase de réordonnement.


3) En éxecutant mandelbrot_maitre_esclave.py avec 8 coeurs, j'obtiens :
Temps total de calcul (de chaque esclaves) : 2.404888153076172s
Temps total de calcul (paralelle) : 0.3539412021636963s
Temps du calcul de l'image : 0.040602922439575195

Soit un spedup de 6.31 C'est encore plus rapide. l'algorithme maitre-esclave est le plus efficace. C'est logique, c'est celui où les coeurs les plus performants travaillent le plus.

## 2. Produit matrice-vecteur

On considère le produit d'une matrice carrée $A$ de dimension $N$ par un vecteur $u$ de même dimension dans $\mathbb{R}$. La matrice est constituée des cœfficients définis par $A_{ij} = (i+j) \mod N$. 

Par soucis de simplification, on supposera $N$ divisible par le nombre de tâches `nbp` exécutées.

### a - Produit parallèle matrice-vecteur par colonne

Afin de paralléliser le produit matrice–vecteur, on décide dans un premier temps de partitionner la matrice par un découpage par bloc de colonnes. Chaque tâche contiendra $N_{\textrm{loc}}$ colonnes de la matrice. 

- Calculer en fonction du nombre de tâches la valeur de Nloc
- Paralléliser le code séquentiel `matvec.py` en veillant à ce que chaque tâche n’assemble que la partie de la matrice utile à sa somme partielle du produit matrice-vecteur. On s’assurera que toutes les tâches à la fin du programme contiennent le vecteur résultat complet.
- Calculer le speed-up obtenu avec une telle approche

## Réponse

N_loc = dim // size

En exécutant matvec.py, j'obtiens : 
Temps de calcul : 1.4781951904296875e-05

En exécutant matvec_paralellise.py, j'obtiens :
Temps de calcul : 0.0018069744110107422

Soit un spedup de 0.008 C'est pas très efficace. Peut-être que mon code fonctionne mal ou que le calcul matrice vecteur est déjà optimisé.

### b - Produit parallèle matrice-vecteur par ligne

Afin de paralléliser le produit matrice–vecteur, on décide dans un deuxième temps de partitionner la matrice par un découpage par bloc de lignes. Chaque tâche contiendra $N_{\textrm{loc}}$ lignes de la matrice.

- Calculer en fonction du nombre de tâches la valeur de Nloc
- paralléliser le code séquentiel `matvec.py` en veillant à ce que chaque tâche n’assemble que la partie de la matrice utile à son produit matrice-vecteur partiel. On s’assurera que toutes les tâches à la fin du programme contiennent le vecteur résultat complet.
- Calculer le speed-up obtenu avec une telle approche

## Réponse

N_loc = dim // size

En exécutant matvec_paralellise_ligne.py, j'obtiens :
Temps de calcul : 0.009273052215576172

Soit un spedup de 0.002 c'est encore une fois très faible.

## 3. Entraînement pour l'examen écrit

Alice a parallélisé en partie un code sur machine à mémoire distribuée. Pour un jeu de données spécifiques, elle remarque que la partie qu’elle exécute en parallèle représente en temps de traitement 90% du temps d’exécution du programme en séquentiel.

En utilisant la loi d’Amdhal, pouvez-vous prédire l’accélération maximale que pourra obtenir Alice avec son code (en considérant n ≫ 1) ?

À votre avis, pour ce jeu de donné spécifique, quel nombre de nœuds de calcul semble-t-il raisonnable de prendre pour ne pas trop gaspiller de ressources CPU ?

En effectuant son cacul sur son calculateur, Alice s’aperçoit qu’elle obtient une accélération maximale de quatre en augmentant le nombre de nœuds de calcul pour son jeu spécifique de données.

En doublant la quantité de donnée à traiter, et en supposant la complexité de l’algorithme parallèle linéaire, quelle accélération maximale peut espérer Alice en utilisant la loi de Gustafson ?


## Réponse :

1) D'après la loi d'Amdhal pour n >> 1. On a S = 1/f avec f la partie du code qui ne peut être parallelisé. Donc S = 10 au mieux.

2) En calculant l'efficacité on remarque qu'on descends en dessous de 50 % d'efficacité à partir de n= 11. Pour moi il semble raisonnable de prendre 10 ou 11 noeuds pour avoir un speed up de 5,26 ou 5,50.

3) Si Alice avait un speed up de 4, cela signifie qu'elle utilise n = 6 noeuds. La loi de Gustafson nous donne
S = n + (1-n)*t_s, où t_s est le temps pris pour exécuter en séquentiel. Or on a t_s + t_p = 1 (une unité de temps). t_p prend 90% du temps donc finalemnt t_s = 0,1. Le calcul donne S = 5,5.