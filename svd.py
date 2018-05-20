import numpy as np
import itertools
import os, PIL, PIL.Image

script_dir = os.path.dirname(__file__)
rel_path = "images/gray.png"
abs_file_path = os.path.join(script_dir, rel_path)

rawImage = PIL.Image.open(abs_file_path)
rawImage.load()

A = np.asarray(rawImage)
Aherm = A.transpose() # Always real-valued, so transpose is fine
AstarA = np.matmul(Aherm, A)

AstarA_eigenValues, AstarA_eigenVectors = np.linalg.eig(AstarA)

#for evalue in AstarA_eigenValues:
#    if evalue < 0:
#        print(evalue)

U, s, V = np.linalg.svd(A)
S = np.diagflat(s)

Areconstructed = np.matmul(np.matmul(U, S), V).astype(int)

# Show comparison
print(A)
print(Areconstructed)

std = np.std(A-Areconstructed)
print ("MSE:")
print(std**2)
