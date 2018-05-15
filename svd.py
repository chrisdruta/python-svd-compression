import numpy as np
import os, PIL, PIL.Image

script_dir = os.path.dirname(__file__)
rel_path = "images/gray.png"
abs_file_path = os.path.join(script_dir, rel_path)

rawImage = PIL.Image.open(abs_file_path)
rawImage.load()

A = np.asarray(rawImage)
Aherm = A.transpose() # Always real-valued, so transpose is fine
AstarA = np.matmul(Aherm, A)

print("Normal check:")

offset1 = np.matmul(A, Aherm) - AstarA
offset2 = AstarA - np.matmul(A, Aherm)

if np.array_equal(offset1, offset2):
    print("Offsets equal")
else:
    print("Offsets not equal")

#AstarA_eigenValues, AstarA_eigenVectors = np.linalg.eig(AstarA)

#for evalue in AstarA_eigenValues:
#    if evalue < 0:
#        print(evalue)