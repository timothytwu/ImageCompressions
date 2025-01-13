import numpy as np
import numpy.linalg as npla
from scipy import linalg as spla
import matplotlib.pyplot as plt
np.set_printoptions(precision = 4)

image = plt.imread( "Paris.jpg" )
M = np.float64( image)
U, sigma, Vt = spla.svd(M)
nrows, ncols = M.shape
M_storage = nrows*ncols
Mk = np.zeros(M.shape)
summed = np.sum(sigma)
terms = 0;
preserved = 0
while preserved/summed < 0.99:
    Mk+=sigma[terms] * np.outer( U[:,terms], Vt[terms,:] )
    preserved += sigma[terms]
    terms+=1
print(terms)

#10 terms

terms = 10
Mtemp = np.zeros(M.shape)
for i in range(terms):
    Mtemp +=sigma[i] * np.outer( U[:,i], Vt[i,:] )
Mtemp_storage = terms*(nrows + ncols + 1)
print("Compression Ratio",Mtemp_storage/M_storage)
plt.figure(figsize=(20,20))
plt.subplot(1,2,1)
plt.title("Original image")
plt.imshow(M)
plt.subplot(1,2,2)
plt.title(f"Image with {terms} terms")
plt.imshow(Mtemp)

#100 terms

terms = 100
Mtemp = np.zeros(M.shape)
for i in range(terms):
    Mtemp +=sigma[i] * np.outer( U[:,i], Vt[i,:] )
Mtemp_storage = terms*(nrows + ncols + 1)
print("Compression Ratio",Mtemp_storage/M_storage)
plt.figure(figsize=(20,20))
plt.subplot(1,2,1)
plt.title("Original image")
plt.imshow(M)
plt.subplot(1,2,2)
plt.title(f"Image with {terms} terms")
plt.imshow(Mtemp)

#150 terms

terms = 150
Mtemp = np.zeros(M.shape)
for i in range(terms):
    Mtemp +=sigma[i] * np.outer( U[:,i], Vt[i,:] )
Mtemp_storage = terms*(nrows + ncols + 1)
print("Compression Ratio",Mtemp_storage/M_storage)
plt.figure(figsize=(20,20))
plt.subplot(1,2,1)
plt.title("Original image")
plt.imshow(M)
plt.subplot(1,2,2)
plt.title(f"Image with {terms} terms")
plt.imshow(Mtemp)

#200 terms

terms = 200
Mtemp = np.zeros(M.shape)
for i in range(terms):
    Mtemp +=sigma[i] * np.outer( U[:,i], Vt[i,:] )
Mtemp_storage = terms*(nrows + ncols + 1)
print("Compression Ratio",Mtemp_storage/M_storage)
plt.figure(figsize=(20,20))
plt.subplot(1,2,1)
plt.title("Original image")
plt.imshow(M)
plt.subplot(1,2,2)
plt.title(f"Image with {terms} terms")
plt.imshow(Mtemp)

#300 terms

terms = 300
Mtemp = np.zeros(M.shape)
for i in range(terms):
    Mtemp +=sigma[i] * np.outer( U[:,i], Vt[i,:] )
Mtemp_storage = terms*(nrows + ncols + 1)
print("Compression Ratio",Mtemp_storage/M_storage)
plt.figure(figsize=(20,20))
plt.subplot(1,2,1)
plt.title("Original image")
plt.imshow(M)
plt.subplot(1,2,2)
plt.title(f"Image with {terms} terms")
plt.imshow(Mtemp)
