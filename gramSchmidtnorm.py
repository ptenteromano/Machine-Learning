# This function does the Gram-Schmidt process
# It transforms a matrix into a matrix of orthonormal vectors

import numpy as np
import numpy.linalg as la


# A => Matrix
def gsBasis(A):
  verySmallNumber = 1e-14
  B = np.array(A, dtype=np.float_)
  
  # loop over vectors in the matrix (column-wise)
  for i in range(B.shape[1]):
    # loop over all previous vectors before 'i'
    for j in range(i):
      # do the GS procedure
      B[:,i] = B[:,i] - (B[:,i] @ B[:,j]) * B[:,j]

    # if vector is linearly independent, normalize
    if la.norm(B[:,i]) > verySmallNumber:
      B[:, i] = B[:, i] / la.norm(B[:, i])
    else:
      B[:,i] = np.zeros_like(B[:,i])
  
  return B

# Some matrix definition
V = np.array([[1,0,2,6],
              [0,1,8,2],
              [2,8,3,1],
              [1,-6,2,3]], dtype=np.float_)

# A non-square matrix
U = np.array([[3,2,3],
              [2,5,-1],
              [2,4,8],
              [12, 2, 1]], dtype=np.float_)

print(gsBasis(V))
Normed = gsBasis(V)

# gsBasis on an orthonormal matrix returns itself
print(gsBasis(Normed))

def dimensions(A) :
    return np.sum(la.norm(gsBasis(A), axis=0))

print(dimensions(V))


# Now let's see what happens when we have one vector that is a linear combination of the others.
C = np.array([[1,0,2],
              [0,1,-3],
              [1,0,2]], dtype=np.float_)
gsBasis(C)