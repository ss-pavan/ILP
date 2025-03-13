import sys
import numpy as np
from cplex import Cplex



def to_equations (c_T, A, b):

  l = len(A)
  AM= [[]]*(l+1)
  for i in range(l):
    AM[i] = A[i] + [0]*i + [1] + [0]*(l-1-i) + b[i]

  AM[-1] = c_T + [0]*(l) + [0]
  #print(AM)
  AM = np.array(AM, dtype=np.float64)
  print(f"\nInitial matrix : \n{AM}")

  return AM

def read_mps(file):
    # Initialize CPLEX model
    model = Cplex()
    model.read(file)


    # Get the number of variables and constraints
    num_vars = model.variables.get_num()
    num_constraints = model.linear_constraints.get_num()

    # Initialize the A matrix, b vector, and c vector
    A = np.zeros((num_constraints, num_vars))
    b = np.array(model.linear_constraints.get_rhs())  # RHS is the b vector
    c = np.array(model.objective.get_linear())  # Linear coefficients in the objective function are the c vector

    # Fill the A matrix
    for i in range(num_constraints):
        row = model.linear_constraints.get_rows(i)
        for j, col_idx in enumerate(row.ind):
            A[i, col_idx] = row.val[j]  # Populate A matrix with the coefficients

    return A, b, c


def print_matrix(matrix):
    np.savetxt(sys.stdout, matrix, fmt='%.2f', delimiter='  ')

def ph1_tableau(n, m, b, c, A):
    tableau =  np.zeros(shape=(m+1, n+1+m), dtype=np.float128)
    tableau[1:, 0] = np.array(b)  ### initial basic variables are the slack variables
    tableau[1:m+1, 1:n+1] = np.array(A) ### original constraints
    tableau[1:m+1, n+1:] = np.identity(m) ### appended slack constraints

    for constraint_idx in range(1, m+1):
        if np.round(tableau[constraint_idx, 0], 7) < 0:
            tableau[constraint_idx, :] *= -1


    extra = np.vstack((np.zeros(m), np.identity(m)))
    tableau = np.hstack((tableau, extra))
    A_new = tableau[1:, 1:]
    c = np.zeros(shape=(n+m+m), dtype=np.float128)
    cb = np.ones(shape=(m), dtype=np.float128)
    c[n+m:] = np.ones(shape=(m), dtype=np.float128)
    c_red = c - cb@A_new
    tableau[0, 0] = -np.sum(tableau[1:, 0])
    tableau[0, 1:] = c_red
    art_ind = np.zeros(m)
    for i in range(0, m):
        art_ind[i] = 1+i+n+m

    return tableau, art_ind

def ph2_tableau(tableau, art_ind, n, m, c):
    
    while True:
        col = -1
        row = -1
        for i, index in enumerate(art_ind):
            if index >= m + n + 1:
                row = i+1
                break
        if row == -1:
            break

        for i in range(1, m+n+1):
            if np.round(tableau[row, i], 7) != 0:
                col = i
                break
        
        if col == -1:
            # remove the lth row
            print_matrix(tableau)
            print(f"Found redundant constraint")
            tableau = np.delete(tableau, (row), axis=0)
            art_ind = np.delete(art_ind, (row-1))
            m-=1
            
        else:
            # remove the pivot row and add the pivot column

            art_ind[row-1] = col            
            tableau[row, :] /= tableau[row][col]
            for row_idx in range(tableau.shape[0]):
                if(row_idx != row):
                    tableau[row_idx, : ] -= tableau[row, :] * tableau[row_idx][col]
            


    # remove the excess columns
    # tableau = np.round(tableau, 7)    
    tableau = tableau[:, :m+n+1]
    
    # recompute reduced costs and original cost
    c_new = np.hstack((c, np.zeros(m)))
    cb = np.zeros(m)
    A  = tableau[1:, 1:]
    Ab = np.zeros((m, m))

    for i, index in enumerate(art_ind):
        Ab[:, i] =A[: , int(index)-1]
        cb[i] = c_new[int(index)-1]
    c_red = c_new - cb@np.linalg.inv(Ab)@A

    tableau[0][0] = -cb@tableau[1:, 0]
    tableau[0][1:] = c_red
    # tableau = np.round(tableau, 7)
    return tableau, art_ind
        
             
def simplex(tableau, art_ind):
    
    eps = 1e-10
    optimal_found = not np.any(np.round(tableau[0, 1:], 7) < 0.0 ) ### is any reduced cost < 0 
    iterations = 0
    while(not optimal_found):
        print("-"*50)
        col = np.argmax(np.round(tableau[0, 1:], 7) < 0.0) + 1
        with np.errstate(divide='ignore', invalid='ignore'):
            RM = tableau[1:, 0]/tableau[1:,col]
        RM[np.round(tableau[1:,col], 7)<0] = np.inf
        row = np.where(np.logical_and(np.round(RM, 7)>=0, RM==np.amin(RM[np.round(RM, 7) >=0])))[0][0]+1
        art_ind[row-1] = col
        print(f"Pivot: ({row}, {col})")
        tableau[row, :] /= tableau[row][col]
        for row_idx in range(tableau.shape[0]):
            if(row_idx != row):
                tableau[row_idx, : ] -= tableau[row, :] * tableau[row_idx][col]
        iterations += 1
        # tableau = np.round(tableau, 7)
        print(f"Table After {iterations} iterations of Primal Tableau")
        print_matrix(tableau)
        optimal_found = not np.any(np.round(tableau[0, 1:], 7) < 0.0 )
        
    return tableau, art_ind
      


#c = [40,30]
#A = [
#     [1,1],
#     [2,1]
#     ]
#
#b = [12,16]


mps_file = sys.argv[1]
A,b,c = read_mps(mps_file)

#c = [2,1,3]
#A = [
#     [1,1,3],
#     [2,2,5],     [4,1,2]
#     ]
#b = [30, 24, 36]
n = len(c)
m = len(b)


tableau, art_ind = ph1_tableau(n, m, b, c, A)
print_matrix(tableau)
tableau, art_ind = simplex(tableau, art_ind)
if np.round(tableau[0][0], 7) != 0:
    print(f"No feasible solution exists")
else:

    tableau, art_ind = ph2_tableau(tableau, art_ind, n, m, c)

    # 2nd phase of simplex
    tableau, art_ind = simplex(tableau, art_ind)

