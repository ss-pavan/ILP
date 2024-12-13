import math
import numpy as np
import logging
import time
from cplex import Cplex
import sys


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

def initial_tableau(c_T,A,b):
    num_constraints = len(A)
    # Add slack variables (Identity matrix)
    I = np.eye(num_constraints)  # Slack variables
    
    # Construct the full tableau
    # Combine A with the slack variables and b
    tableau = np.hstack((A, I, b.reshape(-1, 1)))  # Add A, slack variables (I), and b
    
    # Add the objective function row (negated c and 0s for slack variables)
    c_row = np.hstack((c_T, np.zeros(num_constraints + 1)))  # Objective function row (-c and 0's for slacks and b part)
    initial_simplex_tableau = np.vstack((tableau, c_row))  # Add objective row at the below
    print("initial tableau : ",initial_simplex_tableau)    
    return initial_simplex_tableau

def pivot_column(AM):
  #index of max of last row
  print(f"Last row :\n{AM[-1,:]}")
  #print("column", np.argmax(AM[-1,:][:-1]))
  #print(AM[-1][np.argmax(AM[-1,:][:-1])])
  return np.argmax(AM[-1,:][:-1])
  #last_row = AM[-1, :-1]  # Last row, exclude the RHS column
  #max_value = np.max(last_row)
  #max_indices = np.where(last_row == max_value)[0]
  #print(max_indices)
  #return max_indices[0]

def pivot_row(AM, column):
  #ratio matrix
  RM = np.array([], dtype = np.float64)
  print("debugggggggggggggggggggggggggggggggg")
  print(AM[:,-1])
  print(AM[:,column])
  with np.errstate(divide='ignore', invalid='ignore'):
    RM = AM[:,-1] / AM[:,column]

  print(RM)
  RM = np.where((RM >= 0) & (AM[:,column] != 0), RM, np.inf)
  print(f"\nRatio Matrix : {(RM)}")
  print(RM)
  
  if np.all(np.isinf(RM[:-1])):
    return -1 #unbounded

  min_value = np.min(RM[:-1])
  min_indices = np.where(RM == min_value)[0]
  if len(min_indices) > 1:
    values = AM[:,column][min_indices]
    max_value = np.max(values)
    max_indices = np.where(AM[:,column] == max_value)[0]
    return max_indices[0]
  else:
    return min_indices[0]
  #index of smallest quotient
  #return np.argmin(RM[:-1])

def pivot(AM, column, row):

  print(f"\n pivot (row, column) : {row, column}")
  pe = AM[row, column]
  #print(AM)
  AM[row,:] = AM[row,:]/pe #making pivot element as 1
  print(f"\nAfter making pivot element as 1 : \n{AM}")

  for i in range(AM.shape[0]): #do matrix manipulations except pivot row
    if i == row:
      continue
    else:
      mul = AM[i,column]
      #print(mul)

      AM[i,:] = AM[i,:] - mul*AM[row,:]
      print(f"\nAfter {i}th row matrix manipulation : \n{AM}")

  return AM

def count_flops(AM, column, row):
    # Floating-point operations for pivot
    c = AM.shape[1]  # Number of columns
    r = AM.shape[0]
    tc = 0
    tc += c-1
    tc += r + r + r + r-1
    tc += c + (c + c)*(r - 1)

    # One division to normalize the row, and `l` multiplications and subtractions per row

    return tc

def simplex(c_T,A,b):
  start_time = time.time()
  #AM = to_equations(c_T,A,b)

  AM = initial_tableau(c_T,A,b)
  xc = np.array([0]*(AM.shape[1]))
  cov = np.array([0]*AM.shape[1])
  eps = 1e-10
  total_flops = 0
  it = 0
  while np.any(AM[-1,:]>eps):
      print("Iteration: ",it)
      print(AM[-1])

      col = pivot_column(AM)
      row = pivot_row(AM,col)
      if row == -1:
        print("Unbounded")
        break
      total_flops += count_flops(AM, col, row)
      AM = pivot(AM, col, row)
      xc[col] = row
      cov[col] = 1

      it += 1

  #print(xc)
  #print(cov)
  #x = [AM[:,-1][i]  for i in xc]
  if row != -1:
    x = [AM[:,-1][xc[i]] if cov[i] else 0 for i in range(len(xc))]
    z = abs(AM[:,-1][-1])

    print("Finished")
    print("Solution: x = ",x, "z = ",z)


  end_time = time.time()
  execution_time = end_time - start_time
  if execution_time > 0:
      flops_per_second = total_flops / execution_time
      print(f"Execution Time: {execution_time:.6f} seconds")
      print(f"Total FLOPs: {total_flops}")
      print(f"FLOPS: {flops_per_second:.2e} FLOPS")
  else:
      print("Execution time is too small to measure.")


_log = logging.getLogger(' ')
_log.setLevel(logging.DEBUG)



c = [40,30]
A = [
     [1,1],
     [2,1]
     ]

b = [12,16]
A = np.array(A, dtype=np.float64)
b = np.array(b, dtype=np.float64)
c_T = np.array(c, dtype=np.float64)

mps_file = sys.argv[1]
A,b,c_T = read_mps(mps_file)
print(A)
print(np.count_nonzero(A))
print(b)
print(c_T)
simplex(c_T,A,b)
