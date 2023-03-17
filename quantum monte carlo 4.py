import numpy as np
import matplotlib.pyplot as plt


# Calculate the potential energy of the particle
def potential_energy(x):
    return 0.5*K*x**2

# Define the Metropolis Monte Carlo algorithm
def metropolis(x, delta, T): # T is an array this is why youre getting the error
    x_new = x + np.random.uniform(-delta, delta)
    delta_E = potential_energy(x_new) - potential_energy(x)
    if delta_E < 0 or np.random.rand() < np.exp(-delta_E/(kb*T)):
        return x_new
    else:
        return x
#delta depends on the the where the particle is in the well and the energy 
# it is receiving but delta = 1 for now


# Extra Elements from an array every so often (used code from this link: https://stackoverflow.com/questions/50685409/select-n-evenly-spaced-out-elements-in-array-including-first-and-last)
def spac_arr(arr, numElems):
    idx = np.round(np.linspace(0, len(arr) - 1, numElems)).astype(int)
    idx = np.linspace(0, len(arr) - 1, numElems).astype(int)
    idx = np.linspace(0, len(arr) - 1, numElems, dtype='int')
    return idx

def x_expectation(x):
    # Define the number of blocks to create
    num_blocks = 10
    
    # Calculate the size of each block
    block_size = len(positions) // num_blocks
    
    # Reshape the array into a 2D array with the desired number of rows
    # and the calculated block size for the number of columns
    arr_blocks = positions.reshape(num_blocks, block_size)
     
    # Calculate the mean for each row (i.e. each block)
    block_means = np.mean(arr_blocks, axis=1)
     
    # Remove the first block
    block_means = block_means[1:] 
    
    # take the mean of the block means
    expectation_x = np.mean(block_means)
 
    # Calculate the standard deviation of the expectation
    s_x = np.std(block_means)/np.sqrt(num_blocks-1-1)
    
    return expectation_x, s_x

def xsqr_expectation(x):
    sqr_positions = positions**2
    # Define the number of blocks to create
    num_blocks = 10
    
    # Calculate the size of each block
    block_size = len(positions) // num_blocks
     
    # Reshape the array into a 2D array with the desired number of rows
    # and the calculated block size for the number of columns
    sqr_arr_block = sqr_positions.reshape(num_blocks, block_size)
    
    # Calculate the mean for each row (i.e. each block)
    sqr_block_means = np.mean(sqr_arr_block, axis=1)
    
    # Remove the first block
    sqr_block_means = sqr_block_means[1:]
    
    # take the mean of the block means
    expectation_xsqr = np.mean(sqr_block_means)
    
    # Calculate the standard deviation of the expectation 
    s_xsqr = np.std(sqr_block_means)/np.sqrt(num_blocks-1-1)
    
    return expectation_xsqr, s_xsqr

# Set up the parameters
N = 1000        # Number of steps
T = 0.1          # Temperature
K = 1.0         # Spring Constant
kb = 1.0         # Boltzmann constant
delta = 1.0      # Step size
x = 0.0         # Initial position

# Run the simulation
positions = np.zeros(N)
for i in range(N):
    x = metropolis(x, delta, T)
    positions[i] = x


# Expected Values for Statistics
print((kb*T/K), 0, (kb*T/K))
# Statistics
print(x_expectation(positions))
print(xsqr_expectation(positions))


# Plot the results
plt.plot(positions)
plt.xlabel('Step')
plt.ylabel('Position')
plt.show()

print()
#%% Graphs of expectation values versus T

# Set up the parameters
N = 1000        # Number of steps
T =  np.linspace(0.1, 3.0, 50)      # Temperature
K = 1.0         # Spring Constant
kb = 1.0         # Boltzmann constant
delta = 1.0      # Step size
x = 0.0         # Initial position

# Run the simulation for different T and get arrays for expectation values
# a is the array of the x expectations and errors at different T
# b is the array for the x^2 expectations and errors at different T
a = np.zeros(np.size(T))
b = np.zeros(np.size(T))
a_error = np.zeros(np.size(T))
b_error = np.zeros(np.size(T))

                           
n=0
for count in T:
    positions = np.zeros(N)
    for i in range(N):
        x = metropolis(x, delta, count)
        positions[i] = x
    arr1 = x_expectation(positions)
    arr2 = xsqr_expectation(positions)
    a[n]= arr1[0]
    b[n] = arr2[0]
    a_error[n] = arr1[1]
    b_error[n] = arr2[1]
    n+=1

print(a)
plt.plot(T, kb*T/K)
plt.errorbar(T, a,
             yerr=a_error,
             capsize=4)
print(b)
#plt.plot(b)
plt.plot(T, np.zeros(np.size(T)))
plt.errorbar(T, b,
             yerr=b_error,
             capsize=4)


