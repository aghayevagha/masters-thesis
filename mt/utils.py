import numpy as np
from user import User

def binary_representation(v, k):
    """Convert vector v into its binary representation."""
    # Ensure k is correct by rounding up
    n = len(v)
    bin_matrix = np.zeros((n * k, 1), dtype=int)
    
    print(f"Converting v = {v} to binary with {k} bits per element")
    
    for i in range(n):
        binary_str = format(v[i], f'0{k}b')  # Convert to binary with k bits
        print(f"v[{i}] = {v[i]} → Binary: {binary_str}")
        bin_matrix[i * k:(i + 1) * k, 0] = [int(b) for b in reversed(binary_str)]
    
    print("Binary representation matrix:")
    print(bin_matrix)
    return bin_matrix

def gadget_matrix(n, k):
    """Constructs the gadget matrix G = I_n ⊗ g where g = [1, 2, ..., 2^(k-1)]"""
     # Ensure k is correct by rounding up
    g = np.array([2**i for i in range(k)])  # g vector
    I_n = np.eye(n, dtype=int)  # Identity matrix
    G = np.kron(I_n, g)  # Compute tensor product
    
    print("Gadget vector g:")
    print(g)
    print("Identity matrix I_n:")
    print(I_n)
    print("Gadget matrix G:")
    print(G)
    
    return G

def verify_gadget_matrix(v, k):
    """Check if G * binary_representation(v) = v."""
    n = len(v)
    G = gadget_matrix(n, k)  # Create G
    bin_v = User.binary_representation(v, k)  # Binary representation of v
    reconstructed_v = G @ bin_v  # Matrix multiplication
    
    print("Reconstructed v:")
    print(reconstructed_v.flatten())
    
    return np.all(reconstructed_v.flatten() == v), reconstructed_v.flatten()




def sample_trapdoor_matrix(n, m_star, q, k):
    """Sample A with an inhomogeneous trapdoor using a discrete Gaussian sampler for R."""
    B = np.random.randint(0, q, size=(n, m_star))  # Random B matrix n*m_star
    sigma = np.sqrt(k)  # Standard deviation for Gaussian sampling
    # R = np.rint(norm.rvs(scale=sigma, size=(m_star, n*k))).astype(int) % q  # Gaussian sampled R matrix m_star*nk
    R = discrete_gaussian(s = sigma, shape = (m_star, n*k), t_factor=10) % q
    G = gadget_matrix(n, k)  # Gadget matrix G n*nk
    A = np.hstack((B, G - (B @ R) % q )) % q # Compute A n* (m_star+nk)
 
    return A, R

def discrete_gaussian(s, shape=(1,), t_factor=10):
    """
    Samples from the discrete Gaussian distribution D_Z,s using rejection sampling.

    Parameters:
    - s: Standard deviation of the Gaussian.
    - shape: Tuple defining the shape of the output array.
    - t_factor: Determines the cut-off interval Z = [-t*s, t*s].

    Returns:
    - A NumPy array of sampled integers with the specified shape.
    """
    size = np.prod(shape)  # Total number of samples needed
    samples = []
    t = t_factor * np.sqrt(np.log(size + 1))  # Ensures negligible probability mass outside
    interval = int(np.ceil(t * s))  # Define the interval [-interval, interval]

    while len(samples) < size:
        z = np.random.randint(-interval, interval + 1)  # Step 1: Sample uniformly
        prob = np.exp(-np.pi * z**2 / s**2)  # Step 2: Compute acceptance probability
        if np.random.rand() < prob:  # Step 3: Accept with probability ρ_s(z)
            samples.append(z)

    return np.array(samples).reshape(shape)  # Reshape into the desired shape



# # Example usage
# q = 29  # Modulus
# k = int(np.ceil(np.log2(q)))
# n=4
# m = k*n*2
# lamb = 5
# m_star = m + lamb 
# v = np.random.randint(0,q,size = (n,))  # Example vector with elements < q
# binary_rep = binary_representation(v,k)


# #is_correct, reconstructed_v = verify_gadget_matrix(v, k)
# #print("Verification passed:", is_correct)
# A, R = sample_trapdoor_matrix(n, m_star, q, k)
# I = np.eye(n*k)
# T = np.vstack((R,I))
# users_key = (T @ binary_rep) % q


# #print(A.shape, T.shape)

# print(v)
# print( (A @ users_key) % q )
