import numpy as np
import sagemath as sm
import hashlib
from utils import discrete_gaussian, gadget_matrix, binary_representation
class IBE:
    def __init__(self, n = 10 , q = 10003, lamb = 10):
        self.lamb = lamb
        self.n = n
        self.q = q
        self.k = int(np.ceil(np.log2(q)))
        self.m = self.n* self.k
        self.lamb = lamb
        self.m_star = self.m + self.lamb
        self.sigma = 10
        self.t_factor = 5
        A, R, T = self.generate_master_keys() 
        self.mpk = A
        self.msk = T

    def generate_master_keys(self):
        return self.sample_trapdoor_matrix(self.n,self.m_star, self.q, self.k)
    
    def generate_secret_key(self, v):
        bin_rep = binary_representation(v, self.k)
        users_secret_key = (self.msk @ bin_rep) % self.q
        return users_secret_key
    

    def encrypt(self, message, id):
        #v = self.Hash(id,self.n)
        v = id
        temp_n,temp_m = self.mpk.shape
        s = np.random.uniform(low=-self.q +1, high= self.q-1, size=(temp_n, 1)).astype(int)

        # errors
        error_term = discrete_gaussian( s = self.sigma, shape = (temp_m,1), t_factor=self.t_factor) 
        e_prime = discrete_gaussian(s = self.sigma, shape = (1), t_factor=self.t_factor) 
        # print("error term",error_term)
        # print("matmul", np.matmul(s.T,self.mpk))
        # print(s.T,self.mpk)
        c1 = np.matmul(s.T,self.mpk) + error_term.T

        c2 = np.matmul(s.T,v) + e_prime + message * np.floor(self.q/2)

        return (c1, c2, error_term, e_prime)


    def decrypt(self, ciphertext, key):
        c1,c2 = ciphertext
        alpha = c2 - c1 @ key
        alpha= alpha % self.q
        #print("alpha",alpha)
        message = 0
        if  abs(alpha - self.q/2)< self.q/4:
            message = 1
        return (message, alpha)
    

    def sample_trapdoor_matrix(self,n, m_star, q, k):
        """Sample A with an inhomogeneous trapdoor using a discrete Gaussian sampler for R."""
        B = np.random.randint(0, q, size=(n, m_star))  # Random B matrix n*m_star
        #sigma = np.sqrt(k)  # Standard deviation for Gaussian sampling
        # R = np.rint(norm.rvs(scale=sigma, size=(m_star, n*k))).astype(int) % q  # Gaussian sampled R matrix m_star*nk
        R = discrete_gaussian(s = self.sigma, shape = (m_star, n*k), t_factor=self.t_factor) 
        G = gadget_matrix(n, k)  # Gadget matrix G n*nk
        A = np.hstack((B, G - (B @ R) % q )) % q # Compute A n* (m_star+nk)
        I = np.eye(n*k)
        T = np.vstack((R,I))
        return A, R, T       
    
    def identity_to_vector(self, identity, n, q):
        """Hash an identity string and expand it into n pieces in Z_q^n."""
        
        # 1️⃣ Use SHAKE-256 to generate enough randomness
        hash_bytes = hashlib.shake_256(identity.encode()).digest(n * 4)  # 4 bytes per element
        
        # 2️⃣ Convert to integers mod q
        v = np.array([int.from_bytes(hash_bytes[i * 4 : (i + 1) * 4], "big") % q for i in range(n)])
        
        return v