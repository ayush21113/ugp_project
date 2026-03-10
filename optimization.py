import numpy as np
from config import LR, DELTA_THRESHOLD, MAX_ITER, BETA1, BETA2, EPS

def solve_r(B, C):
    """Solves for optimal r minimizing ||(B*C)r - 1||^2."""
    A = B * C
    # Use least squares to find r
    r, _, _, _ = np.linalg.lstsq(A, np.ones(A.shape[0]), rcond=None)
    return r

def compute_gradient(B, C_samples, r_samples):
    """Computes gradient of error with respect to B."""
    grad = np.zeros_like(B, dtype=float)
    for C, r in zip(C_samples, r_samples):
        y = (B * C) @ r - np.ones(B.shape[0])
        # Grad = 2 * (y * r^T) .* C
        grad += (2 * np.outer(y, r)) * C
    return grad / len(C_samples)

class Adam:
    def __init__(self, shape):
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0
    
    def update(self, B, grad):
        self.t += 1
        self.m = BETA1 * self.m + (1 - BETA1) * grad
        self.v = BETA2 * self.v + (1 - BETA2) * (grad**2)
        m_hat = self.m / (1 - BETA1**self.t)
        v_hat = self.v / (1 - BETA2**self.t)
        return B - LR * m_hat / (np.sqrt(v_hat) + EPS)

def run_optimization(B_init, C_samples):
    """Alternating optimization of B and r."""
    B = B_init.astype(float)
    optimizer = Adam(B.shape)
    prev_error = float('inf')
    
    for i in range(MAX_ITER):
        # 1. Optimize r for each C sample given B
        r_samples = [solve_r(B, C) for C in C_samples]
        
        # 2. Compute total error
        current_error = 0
        for C, r in zip(C_samples, r_samples):
            current_error += np.sum(((B * C) @ r - 1)**2)
        current_error /= len(C_samples)
        
        # 3. Compute gradient
        grad = compute_gradient(B, C_samples, r_samples)
        
        # 4. Check convergence
        diff = abs(prev_error - current_error)
        if diff < DELTA_THRESHOLD:
            print(f"Converged at iteration {i}, Error: {current_error:.6f}, Grad Norm: {np.linalg.norm(grad):.8f}")
            break
        
        if i % 50 == 0:
            print(f"Iteration {i}, Error: {current_error:.6f}, Grad Norm: {np.linalg.norm(grad):.8f}")
            
        prev_error = current_error
        
        # 5. Update B using gradient descent (Adam)
        B = optimizer.update(B, grad)
        
    return B, r_samples
