import numpy as np
from config import K, N, DELTA, LAMBDA, T_LIMIT

def simulate_stragglers(B):
    """
    Generates matrix C based on Poisson completion times.
    For each worker, we draw a time per subset from Poisson(LAMBDA).
    Time must be > 0.
    """
    C = np.zeros_like(B)
    
    # Random draw for each worker (time per subset)
    # Poisson can return 0, but time must be > 0. We use max(1, draw).
    times_per_subset = np.random.poisson(LAMBDA, N)
    times_per_subset = np.clip(times_per_subset, 1, None).astype(float)
    
    for j in range(N):
        # Indices of subsets assigned to this worker
        assigned_indices = np.where(B[:, j] == 1)[0]
        
        # How many can they complete in T_LIMIT?
        num_completed = int(T_LIMIT // times_per_subset[j])
        
        # Mark the first few as completed
        completed_indices = assigned_indices[:num_completed]
        C[completed_indices, j] = 1
        
    return C
