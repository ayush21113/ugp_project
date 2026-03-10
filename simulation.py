import numpy as np
from config import K, N, DELTA, LAMBDA, T_LIMIT

def simulate_stragglers(B):
    """
    Generates matrix C based on Poisson completion times.
    Now, for each subset assigned to each worker, we draw a separate time.
    """
    C = np.zeros_like(B, dtype=float)
    
    # Draw times for all possible (subset, worker) pairs (K x N)
    all_times = np.random.poisson(LAMBDA, (K, N))
    all_times = np.clip(all_times, 1, None).astype(float)
    
    for j in range(N):
        # Indices of subsets assigned to this worker
        assigned_indices = np.where(B[:, j] == 1)[0]
        
        # Worker processes subsets sequentially. Cumulative time is checked against T_LIMIT.
        cumulative_time = 0.0
        for idx in assigned_indices:
            cumulative_time += all_times[idx, j]
            if cumulative_time <= T_LIMIT:
                C[idx, j] = 1.0
            else:
                break
                
    return C
