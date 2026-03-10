import numpy as np
from config import K, N, DELTA, GAMMA

def generate_assignment_matrix():
    """Generates KxN matrix B where B[i,j]=1 if subset i is assigned to worker j."""
    B = np.zeros((K, N), dtype=int)
    # Simple cyclic assignment for demonstration
    # Each subset is replicated GAMMA times
    # Each worker is assigned DELTA subsets
    for j in range(N):
        for d in range(DELTA):
            subset_idx = (j * DELTA + d) % K
            B[subset_idx, j] = 1
    return B
