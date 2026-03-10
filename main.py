import numpy as np
from config import M_SAMPLES
from assignment import generate_assignment_matrix
from simulation import simulate_stragglers

from optimization import run_optimization

def main():
    # 1. Initialize B
    B_init = generate_assignment_matrix()
    
    # 2. Generate C samples
    print(f"Generating {M_SAMPLES} straggling samples (C matrices)...")
    C_samples = [simulate_stragglers(B_init) for _ in range(M_SAMPLES)]
    
    # 3. Initial Error
    r_ones = [np.ones(B_init.shape[1]) for _ in range(M_SAMPLES)]
    init_error = 0
    for C in C_samples:
        init_error += np.sum(((B_init * C) @ np.ones(B_init.shape[1]) - 1)**2)
    init_error /= M_SAMPLES
    print(f"Initial Error (with r=1): {init_error:.6f}")
    print("\nInitial B:")
    print(B_init)
    
    # 4. Run Optimization
    print("\nStarting Alternating Optimization...")
    B_opt, r_opt = run_optimization(B_init, C_samples)
    
    # 5. Final Results
    final_error = 0
    for C, r in zip(C_samples, r_opt):
        final_error += np.sum(((B_opt * C) @ r - 1)**2)
    final_error /= M_SAMPLES
    
    print(f"\nOptimization Complete.")
    print(f"Final Error: {final_error:.6f}")
    print("\nOptimized B (rounded to 2 decimal places):")
    print(np.round(B_opt, 2))
    
    diff_B = np.sum(np.abs(B_opt - B_init))
    print(f"\nTotal Absolute Change in B: {diff_B:.8f}")

if __name__ == "__main__":
    main()
