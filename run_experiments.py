import numpy as np
import os
import subprocess
import time

def run_config(lr, threshold, t_limit, init_type="cyclic"):
    print(f"\n--- Testing: LR={lr}, Threshold={threshold}, T_Limit={t_limit}, Init={init_type} ---")
    
    # 1. Update config.py
    config_content = f"""import numpy as np
K = 10
N = 10
DELTA = 5
GAMMA = 5
M_SAMPLES = 5
LAMBDA = 1.0
T_LIMIT = {t_limit}
LR = {lr}
DELTA_THRESHOLD = {threshold}
MAX_ITER = 500
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8
"""
    with open("config.py", "w") as f:
        f.write(config_content)
    
    # 2. Run Main and capture output
    # We use a modified main.py logic here to control initialization
    import importlib
    import config
    import simulation
    import optimization
    import assignment
    
    importlib.reload(config)
    importlib.reload(simulation)
    importlib.reload(optimization)
    importlib.reload(assignment)
    
    from assignment import generate_assignment_matrix
    from simulation import simulate_stragglers
    from optimization import run_optimization
    
    if init_type == "cyclic":
        B_init = generate_assignment_matrix()
    else:
        # Scale to match cyclic density (roughly)
        B_init = np.random.rand(10, 10)
    
    # Ensure current config is used
    C_samples = [simulate_stragglers(B_init) for _ in range(5)]
    
    # Calculate Initial Error
    init_error = 0
    for C in C_samples:
        init_error += np.sum(((B_init * C) @ np.ones(B_init.shape[1]) - 1)**2)
    init_error /= 5
    
    B_opt, r_opt = run_optimization(B_init, C_samples)
    
    # Calculate Final Error
    final_error = 0
    for C, r in zip(C_samples, r_opt):
        final_error += np.sum(((B_opt * C) @ r - 1)**2)
    final_error /= 5
    
    diff = np.sum(np.abs(B_opt - B_init))
    
    print(f"Initial Error: {init_error:.6f}")
    print(f"Final Error: {final_error:.6f}")
    print(f"Total Change in B: {diff:.8f}")
    return init_error, final_error, diff

experiments = [
    (0.01, 1e-6, 3.0, "cyclic"),    # Moderate LR
    (0.1, 1e-7, 3.0, "cyclic"),     # High LR
    (0.01, 1e-7, 10.0, "cyclic"),   # High Time (More data)
    (0.05, 1e-7, 3.0, "random"),    # Random Initialization
]

results = []
for lr, threshold, t_limit, init in experiments:
    # We need to reload modules because they import config.py at top level
    # For a simple script, we'll just use the logic directly
    res = run_config(lr, threshold, t_limit, init)
    results.append(res)

print("\n\n=== Final Observations Summary ===")
for i, res in enumerate(results):
    print(f"Exp {i}: Init={experiments[i][3]}, LR={experiments[i][0]} -> Change in B: {res[2]:.4f}, Error Drop: {res[0]-res[1]:.4f}")
