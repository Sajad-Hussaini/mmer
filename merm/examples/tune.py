import numpy as np
from ..core.merm import MERM
from ..core.operator import VLinearOperator, ResidualPreconditioner
from ..lanczos_algorithm import slq

def tune_slq_params(model, X, y, groups, random_slopes):
    """
    A function to help tune SLQ parameters using the V from the first iteration.
    """
    print("--- Initializing model to get the first V_op ---")
    # This setup code is copied from the start of your fit() method
    resid_mrg, rand_effects, resid = model.prepare_data(X, y, groups, random_slopes)
    V_op = VLinearOperator(rand_effects, resid)
    M_op = ResidualPreconditioner(rand_effects, resid) # You need the preconditioner too

    print(f"Matrix V is of size: {V_op.shape}x{V_op.shape}")

    # --- Step 1: Tune n_probes ---
    print("\n--- Tuning n_probes (fixing lanczos_steps=30) ---")
    lanczos_steps_fixed = 30
    for n_probes in [5, 10, 20, 40, 60]:
        # Run it a few times to check stability
        results = [slq(V_op, lanczos_steps_fixed, n_probes, M_op)[0] for _ in range(3)]
        avg_logdet = np.mean(results)
        std_logdet = np.std(results)
        print(f"n_probes={n_probes:2d} -> Log-det: {avg_logdet:10.2f} (std: {std_logdet:8.2f})")

    # --- Step 2: Tune lanczos_steps ---
    n_probes_fixed = 40 # Choose a stable value from the step above
    print(f"\n--- Tuning lanczos_steps (fixing n_probes={n_probes_fixed}) ---")
    for lanczos_steps in [10, 20, 30, 50, 80, 120]:
        log_det, _ = slq(V_op, lanczos_steps, n_probes_fixed, M_op)
        print(f"lanczos_steps={lanczos_steps:3d} -> Log-det: {log_det:10.2f}")

# How to use it:
# model = MERM(...)
# tune_slq_params(model, X, y, groups, random_slopes)