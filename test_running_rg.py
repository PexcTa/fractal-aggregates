from src.fractal_aggregate import FractalAggregate
import numpy as np

# Create an instance with a specific random seed
generator = FractalAggregate(random_seed=123)

# Generate a medium-sized aggregate
particles = generator.generate_aggregate(
    N=300,
    bias_factors=(1, 1, 1),
    overlap=0.0,
    inactivation_probability=0.5,
    visualize=True
)

# Calculate and plot running Rg
N_list, Rg_list, df_v1, df_err = generator.calculate_and_plot_running_rg(
    particles,
    figsize=(5, 4),
    fit_start_N=10,
    visualize=True
)

print(f"\nResults from running Rg analysis:")
print(f"Mass fractal dimension (df_v1): {df_v1:.4f} ± {df_err:.4f}")

# Compare with structure factor method
print("\nNow calculating fractal dimension using structure factor method...")
q_array, S_array, Rg_sf, alpha, alpha_err = generator.calculate_structure_factor(
    particles,
    q_min=0.01,
    q_max=10.0,
    n_q=100,
    R_particle=1.0,
    fit_range_factor=2.0,
    visualize=True
)

print(f"\nComparison of fractal dimension methods:")
print(f"Running Rg method (df_v1): {df_v1:.4f} ± {df_err:.4f}")
print(f"Structure Factor method (df_v2): {alpha:.4f} ± {alpha_err:.4f}")
print(f"Difference: {abs(df_v1 - alpha):.4f}")