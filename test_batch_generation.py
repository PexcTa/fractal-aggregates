from src.fractal_aggregate import FractalAggregate
import matplotlib.pyplot as plt
import numpy as np

# Create an instance with a specific random seed
generator = FractalAggregate(random_seed=42)

# Run parameter sweep over inactivation probability p
print("\n=== Running parameter sweep over p ===")
p_results = generator.run_parameter_sweep_p(
    N=100,
    p_range=(0.0, 0.8),
    p_steps=9,
    samples_per_p=5,
    bias_factors=(1, 1, 1),
    overlap=0.0
)

# Run parameter sweep over number of particles N
print("\n=== Running parameter sweep over N ===")
N_results = generator.run_parameter_sweep_N(
    N_range=(10, 200),
    N_steps=8,
    samples_per_N=3,
    p=0.3,
    bias_factors=(1, 1, 1),
    overlap=0.0
)

# Create visualization of results
fig, axes = plt.subplots(2, 2, figsize=(6, 5))
fig.suptitle('Parameter Sweep Results', fontsize=16)

# Plot 1: df vs p
axes[0, 0].plot(p_results['p_values'], p_results['df_v2_values'], 'o-', color='blue', linewidth=2, markersize=8)
axes[0, 0].set_xlabel('Inactivation Probability (p)', fontsize=12)
axes[0, 0].set_ylabel('Mass Fractal Dimension (df)', fontsize=12)
axes[0, 0].set_title('Fractal Dimension vs Inactivation Probability', fontsize=14)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim(1.5, 3.1)

# Plot 2: Rg vs p
axes[0, 1].plot(p_results['p_values'], p_results['Rg_values'], 'o-', color='green', linewidth=2, markersize=8)
axes[0, 1].set_xlabel('Inactivation Probability (p)', fontsize=12)
axes[0, 1].set_ylabel('Radius of Gyration (Rg)', fontsize=12)
axes[0, 1].set_title('Radius of Gyration vs Inactivation Probability', fontsize=14)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: df vs N
axes[1, 0].semilogx(N_results['N_values'], N_results['df_v2_values'], 'o-', color='red', linewidth=2, markersize=8)
axes[1, 0].set_xlabel('Number of Particles (N)', fontsize=12)
axes[1, 0].set_ylabel('Mass Fractal Dimension (df)', fontsize=12)
axes[1, 0].set_title('Fractal Dimension vs Number of Particles', fontsize=14)
axes[1, 0].grid(True, which="both", alpha=0.3)
axes[1, 0].set_ylim(1.5, 3.1)

# Plot 4: Porosity vs p
axes[1, 1].plot(p_results['p_values'], p_results['porosities'], 'o-', color='purple', linewidth=2, markersize=8)
axes[1, 1].set_xlabel('Inactivation Probability (p)', fontsize=12)
axes[1, 1].set_ylabel('Porosity (ϵ)', fontsize=12)
axes[1, 1].set_title('Porosity vs Inactivation Probability', fontsize=14)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print summary statistics
print("\n=== Summary Statistics ===")
print(f"p sweep results - df range: {np.min(p_results['df_v2_values']):.3f} to {np.max(p_results['df_v2_values']):.3f}")
print(f"p sweep results - Rg range: {np.min(p_results['Rg_values']):.3f} to {np.max(p_results['Rg_values']):.3f}")
print(f"N sweep results - df range: {np.min(N_results['df_v2_values']):.3f} to {np.max(N_results['df_v2_values']):.3f}")