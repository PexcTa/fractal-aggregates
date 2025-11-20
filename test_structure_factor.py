from src.fractal_aggregate import FractalAggregate
import matplotlib.pyplot as plt

# Create an instance with a specific random seed
generator = FractalAggregate(random_seed=123)

# Generate a medium-sized aggregate
particles = generator.generate_aggregate(
    N=200,
    bias_factors=(1, 1, 1),
    overlap=0.0,
    inactivation_probability=0.3,
    visualize=True
)

# Calculate structure factor with visualization
q_array, S_array, Rg, alpha, alpha_err = generator.calculate_structure_factor(
    particles,
    q_min=0.01,
    q_max=10.0,
    n_q=100,
    R_particle=1.0,
    fit_range_factor=2.0,
    visualize=True
)

print(f"\nStructure Factor Results:")
print(f"Radius of Gyration (Rg): {Rg:.4f}")
print(f"Fitted alpha: {alpha:.4f} ± {alpha_err:.4f}")

# Determine fractal dimensions
if alpha <= 3.0:
    df = alpha
    print(f"Mass fractal dimension df = {df:.4f}")
else:
    df = 3.0
    ds = 6.0 - alpha
    print(f"Mass fractal dimension df = 3.0 (compact)")
    print(f"Surface fractal dimension ds = {ds:.4f}")