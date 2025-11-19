
from src.fractal_aggregate import FractalAggregate

# Create an instance with a specific random seed
generator = FractalAggregate(random_seed=123)

# Generate a small aggregate
particles = generator.generate_aggregate(
    N=50,
    bias_factors=(1, 1, 1),
    overlap=0.0,
    inactivation_probability=0.0,
    visualize=True
)

# Calculate and display metrics
shape_factor = generator.calculate_shape_factor(particles)
radius_gyration = generator.calculate_radius_of_gyration(particles)

print(f"\nAggregate Metrics:")
print(f"Shape Factor: {shape_factor:.4f}")
print(f"Radius of Gyration: {radius_gyration:.4f}")