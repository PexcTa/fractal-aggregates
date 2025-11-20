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

print(f"Successfully generated {len(particles)} particles!")