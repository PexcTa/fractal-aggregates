from src.fractal_generator import FractalAggregate

# Create an instance
generator = FractalAggregate(random_seed=123)

# Generate a small aggregate
particles = generator.generate_single_aggregate(N=10, visualize=False)

print("Setup successful!")