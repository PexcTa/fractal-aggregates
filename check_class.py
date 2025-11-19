from src.fractal_aggregate import FractalAggregate
import inspect

# Create an instance
generator = FractalAggregate()

# List all methods and attributes
print("\n=== Class Methods and Attributes ===")
for name, member in inspect.getmembers(generator):
    if not name.startswith('_'):
        print(f"- {name}")

# Check if specific methods exist
print(f"\nHas calculate_shape_factor: {'Yes' if hasattr(generator, 'calculate_shape_factor') else 'No'}")
print(f"Has calculate_radius_of_gyration: {'Yes' if hasattr(generator, 'calculate_radius_of_gyration') else 'No'}")
print(f"Has generate_aggregate: {'Yes' if hasattr(generator, 'generate_aggregate') else 'No'}")