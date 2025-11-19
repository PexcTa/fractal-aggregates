import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from tqdm import tqdm

class FractalAggregate:
    """
    A class to generate and analyze fractal particle aggregates based on the
    Porous Eden Model described in Guesnet et al. (2019).
    
    This class provides a clean, object-oriented interface for:
    - Generating fractal aggregates with tunable properties
    - Calculating morphological metrics (shape factor, radius of gyration, etc.)
    - Visualizing results
    - Performing batch analyses and parameter sweeps
    
    Parameters:
    - random_seed: int, seed for reproducibility (default: 42)
    """
    
    def __init__(self, random_seed=42):
        """
        Initialize the FractalAggregate class with a random seed.
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.aggregates = []  # List to store generated aggregates
    
    def _random_unit_vector(self, bias_factors=(1, 1, 1)):
        """
        Generate a random unit vector in 3D space with optional bias.
        
        Parameters:
        - bias_factors: tuple of three numbers (bx, by, bz) controlling directional preference
        
        Returns:
        - numpy array: 3D unit vector
        """
        # Generate a base random direction using spherical coordinates
        theta = np.random.uniform(0, 2 * np.pi)  # Azimuthal angle
        phi = np.arccos(np.random.uniform(-1, 1))  # Polar angle
        
        # Convert to Cartesian coordinates
        base_x = np.sin(phi) * np.cos(theta)
        base_y = np.sin(phi) * np.sin(theta)
        base_z = np.cos(phi)
        
        # Apply bias factors to create anisotropic distribution
        biased_components = np.array([
            base_x * bias_factors[0],
            base_y * bias_factors[1], 
            base_z * bias_factors[2]
        ])
        
        # Normalize to get a unit vector
        biased_direction = biased_components / np.linalg.norm(biased_components)
        
        return biased_direction
    
    def _is_space_available(self, position, particles, min_distance=2.0):
        """
        Check if there's enough space to place a particle at the given position.
        
        Parameters:
        - position: numpy array of 3D coordinates
        - particles: list of particle dictionaries
        - min_distance: minimum distance between particle centers
        
        Returns:
        - bool: True if space is available, False otherwise
        """
        for particle in particles:
            distance = euclidean(position, particle['position'])
            if distance < min_distance:
                return False  # Overlap detected
        return True  # No overlaps found
    
    def _is_direction_valid(self, source_position, direction, particles, overlap=0.0):
        """
        Check if there's enough space in a given direction from a source particle.
        
        Parameters:
        - source_position: numpy array of 3D coordinates
        - direction: 3D unit vector
        - particles: list of particle dictionaries
        - overlap: fraction of overlap between particle spheres (0.0 to 0.999)
        
        Returns:
        - tuple: (space_available: bool, target_position: numpy array)
        """
        # Effective distance between centers = 2.0 * (1 - overlap)
        effective_distance = 2.0 * (1 - overlap)
        
        # Calculate target position (center of new particle)
        target_position = source_position + direction * effective_distance
        
        # Check if this position is valid (no overlap with existing particles)
        space_available = self._is_space_available(target_position, particles, min_distance=effective_distance)
        
        return space_available, target_position
    
    def generate_aggregate(self, N=1000, bias_factors=(1, 1, 1), overlap=0.0, 
                          inactivation_probability=0.0, max_particles_for_spheres=200, 
                          visualize=True):
        """
        Generate a single fractal aggregate using the Porous Eden Model.
        
        Parameters:
        - N: int, total number of particles to generate (including seed)
        - bias_factors: tuple of three numbers (bx, by, bz) controlling directional preference
        - overlap: float, fraction of overlap between particle spheres (0.0 to 0.999)
        - inactivation_probability: float, probability that an active particle becomes inactive
        - max_particles_for_spheres: int, max particles to render with spheres for visualization
        - visualize: bool, whether to display the 3D visualization
        
        Returns:
        - particles: list of dictionaries containing particle information
        """
        # Validate inputs
        if not isinstance(N, int) or N < 1:
            raise ValueError("N must be a positive integer")
        if len(bias_factors) != 3:
            raise ValueError("bias_factors must be a tuple of exactly three numbers")
        if overlap < 0 or overlap >= 1:
            raise ValueError("overlap must be between 0 (inclusive) and 1 (exclusive)")
        if inactivation_probability < 0 or inactivation_probability > 1:
            raise ValueError("inactivation_probability must be between 0.0 and 1.0")
        
        print(f"Generating {N} particles...")
        
        # Initialize particle storage
        particles = []
        
        # Create seed particle at the origin (0, 0, 0)
        seed_particle = {
            'position': np.array([0, 0, 0]),
            'added_step': 0,
            'inactive': False  # Seed is initially active
        }
        particles.append(seed_particle)
        
        # Main growth loop
        step = 1  # First new particle will be step 1; seed is step 0
        
        # Set random seed for reproducibility with the instance seed
        np.random.seed(self.random_seed)
        
        while len(particles) < N:
            # Select a random ACTIVE particle (not deactivated)
            active_particles = [p for p in particles if not p['inactive']]
            
            # Safety check: cannot proceed if no active particles remain
            if len(active_particles) == 0:
                print(f"Warning: All particles deactivated before reaching {N} particles. Stopping.")
                break
                
            # If only one active particle remains, we MUST keep it active (inactivation forbidden)
            if len(active_particles) == 1:
                selected_particle = active_particles[0]
            else:
                # Choose a random active particle
                selected_particle = np.random.choice(active_particles)
            
            # Choose a random direction with optional bias
            direction = self._random_unit_vector(bias_factors=bias_factors)
            
            # Check if there's enough space in this direction
            space_available, target_position = self._is_direction_valid(
                selected_particle['position'], direction, particles, overlap
            )
            
            if space_available:
                # Add new particle
                new_particle = {
                    'position': target_position,
                    'added_step': step,
                    'inactive': False  # New particles start as active
                }
                particles.append(new_particle)
                
                # Inactivate the chosen particle with probability p (only if more than one active particle exists)
                if len(active_particles) > 1 and np.random.random() < inactivation_probability:
                    selected_particle['inactive'] = True
                
                step += 1
                
                # Progress indicator
                if N > 100 and len(particles) % (N // 10) == 0:
                    active_count = sum(1 for p in particles if not p['inactive'])
                    inactive_count = sum(1 for p in particles if p['inactive'])
                    print(f"Progress: {len(particles)}/{N} particles generated, "
                          f"Active: {active_count}, Inactive: {inactive_count}")
        
        print(f"Generation complete: {len(particles)} particles created")
        
        # Store the aggregate for later use
        self.aggregates.append({
            'particles': particles,
            'parameters': {
                'N': N,
                'bias_factors': bias_factors,
                'overlap': overlap,
                'inactivation_probability': inactivation_probability,
                'random_seed': self.random_seed
            }
        })
        
        # Visualization
        if visualize:
            self.visualize_aggregate(particles, max_particles_for_spheres=max_particles_for_spheres)
        
        print(f"\nFinal aggregate stats:")
        print(f"  - Total particles: {len(particles)}")
        print(f"  - Active particles: {sum(1 for p in particles if not p['inactive'])}")
        print(f"  - Inactive particles: {sum(1 for p in particles if p['inactive'])}")
        print(f"  - Inactivation probability p: {inactivation_probability}")
        print(f"  - Overlap: {overlap} → center-to-center distance = {2.0 * (1 - overlap):.3f}")
        print(f"  - Bias factors: {bias_factors}")
        print(f"  - Random seed: {self.random_seed}")
        
        return particles
    
    def calculate_shape_factor(self, particles):
        """
        Calculate the shape factor of the fractal aggregate based on the inertia tensor.
        According to Guesnet et al. (2019), the shape factor is the square root of the ratio 
        of the largest eigenvalue to the smallest eigenvalue of the inertia matrix.
        
        Parameters:
        - particles: list of dictionaries returned by generate_aggregate()
                     Must contain 'position' key with numpy array of 3D coordinates.
        
        Returns:
        - shape_factor: float, the calculated shape factor (>= 1.0)
        """
        # Extract all particle positions
        positions = np.array([p['position'] for p in particles])
        
        # Calculate the center of mass (centroid) of the aggregate
        centroid = np.mean(positions, axis=0)
        
        # Center the particle positions around the centroid
        centered_positions = positions - centroid
        
        # Calculate the inertia tensor (moment of inertia tensor)
        # For a set of point masses (each assumed to have mass=1), the inertia tensor I is:
        # I_ij = sum_k (r_k^2 * delta_ij - r_{k,i} * r_{k,j})
        # where r_k is the position vector of particle k relative to the centroid,
        # r_{k,i} is the i-th component of r_k, and delta_ij is the Kronecker delta.
        
        # Initialize the inertia tensor (3x3 matrix)
        I = np.zeros((3, 3))
        
        # Compute the inertia tensor
        for pos in centered_positions:
            x, y, z = pos
            r_squared = np.dot(pos, pos)  # r_k^2 = x^2 + y^2 + z^2
            
            # I_xx = sum(y^2 + z^2)
            I[0, 0] += r_squared - x**2
            # I_yy = sum(x^2 + z^2)
            I[1, 1] += r_squared - y**2
            # I_zz = sum(x^2 + y^2)
            I[2, 2] += r_squared - z**2
            
            # I_xy = I_yx = -sum(x*y)
            I[0, 1] -= x * y
            I[1, 0] -= x * y
            # I_xz = I_zx = -sum(x*z)
            I[0, 2] -= x * z
            I[2, 0] -= x * z
            # I_yz = I_zy = -sum(y*z)
            I[1, 2] -= y * z
            I[2, 1] -= y * z
        
        # The inertia tensor is symmetric, so we can use np.linalg.eigh for real symmetric matrices
        eigenvalues, _ = np.linalg.eigh(I)
        
        # Sort eigenvalues in ascending order
        eigenvalues_sorted = np.sort(eigenvalues)
        
        # Extract the smallest and largest eigenvalues
        lambda_min = eigenvalues_sorted[0]
        lambda_max = eigenvalues_sorted[-1]
        
        # Calculate shape factor as sqrt(lambda_max / lambda_min)
        # If lambda_min is zero (all points are collinear or coincident), avoid division by zero
        if lambda_min < 1e-15:
            # In theory, this should be extremely rare for a non-trivial aggregate
            # We'll return a very large number to indicate extreme anisotropy
            shape_factor = np.inf
        else:
            shape_factor = np.sqrt(lambda_max / lambda_min)
        
        return shape_factor
    
    def calculate_radius_of_gyration(self, particles):
        """
        Calculate the radius of gyration (Rg) of the fractal aggregate as defined in
        Guesnet et al. (2019), Physica A 513, Eq. (2).
        
        Rg = sqrt( (1/N) * sum_{i,j>i} (rij)^2 )
        where N is the number of particles and rij is the distance between particles i and j.
        
        Parameters:
        - particles: list of dictionaries returned by generate_aggregate()
                     Must contain 'position' key with numpy array of 3D coordinates.
        
        Returns:
        - Rg: float, the calculated radius of gyration.
        """
        # Extract all particle positions
        positions = np.array([p['position'] for p in particles])
        N = len(positions)
        
        # If there's only one particle, Rg is 0
        if N <= 1:
            return 0.0
        
        # Calculate the sum of squared distances between all unique pairs (i, j) where j > i
        sum_sq_distances = 0.0
        for i in range(N):
            for j in range(i + 1, N):  # j > i ensures each pair is counted once
                rij = euclidean(positions[i], positions[j])
                sum_sq_distances += rij ** 2
        
        # Calculate Rg according to the formula
        Rg = np.sqrt(sum_sq_distances / N)
        
        return Rg
    
    def visualize_aggregate(self, particles, max_particles_for_spheres=200, figsize=(12, 9)):
        """
        Visualize a fractal aggregate in 3D.
        
        Parameters:
        - particles: list of particle dictionaries
        - max_particles_for_spheres: int, max particles to render with spheres
        - figsize: tuple, figure size
        
        Returns:
        - fig, ax: matplotlib figure and axis objects
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        all_positions = np.array([p['position'] for p in particles])
        
        # Compute cubic bounding box (equal x,y,z limits)
        max_extent = np.max(np.abs(all_positions))
        margin = max(5, max_extent * 0.1)
        ax.set_xlim(-max_extent - margin, max_extent + margin)
        ax.set_ylim(-max_extent - margin, max_extent + margin)
        ax.set_zlim(-max_extent - margin, max_extent + margin)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Porous Eden Aggregate ({len(particles)} particles)')
        
        # Plot scatter points for all particles
        active_positions = np.array([p['position'] for p in particles if not p['inactive']])
        inactive_positions = np.array([p['position'] for p in particles if p['inactive']])
        
        if len(active_positions) > 0:
            ax.scatter(active_positions[:, 0], active_positions[:, 1], active_positions[:, 2],
                      c='blue', s=20, alpha=0.8, label='Active Particles')
        
        if len(inactive_positions) > 0:
            ax.scatter(inactive_positions[:, 0], inactive_positions[:, 1], inactive_positions[:, 2],
                      c='gray', s=15, alpha=0.6, label='Inactive Particles')
        
        # Only plot transparent spheres if N <= threshold
        if len(particles) <= max_particles_for_spheres:
            u = np.linspace(0, 2 * np.pi, 15)
            v = np.linspace(0, np.pi, 15)
            
            # Radius of each particle is always 1.0
            for pos in all_positions:
                x = 1.0 * np.outer(np.cos(u), np.sin(v)) + pos[0]
                y = 1.0 * np.outer(np.sin(u), np.sin(v)) + pos[1]
                z = 1.0 * np.outer(np.ones(np.size(u)), np.cos(v)) + pos[2]
                ax.plot_surface(x, y, z, color='blue', alpha=0.1, rstride=1, cstride=1, linewidth=0)
            
            print(f"Rendered {len(particles)} transparent spheres (radius = 1.0)")
        
        # Ensure equal scaling on all axes
        ax.set_box_aspect([1, 1, 1])
        
        # Improve viewing angle
        ax.view_init(elev=20, azim=225)
        
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
        
        return fig, ax