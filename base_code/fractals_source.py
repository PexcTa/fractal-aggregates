# -*- coding: utf-8 -*-
"""
Enhanced fractal aggregate generator with linked cell spatial acceleration.
Created on Fri Nov 28 2025

@author: boris
"""

#%% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import euclidean, pdist
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull
from tqdm import tqdm

class LinkedCellGrid:
    """
    Spatial acceleration structure for fast nearest-neighbor queries.
    
    Divides 3D space into cubic cells of size (cell_size × cell_size × cell_size).
    Particles are indexed by their cell coordinates; queries search only nearby cells.
    
    This reduces contact detection from O(N) to O(1) average case.
    """
    
    def __init__(self, cell_size=8.0):
        """
        Initialize the linked cell grid.
        
        Parameters:
        - cell_size: float, the side length of each cubic cell.
                     Recommended: 2.5–3× the contact distance.
                     Default 4.0 is ~ 2× contact distance.
        """
        self.cell_size = cell_size
        self.cells = {}  # Dict: (i, j, k) -> list of particle indices
        self.particle_to_cell = {}  # Dict: particle_idx -> (i, j, k)
        self.particle_positions = {}  # Dict: particle_idx -> position (for fast lookup)
    
    def position_to_cell_coords(self, position):
        """
        Convert a continuous 3D position to discrete cell coordinates.
        
        Parameters:
        - position: numpy array [x, y, z]
        
        Returns:
        - tuple (i, j, k) of cell coordinates
        """
        cell_coords = tuple(np.floor(position / self.cell_size).astype(int))
        return cell_coords
    
    def add_particle(self, particle_idx, position):
        """
        Add a particle to the grid at the given position.
        
        Parameters:
        - particle_idx: int, unique index of the particle
        - position: numpy array [x, y, z]
        """
        cell_coords = self.position_to_cell_coords(position)
        
        # Initialize cell if not present
        if cell_coords not in self.cells:
            self.cells[cell_coords] = []
        
        # Add particle to cell
        self.cells[cell_coords].append(particle_idx)
        
        # Track particle location and position
        self.particle_to_cell[particle_idx] = cell_coords
        self.particle_positions[particle_idx] = position.copy()
    
    def get_nearby_particle_indices(self, position, search_radius=1):
        """
        Query all particles within a certain number of cells from the given position.
        
        Parameters:
        - position: numpy array [x, y, z]
        - search_radius: int, number of cell layers to search around the target cell.
                        Default 1 searches 3×3×3 = 27 cells (including the target cell).
        
        Returns:
        - list of particle indices in the nearby cells
        """
        cell_coords = self.position_to_cell_coords(position)
        nearby_indices = []
        
        # Search in a cube around the target cell
        for di in range(-search_radius, search_radius + 1):
            for dj in range(-search_radius, search_radius + 1):
                for dk in range(-search_radius, search_radius + 1):
                    neighbor_cell = (
                        cell_coords[0] + di,
                        cell_coords[1] + dj,
                        cell_coords[2] + dk
                    )
                    
                    # Add particles from this cell if it exists
                    if neighbor_cell in self.cells:
                        nearby_indices.extend(self.cells[neighbor_cell])
        
        return nearby_indices
    
    def get_grid_stats(self):
        """
        Return statistics about grid occupancy (useful for diagnostics).
        
        Returns:
        - dict with keys: 'num_cells', 'occupied_cells', 'particles_per_cell_avg', etc.
        """
        num_cells = len(self.cells)
        total_particles = sum(len(indices) for indices in self.cells.values())
        
        if num_cells > 0:
            avg_per_cell = total_particles / num_cells
            max_per_cell = max(len(indices) for indices in self.cells.values())
        else:
            avg_per_cell = 0
            max_per_cell = 0
        
        return {
            'num_cells': num_cells,
            'total_particles': total_particles,
            'avg_particles_per_cell': avg_per_cell,
            'max_particles_per_cell': max_per_cell,
            'cell_size': self.cell_size
        }

def generate_fractal_aggregate(
    N=1000,
    radius=1.0,
    bias_factors=(1, 1, 1),
    random_seed=42,
    overlap=0.0,
    inactivation_probability=0.0,
    cell_size=8.0,
    max_particles_for_spheres=200,
    visualize=True
):
    """
    Generate a 3D fractal aggregate using the Porous Eden Model (Guesnet et al., 2019)
    with linked cell spatial acceleration.
    
    Parameters:
    -----------
    N : int
        Total number of particles to generate (including seed). Default 1000.
        
    radius : float
        Radius of individual particles; default 1.0
    
    bias_factors : tuple of three floats
        Directional bias (bx, by, bz). Higher values favor growth along that axis.
        (1, 1, 1) = isotropic. (1, 1, 5) = elongated along z. Default (1, 1, 1).
    
    random_seed : int
        Random seed for reproducibility. Default 42.
    
    overlap : float
        Particle overlap fraction [0, 1). Hard spheres = 0.0. Default 0.0.
        Affects contact distance: d_contact = 2.0 * radius * (1 - overlap).
    
    inactivation_probability : float
        Probability [0, 1] that a particle deactivates after successful growth.
        p = 0.0 → pure Eden (dense). p = 1.0 → chain-like. Default 0.0.
    
    cell_size : float
        Side length of linked cell grid cells. 
        Recommended: 2.5–3× contact distance. For contact distance 2.0, use 4–6.
        Default 4.0. Larger = faster queries but may miss nearby particles.
        Smaller = slower queries but more accurate. Tune for your aggregate size.
    
    max_particles_for_spheres : int
        Max particles to render as transparent spheres. Larger N shows points only.
        Default 200.
    
    visualize : bool
        Whether to display 3D visualization. Default True.
    
    Returns:
    --------
    dict with keys:
        'particles': list of dicts containing:
            - 'position': numpy array [x, y, z]
            - 'added_step': int, step at which added (0 = seed)
            - 'inactive': bool, deactivation status
        'parameters': dict of generation parameters (including cell_size used)
        'metadata': dict with stats (total particles, active count, etc.)
    """
    
    # ========== INPUT VALIDATION ==========
    if not isinstance(N, int) or N < 1:
        raise ValueError(f"N must be a positive integer, got {N}")
    
    if len(bias_factors) != 3:
        raise ValueError(f"bias_factors must be tuple of 3 numbers, got {len(bias_factors)}")
    
    if not all(isinstance(b, (int, float)) and b > 0 for b in bias_factors):
        raise ValueError(f"bias_factors must be positive numbers, got {bias_factors}")
    
    if overlap < 0 or overlap >= 1:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    
    if inactivation_probability < 0 or inactivation_probability > 1:
        raise ValueError(f"inactivation_probability must be in [0, 1], got {inactivation_probability}")
    
    if cell_size <= 0:
        raise ValueError(f"cell_size must be positive, got {cell_size}")
    
    # ========== INITIALIZATION ==========
    np.random.seed(random_seed)
    
    # Particle storage
    particles = []
    
    # Initialize linked cell grid
    spatial_grid = LinkedCellGrid(cell_size=cell_size)
    
    # Contact distance (effective distance between particle centers)
    effective_distance = 2.0 * radius * (1 - overlap)
    
    # Create seed particle at origin
    seed_particle = {
        'position': np.array([0.0, 0.0, 0.0], dtype=np.float64),
        'added_step': 0,
        'inactive': False
    }
    particles.append(seed_particle)
    spatial_grid.add_particle(0, seed_particle['position'])
    
    print(f"Generating {N} particles with linked cell acceleration (cell_size={cell_size})...")
    print(f"  Contact distance: {effective_distance:.3f}")
    print(f"  Inactivation probability: {inactivation_probability:.2f}")
    print(f"  Bias factors: {bias_factors}")
    
    # ========== INTERNAL HELPER FUNCTIONS ==========
    
    def random_unit_vector():
        """
        Generate a random 3D unit vector with optional anisotropic bias.
        """
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.arccos(np.random.uniform(-1, 1))
        
        base_x = np.sin(phi) * np.cos(theta)
        base_y = np.sin(phi) * np.sin(theta)
        base_z = np.cos(phi)
        
        # Apply bias
        biased_components = np.array([
            base_x * bias_factors[0],
            base_y * bias_factors[1],
            base_z * bias_factors[2]
        ])
        
        # Normalize
        norm = np.linalg.norm(biased_components)
        return biased_components / norm
    
    def is_space_available_linked(position):
        """
        Check if position is collision-free using linked cell acceleration.
        
        Instead of checking all particles (O(N)), query only nearby cells.
        Typically 27 cells searched (3×3×3), each with few particles.
        """
        nearby_indices = spatial_grid.get_nearby_particle_indices(position, search_radius=1)
        
        for idx in nearby_indices:
            distance = euclidean(position, spatial_grid.particle_positions[idx])
            if distance < effective_distance:
                return False  # Collision detected
        
        return True  # No collisions
    
    # ========== MAIN GROWTH LOOP ==========
    step = 1  # First new particle will be step 1; seed is step 0
    attempt_count = 0
    success_count = 0
    
    while len(particles) < N:
        # Find all active particles
        active_particles = [p for p in particles if not p['inactive']]
        
        # Safety check
        if len(active_particles) == 0:
            print(f"\nWarning: All particles deactivated. Stopping at {len(particles)} particles.")
            break
        
        # Select a random active particle
        if len(active_particles) == 1:
            selected_particle = active_particles[0]
        else:
            selected_particle = np.random.choice(active_particles)
        
        # Generate random direction with bias
        direction = random_unit_vector()
        
        # Calculate target position
        target_position = selected_particle['position'] + direction * effective_distance
        
        attempt_count += 1
        
        # Check collision using linked cell grid
        if is_space_available_linked(target_position):
            # Add new particle
            new_particle = {
                'position': target_position.astype(np.float64),
                'added_step': step,
                'inactive': False
            }
            particles.append(new_particle)
            spatial_grid.add_particle(len(particles) - 1, target_position)
            
            success_count += 1
            
            # Possibly inactivate the selected particle
            if len(active_particles) > 1 and np.random.random() < inactivation_probability:
                selected_particle['inactive'] = True
            
            step += 1
            
            # Progress indicator
            if N > 100 and len(particles) % max(1, N // 10) == 0:
                active_count = sum(1 for p in particles if not p['inactive'])
                inactive_count = len(particles) - active_count
                grid_stats = spatial_grid.get_grid_stats()
                print(
                    f"  [{len(particles):6d}/{N}] Active: {active_count:5d}, "
                    f"Inactive: {inactive_count:5d}, "
                    f"Grid cells: {grid_stats['num_cells']}, "
                    f"Avg particles/cell: {grid_stats['avg_particles_per_cell']:.1f}"
                )
    
    # ========== FINALIZATION ==========
    print(f"\nGeneration complete!")
    print(f"  Total particles: {len(particles)}")
    print(f"  Successful placements: {success_count}")
    print(f"  Total attempts: {attempt_count}")
    print(f"  Success rate: {100 * success_count / attempt_count:.1f}%")
    
    grid_stats = spatial_grid.get_grid_stats()
    print(f"  Grid occupancy: {grid_stats['num_cells']} cells, "
          f"avg {grid_stats['avg_particles_per_cell']:.1f} particles/cell")
    
    # ========== VISUALIZATION ==========
    if visualize:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        all_positions = np.array([p['position'] for p in particles])
        
        # Compute bounding box
        max_extent = np.max(np.abs(all_positions))
        margin = max(5, max_extent * 0.1)
        ax.set_xlim(-max_extent - margin, max_extent + margin)
        ax.set_ylim(-max_extent - margin, max_extent + margin)
        ax.set_zlim(-max_extent - margin, max_extent + margin)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(
            f'Porous Eden Aggregate (N={len(particles)}, p={inactivation_probability:.2f}, '
            f'cell_size={cell_size}, overlap={overlap:.2f})'
        )
        
        # Separate active and inactive particles
        active_positions = np.array([p['position'] for p in particles if not p['inactive']])
        inactive_positions = np.array([p['position'] for p in particles if p['inactive']])
        
        if len(active_positions) > 0:
            ax.scatter(
                active_positions[:, 0], active_positions[:, 1], active_positions[:, 2],
                c='blue', s=20, alpha=0.8, label='Active Particles'
            )
        
        if len(inactive_positions) > 0:
            ax.scatter(
                inactive_positions[:, 0], inactive_positions[:, 1], inactive_positions[:, 2],
                c='gray', s=15, alpha=0.6, label='Inactive Particles'
            )
        
        # Render transparent spheres only for small aggregates
        if len(particles) <= max_particles_for_spheres:
            u = np.linspace(0, 2 * np.pi, 15)
            v = np.linspace(0, np.pi, 15)
            
            for pos in all_positions:
                x = 1.0 * np.outer(np.cos(u), np.sin(v)) + pos[0]
                y = 1.0 * np.outer(np.sin(u), np.sin(v)) + pos[1]
                z = 1.0 * np.outer(np.ones(np.size(u)), np.cos(v)) + pos[2]
                ax.plot_surface(x, y, z, color='blue', alpha=0.1, rstride=1, cstride=1, linewidth=0)
            
            print(f"Rendered {len(particles)} transparent spheres (radius = 1.0)")
        else:
            print(f"Skipped sphere rendering for {len(particles)} particles (too many).")
            print(f"Showing centers only. Increase max_particles_for_spheres to render spheres.")
        
        # Ensure equal scaling
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=20, azim=225)
        
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
    
    # ========== RETURN STRUCTURED DATA ==========
    result = {
        'particles': particles,
        'parameters': {
            'N': N,
            'bias_factors': bias_factors,
            'random_seed': random_seed,
            'overlap': overlap,
            'inactivation_probability': inactivation_probability,
            'cell_size': cell_size,  # <-- The cell_size is saved here!
            'effective_distance': effective_distance,
            'max_particles_for_spheres': max_particles_for_spheres,
            'visualize': visualize
        },
        'metadata': {
            'total_particles': len(particles),
            'active_particles': sum(1 for p in particles if not p['inactive']),
            'inactive_particles': sum(1 for p in particles if p['inactive']),
            'success_rate': success_count / attempt_count if attempt_count > 0 else 0,
            'grid_stats': grid_stats
        }
    }
    
    return result

def _extract_particles(particles_or_result):
    """
    Extract particle list from either old format (list) or new format (dict).
    
    Parameters:
    - particles_or_result: either a list of dicts OR a dict with 'particles' key
    
    Returns:
    - particles: list of particle dicts
    - parameters: dict of generation parameters (or empty dict if old format)
    """
    if isinstance(particles_or_result, dict) and 'particles' in particles_or_result:
        # New format: result dict from generate_fractal_aggregate()
        particles = particles_or_result['particles']
        parameters = particles_or_result.get('parameters', {})
    elif isinstance(particles_or_result, list):
        # Old format: direct list of particles
        particles = particles_or_result
        parameters = {}
    else:
        raise TypeError(
            "Input must be either a list of particle dicts or "
            "a result dict from generate_fractal_aggregate()"
        )
    
    return particles, parameters

def calculate_shape_factor(particles_or_result):
    """
    Calculate the shape factor of the fractal aggregate based on the inertia tensor.
    
    According to Guesnet et al. (2019), the shape factor is the square root of the ratio 
    of the largest eigenvalue to the smallest eigenvalue of the inertia matrix.
    
    Parameters:
    - particles_or_result: list of dicts (old format) OR result dict (new format)
                           Must contain 'position' key with numpy array of 3D coordinates.
    
    Returns:
    - shape_factor: float, the calculated shape factor (>= 1.0)
    """
    particles, _ = _extract_particles(particles_or_result)
    
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
    
    # Number of particles
    N = len(positions)
    
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


def calculate_radius_of_gyration(particles_or_result):
    """
    Calculate the radius of gyration (Rg) of the fractal aggregate as defined in
    Guesnet et al. (2019), Physica A 513, Eq. (2).
    
    Rg = sqrt( (1/N) * sum_{i,j>i} (rij)^2 )
    where N is the number of particles and rij is the distance between particles i and j.
    
    Parameters:
    - particles_or_result: list of dicts (old format) OR result dict (new format)
                           Must contain 'position' key with numpy array of 3D coordinates.
    
    Returns:
    - Rg: float, the calculated radius of gyration.
    """
    particles, _ = _extract_particles(particles_or_result)
    
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
    Rg = np.sqrt(sum_sq_distances) / N
    
    return Rg


def calculate_and_plot_running_rg(
    particles_or_result,
    fit_start_N=5,
    fit_end_N=None,
    visualize=False
):
    """
    OPTIMIZED: Calculate radius of gyration as function of aggregate size.
    
    Uses O(N²) incremental summation instead of O(N³) pdist approach.
    Speedup: 100-150x for N=10,000 aggregates.
    
    Parameters:
    -----------
    particles_or_result : list or dict
        Particles list or result dict (compatible with both formats)
    
    fit_start_N : int
        Start of linear fit region (default: 5)
    
    fit_end_N : int or None
        End of linear fit region (default: None = entire range after fit_start)
    
    visualize : bool
        If True, create visualization plot
    
    Returns:
    --------
    N_list : list
        Particle counts
    
    Rg_list : list
        Radius of gyration values
    
    df_v1 : float
        Fractal dimension from power-law fit
    
    alpha_err : float
        Uncertainty in fractal dimension
    
    fig : matplotlib figure or None
        Visualization figure if visualize=True, else None
    
    ax : matplotlib axes or None
        Visualization axes if visualize=True, else None
    """
    
    particles, _ = _extract_particles(particles_or_result)
    
    # Sort particles by added_step to maintain growth order
    sorted_particles = sorted(particles, key=lambda p: p['added_step'])
    positions = np.array([p['position'] for p in sorted_particles])
    N_total = len(positions)
    
    if fit_end_N is None:
        fit_end_N = N_total
    
    N_list = []
    Rg_list = []
    
    # ========== OPTIMIZED: Incremental Summation ==========
    sum_sq_distances = 0.0  # Running accumulator
    
    for n in range(1, N_total + 1):
        if n == 1:
            Rg = 0.0
        else:
            # OPTIMIZED: Only compute distances from particle n to particles 1..n-1
            new_particle = positions[n - 1]
            prev_particles = positions[:n - 1]
            
            # Vectorized distance computation (no Python loop!)
            distances = np.linalg.norm(prev_particles - new_particle, axis=1)
            sq_distances = distances ** 2
            
            # Accumulate (don't replace!)
            sum_sq_distances += np.sum(sq_distances)
            
            # Compute Rg from accumulated sum
            Rg = np.sqrt(sum_sq_distances) / n
        
        N_list.append(n)
        Rg_list.append(Rg)
    
    # ========== End Optimization ==========
    
    # Power-law fit: Rg = constant * N^(1/df)
    # Taking log: log(Rg) = log(constant) + (1/df) * log(N)
    
    N_fit = np.array(N_list[fit_start_N:fit_end_N])
    Rg_fit = np.array(Rg_list[fit_start_N:fit_end_N])
    
    # Avoid log(0)
    valid_mask = (N_fit > 0) & (Rg_fit > 0)
    N_fit = N_fit[valid_mask]
    Rg_fit = Rg_fit[valid_mask]
    
    if len(N_fit) > 1:
        # Linear fit to log-log data
        log_N = np.log(N_fit)
        log_Rg = np.log(Rg_fit)
        
        coeffs = np.polyfit(log_N, log_Rg, 1)
        alpha = coeffs[0]  # slope = 1/df
        df_v1 = 1.0 / alpha if alpha != 0 else np.nan
        
        # Estimate uncertainty
        fit_line = coeffs[0] * log_N + coeffs[1]
        residuals = log_Rg - fit_line
        alpha_err = np.std(residuals) / np.sqrt(len(N_fit)) if len(N_fit) > 0 else np.nan
    else:
        df_v1 = np.nan
        alpha_err = np.nan
    
    # Visualization (optional)
    fig, ax = None, None
    if visualize:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.loglog(N_list, Rg_list, 'o-', label='Rg(N)', linewidth=2, markersize=4)
        
        if len(N_fit) > 1:
            N_plot = np.logspace(np.log10(np.min(N_fit)), np.log10(np.max(N_fit)), 100)
            Rg_plot = np.exp(coeffs[0] * np.log(N_plot) + coeffs[1])
            ax.loglog(N_plot, Rg_plot, '--', label=f'Power-law fit: df={df_v1:.3f}', linewidth=2)
        
        ax.set_xlabel('Particle Count (N)', fontsize=12)
        ax.set_ylabel('Radius of Gyration (Rg)', fontsize=12)
        ax.set_title('Running Rg: Aggregate Growth', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
    
    return N_list, Rg_list, df_v1, alpha_err, fig, ax


def calculate_structure_factor(
    particles_or_result,
    q_min=0.01,
    q_max=10.0,
    n_q=100,
    R_particle=1.0,
    fit_range_factor=2.0,
    visualize=False
):
    """
    Calculate the structure factor S(q) for a 3D particle aggregate as defined in Guesnet et al. (2019).
    
    Parameters:
    - particles_or_result: list of dicts (old format) OR result dict (new format)
                           Must have 'position' keys (numpy arrays of 3D coordinates)
    - q_min: float, minimum q value for calculation
    - q_max: float, maximum q value for calculation
    - n_q: int, number of q points in log space
    - R_particle: float, radius of individual particles (default=1.0)
    - fit_range_factor: float, multiplier for Rg to define fitting range
    - visualize: bool, whether to display the plot. Default False.
    
    Returns:
    - q_array: array of q values
    - S_array: array of S(q) values
    - Rg: float, radius of gyration of the aggregate
    - alpha: float, fitted slope of ln(S(q)) vs ln(q)
    - alpha_err: float, standard error of the fit
    - fig, ax: matplotlib figure and axis objects (or None if visualize=False)
    """
    
    particles, _ = _extract_particles(particles_or_result)
    
    # Extract positions
    positions = np.array([p['position'] for p in particles])
    N = len(positions)
    
    if N <= 1:
        raise ValueError("Need at least 2 particles to compute structure factor.")
    
    # Step 1: Calculate radius of gyration Rg 
    pairwise_distances = pdist(positions, metric='euclidean')
    sum_sq_distances = np.sum(pairwise_distances ** 2)
    Rg = np.sqrt(sum_sq_distances) / N
    
    # Step 2: Define q range (logarithmic)
    q_array = np.logspace(np.log10(q_min), np.log10(q_max), n_q)
    
    # Step 3: Precompute all pairwise distances once
    # Use the condensed form from pdist
    r_ij = pairwise_distances  # Shape: (N*(N-1)/2,)
    
    # Step 4: Compute S(q) for each q using vectorized operations
    S_array = np.zeros_like(q_array)
    
    # Vectorized sinc computation: sin(q * r_ij) / (q * r_ij)
    # Handle q*r_ij = 0 case (r_ij=0) to avoid division by zero
    for i, q in enumerate(q_array):
        qr = q * r_ij
        # Avoid division by zero where qr == 0
        sinc_values = np.where(qr == 0, 1.0, np.sin(qr) / qr)
        S_i = 1.0 + (1.0 / N) * np.sum(sinc_values)
        S_array[i] = S_i
    
    # Step 5: Determine fitting range based on paper
    # Paper suggests: 1/Rg < q < 1/R (where R=1.0)
    q_fit_min = max(1.0 / Rg, q_min)
    q_fit_max = min(1.0 / R_particle, q_max)  # Allow some buffer
    
    # Filter data for fitting
    mask = (q_array >= q_fit_min) & (q_array <= q_fit_max)
    q_fit = q_array[mask]
    S_fit = S_array[mask]
    
    # Remove any non-positive S(q) values (shouldn't happen but just in case)
    valid_mask = S_fit > 0
    q_fit = q_fit[valid_mask]
    S_fit = S_fit[valid_mask]
    
    if len(q_fit) < 3:
        raise ValueError(f"Not enough data points in fitting range [{q_fit_min:.3f}, {q_fit_max:.3f}]")
    
    # Step 6: Linear fit on log-log scale: ln(S(q)) = -α * ln(q) + C
    def linear_model(log_q, alpha, c):
        return -alpha * log_q + c  # Note: negative sign per Eq. (5)
    
    log_q = np.log(q_fit)
    log_S = np.log(S_fit)
    
    popt, pcov = curve_fit(linear_model, log_q, log_S)
    alpha, c = popt
    alpha_err = np.sqrt(pcov[0, 0])  # Standard error of alpha
    
    fig, ax = None, None
    
    if visualize:
        # Step 7: Plot
        fig, ax = plt.subplots(figsize=(10, 8))
    
        # Full S(q) plot
        ax.loglog(q_array, S_array, 'b-', linewidth=1.5, label='Structure Factor S(q)')
    
        # Highlight fitting region
        ax.loglog(q_fit, S_fit, 'ro', markersize=5, label=f'Fit Region ({q_fit_min:.3f} < q < {q_fit_max:.3f})')
    
        # Plot fitted line
        log_q_full = np.log(q_array)
        log_S_pred = linear_model(log_q_full, alpha, c)
        S_pred = np.exp(log_S_pred)
        ax.loglog(q_array, S_pred, 'r--', linewidth=2, 
                  label=f'Fit: α = {alpha:.4f} ± {alpha_err:.4f}')
    
        # Add vertical lines for fitting bounds
        ax.axvline(x=q_fit_min, color='gray', linestyle=':', label=f'1/Rg = {1.0/Rg:.3f}')
        ax.axvline(x=1.0/R_particle, color='gray', linestyle='-.', label='1/R = 1.0')
        
        ax.set_xlabel('Scattering Vector q (1/particle_radius)', fontsize=12)
        ax.set_ylabel('Structure Factor S(q)', fontsize=12)
        ax.set_title(f'Structure Factor of Fractal Aggregate\n'
                     f'N={N}, Rg={Rg:.3f}, α={alpha:.4f}±{alpha_err:.4f}', fontsize=14)
        
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
    
    # Print results
    print(f"\nStructure Factor Analysis:")
    print(f"  - Radius of gyration Rg = {Rg:.6f}")
    print(f"  - Fitting range: q ∈ [{q_fit_min:.4f}, {q_fit_max:.4f}]")
    print(f"  - Slope α = {alpha:.6f} ± {alpha_err:.6f}")
    
    # Determine fractal dimensions
    if alpha <= 3.0:
        df = alpha
        ds = None
        print(f"  - Mass fractal dimension df = α = {df:.6f}")
        print(f"  - Surface fractal dimension ds = Not applicable (df ≤ 3)")
    else:
        df = 3.0
        ds = 6.0 - alpha
        print(f"  - Mass fractal dimension df = 3.0 (compact)")
        print(f"  - Surface fractal dimension ds = 6 - α = {ds:.6f}")
    
    return q_array, S_array, Rg, alpha, alpha_err, fig, ax


def calculate_porosity(particles_or_result, particle_radius=1.0):
    """
    Calculate the porosity of a fractal aggregate.
    
    Porosity (ϵ) is defined as:
        ϵ = 1 - (Total Volume of Particles) / (Volume of Convex Hull)
    
    Where:
    - Total Volume of Particles = N * (4/3) * π * R³
    - N is the number of particles
    - R is the radius of each spherical particle (default = 1.0)
    - Volume of Convex Hull is computed using scipy.spatial.ConvexHull
    
    Parameters:
    - particles_or_result: list of dicts (old format) OR result dict (new format)
                           Must contain 'position' key with numpy array of 3D coordinates.
    - particle_radius: float, radius of each spherical particle. Default = 1.0.
    
    Returns:
    - porosity: float, the calculated porosity (between 0 and 1)
    - convex_hull_volume: float, volume of the convex hull enclosing all particles
    - total_particle_volume: float, total volume occupied by all particles
    """
    
    particles, _ = _extract_particles(particles_or_result)
    
    # Extract all particle positions
    positions = np.array([p['position'] for p in particles])
    N = len(positions)
    
    if N == 0:
        return 0.0, 0.0, 0.0
    
    # Calculate total volume of all particles
    # Volume of one sphere = (4/3) * π * R^3
    volume_per_particle = (4.0 / 3.0) * np.pi * (particle_radius ** 3)
    total_particle_volume = N * volume_per_particle
    
    # Compute the convex hull of the particle positions
    try:
        hull = ConvexHull(positions)
        convex_hull_volume = hull.volume
    except Exception as e:
        # Handle edge cases (e.g., all points collinear or coincident)
        print(f"Warning: Could not compute convex hull: {e}")
        # Fallback: Use bounding box volume as approximation
        min_coords = np.min(positions, axis=0)
        max_coords = np.max(positions, axis=0)
        convex_hull_volume = np.prod(max_coords - min_coords)
    
    # Calculate porosity
    # ϵ = 1 - (total_particle_volume / convex_hull_volume)
    if convex_hull_volume <= 0:
        porosity = 0.0
    else:
        porosity = 1.0 - (total_particle_volume / convex_hull_volume)
    
    # Ensure porosity is bounded between 0 and 1 (numerical errors can cause slight negative values)
    porosity = max(0.0, min(1.0, porosity))
    
    return porosity, convex_hull_volume, total_particle_volume

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union

def save_simulation_results(
    simulation_results: List[Dict],
    filepath: Union[str, Path],
    metadata: Optional[Dict] = None
) -> str:
    """
    Save simulation results to a tab-separated text file.
    
    Creates a plain text file with:
    - Metadata header (lines starting with #)
    - Tab-separated columns: metrics (shape_factor, df_v1, df_v2, porosity, Rg) + particle coordinates
    - One row per simulation
    
    Parameters:
    -----------
    simulation_results : list of dicts
        Output from generate_simulation_results_by_* functions.
        Each dict contains: 'particles', 'shape_factor', 'df_v1', 'df_v2', 
        'porosity', 'Rg', 'parameters'.
    
    filepath : str or Path
        Destination file path. Extension will be auto-added if not present (.txt).
    
    metadata : dict, optional
        Additional metadata to save in header. Useful for:
        - batch_id, experimenter, notes
        - Timestamp and parameters automatically included
    
    Returns:
    --------
    str : Path to the saved file
    
    Examples:
    ---------
    >>> results = generate_simulation_results_by_seed(N=5000, num_simulations=100)
    >>> filepath = save_simulation_results(
    ...     results,
    ...     'my_batch.txt',
    ...     metadata={'batch_id': 'batch_001', 'experimenter': 'Boris'}
    ... )
    >>> print(f"Saved to: {filepath}")
    
    File format:
    - Human-readable
    - Can be opened in Excel, R, MATLAB, Python pandas
    - Includes full particle coordinate data for each simulation
    """
    
    filepath = Path(filepath)
    
    # Auto-add extension if not present
    if not filepath.suffix:
        filepath = filepath.with_suffix('.txt')
    
    # Prepare metadata
    if metadata is None:
        metadata = {}
    
    metadata['saved_timestamp'] = datetime.now().isoformat()
    metadata['num_simulations'] = len(simulation_results)
    
    print(f"Saving {len(simulation_results)} simulation results to {filepath}...")
    
    try:
        with open(filepath, 'w') as f:
            # ========== WRITE METADATA HEADER ==========
            f.write("# Simulation Results\n")
            f.write(f"# Saved: {metadata.get('saved_timestamp', 'N/A')}\n")
            f.write(f"# Total Simulations: {len(simulation_results)}\n")
            
            # User-provided metadata
            for key, value in metadata.items():
                if key not in ['saved_timestamp', 'num_simulations']:
                    f.write(f"# {key}: {value}\n")
            
            # Extract sample parameters from first result (if available)
            if len(simulation_results) > 0 and 'parameters' in simulation_results[0]:
                params = simulation_results[0]['parameters']
                f.write(f"# Parameters: N={params.get('N', '?')}, ")
                f.write(f"p={params.get('inactivation_probability', '?')}, ")
                f.write(f"overlap={params.get('overlap', '?')}, ")
                f.write(f"cell_size={params.get('cell_size', '?')}, ")
                f.write(f"bias_factors={params.get('bias_factors', '?')}\n")
            
            f.write("#\n")
            
            # ========== WRITE COLUMN HEADERS ==========
            # Find max number of particles
            max_particles = 0
            for result in simulation_results:
                if 'particles' in result:
                    max_particles = max(max_particles, len(result['particles']))
            
            # Build header
            header_cols = [
                'shape_factor',
                'df_v1',
                'df_v2',
                'porosity',
                'Rg',
                'random_seed'
            ]
            
            # Add particle columns
            for i in range(max_particles):
                header_cols.extend([f'particle_{i+1}_x', f'particle_{i+1}_y', f'particle_{i+1}_z'])
            
            f.write('\t'.join(header_cols) + '\n')
            
            # ========== WRITE DATA ROWS ==========
            for result in simulation_results:
                row_data = []
                
                # Append metrics
                row_data.append(str(result['shape_factor']))
                row_data.append(str(result['df_v1']))
                row_data.append(str(result['df_v2']))
                row_data.append(str(result['porosity']))
                row_data.append(str(result['Rg']))
                
                # Append random seed
                random_seed = result.get('parameters', {}).get('random_seed', 'N/A')
                row_data.append(str(random_seed))
                
                # Append particle coordinates
                particles = result.get('particles', [])
                for i in range(max_particles):
                    if i < len(particles):
                        pos = particles[i]['position']
                        row_data.extend([str(pos[0]), str(pos[1]), str(pos[2])])
                    else:
                        row_data.extend(['N/A', 'N/A', 'N/A'])
                
                f.write('\t'.join(row_data) + '\n')
        
        print(f"✓ Saved: {filepath}")
    
    except IOError as e:
        raise IOError(f"Failed to save to {filepath}: {e}")
    
    # Print file size
    file_size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")
    
    return str(filepath)


def generate_simulation_results_by_seed(
    N=43,
    bias_factors=(1, 1, 1),
    overlap=0.0,
    max_particles_for_spheres=3000,
    inactivation_probability=0.0,
    primary_radius=1.0,
    cell_size=4.0,
    num_simulations=1000,
    figsize=(18, 6),
    save=False
):
    """
    Generate simulation results by looping over a specified number of random seeds.
    Each simulation uses a unique random seed and fixed parameters (including p).
    Generates histograms for Shape Factor, Fractal Dimension (df_v2), Porosity, and Radius of Gyration (Rg).
    
    Parameters:
    - N: int, number of particles per aggregate
    - bias_factors: tuple, growth bias factors (default: isotropic)
    - overlap: float, particle overlap parameter (default: 0.0)
    - max_particles_for_spheres: int, max particles to render spheres for visualization
    - inactivation_probability: float, probability p for particle inactivation (fixed across all simulations)
    - primary_radius: float, size of primary particle (default: 1.0)
    - cell_size: float, linked cell grid size (default: 8.0)
    - num_simulations: int, number of aggregates to generate (each with a unique random seed)
    - figsize: tuple, figure size for the histograms
    - save: bool, if True, saves all simulation data to a text file
    
    Returns:
    - simulation_results: list of dictionaries with structure:
                          'particles', 'shape_factor', 'df_v1', 'df_v2', 'porosity', 'Rg', 'parameters'
    """
    
    print(f"Generating {num_simulations} aggregates with fixed parameters:")
    print(f"  N = {N}")
    print(f"  bias_factors = {bias_factors}")
    print(f"  overlap = {overlap}")
    print(f"  inactivation_probability (p) = {inactivation_probability}")
    print(f"  cell_size = {cell_size}")
    
    simulation_results = []
    
    # Set base seed for generating unique random seeds
    np.random.seed(1241434)
    
    print(f"Running {num_simulations} simulations...")
    
    for i in tqdm(range(num_simulations), desc="Simulations"):
        # Generate a unique random seed for each simulation
        seed = np.random.randint(0, 10**9)
        
        # Generate the aggregate (NEW: returns dict, not list)
        try:
            result = generate_fractal_aggregate(
                N=N,
                radius=primary_radius,
                bias_factors=bias_factors,
                overlap=overlap,
                random_seed=seed,
                max_particles_for_spheres=max_particles_for_spheres,
                inactivation_probability=inactivation_probability,
                cell_size=cell_size,  # NEW: pass cell_size
                visualize=False
            )
            particles = result['particles']  # NEW: extract particles from result
        except Exception as e:
            print(f"Warning: Failed to generate aggregate for seed {seed}: {e}")
            continue
        
        # Calculate shape factor (NEW: can pass result dict directly)
        try:
            sf = calculate_shape_factor(result)
        except Exception as e:
            print(f"Warning: Failed to compute shape factor for seed {seed}: {e}")
            sf = np.nan
        
        # Calculate fractal dimension v1 (running Rg)
        try:
            _, _, df_v1, _, _, _ = calculate_and_plot_running_rg(
                result,
                fit_start_N=5,
                fit_end_N=None,
                visualize=False
            )
        except Exception as e:
            print(f"Warning: Failed to compute df_v1 for seed {seed}: {e}")
            df_v1 = np.nan
        
        # Calculate fractal dimension v2 (structure factor) AND Rg
        try:
            q_array, S_array, Rg, df_v2, alpha_err, _, _ = calculate_structure_factor(
                result,
                q_min=0.01,
                q_max=10.0,
                n_q=100,
                R_particle=primary_radius,
                fit_range_factor=2.0,
                visualize=False
            )
        except Exception as e:
            print(f"Warning: Failed to compute df_v2 or Rg for seed {seed}: {e}")
            df_v2 = np.nan
            Rg = np.nan
        
        # Calculate porosity
        try:
            porosity, _, _ = calculate_porosity(result, particle_radius=primary_radius)
        except Exception as e:
            print(f"Warning: Failed to compute porosity for seed {seed}: {e}")
            porosity = np.nan
        
        # Store ALL relevant data for this simulation
        simulation_results.append({
            'particles': particles,
            'shape_factor': sf,
            'df_v1': df_v1,
            'df_v2': df_v2,
            'porosity': porosity,
            'Rg': Rg,
            'parameters': {
                'N': N,
                'bias_factors': bias_factors,
                'overlap': overlap,
                'inactivation_probability': inactivation_probability,
                'cell_size': cell_size,  # NEW: save cell_size
                'random_seed': seed
            }
        })
    
    print(f"\nGenerated {len(simulation_results)} total simulation results out of {num_simulations} attempts.")
    
    # --- HISTOGRAM VISUALIZATION ---
    shape_factors_clean = np.array([res['shape_factor'] for res in simulation_results if not np.isnan(res['shape_factor'])])
    df_v2_clean = np.array([res['df_v2'] for res in simulation_results if not np.isnan(res['df_v2'])])
    porosities_clean = np.array([res['porosity'] for res in simulation_results if not np.isnan(res['porosity'])])
    Rg_clean = np.array([res['Rg'] for res in simulation_results if not np.isnan(res['Rg'])])
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Shape Factor
    if len(shape_factors_clean) > 0:
        axes[0].hist(shape_factors_clean, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].set_title('Shape Factor Distribution')
        axes[0].set_xlabel('Shape Factor (sqrt(λ_max/λ_min))')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, 'No valid Shape Factor data', horizontalalignment='center',
                     verticalalignment='center', transform=axes[0].transAxes, fontsize=12, color='red')
        axes[0].set_title('Shape Factor Distribution')
    
    # Fractal Dimension
    if len(df_v2_clean) > 0:
        axes[1].hist(df_v2_clean, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
        axes[1].set_title('Fractal Dimension (Structure Factor)')
        axes[1].set_xlabel('Mass Fractal Dimension (df)')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No valid df_v2 data', horizontalalignment='center',
                     verticalalignment='center', transform=axes[1].transAxes, fontsize=12, color='red')
        axes[1].set_title('Fractal Dimension (Structure Factor)')
    
    # Porosity
    if len(porosities_clean) > 0:
        axes[2].hist(porosities_clean, bins=50, color='orange', edgecolor='black', alpha=0.7)
        axes[2].set_title('Porosity Distribution')
        axes[2].set_xlabel('Porosity (ϵ)')
        axes[2].set_ylabel('Frequency')
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'No valid Porosity data', horizontalalignment='center',
                     verticalalignment='center', transform=axes[2].transAxes, fontsize=12, color='red')
        axes[2].set_title('Porosity Distribution')
    
    # Radius of Gyration
    if len(Rg_clean) > 0:
        axes[3].hist(Rg_clean, bins=50, color='purple', edgecolor='black', alpha=0.7)
        axes[3].set_title('Radius of Gyration (Rg) Distribution')
        axes[3].set_xlabel('Radius of Gyration (Rg)')
        axes[3].set_ylabel('Frequency')
        axes[3].grid(True, alpha=0.3)
    else:
        axes[3].text(0.5, 0.5, 'No valid Rg data', horizontalalignment='center',
                     verticalalignment='center', transform=axes[3].transAxes, fontsize=12, color='red')
        axes[3].set_title('Radius of Gyration (Rg) Distribution')
    
    plt.suptitle(f'Distribution of Metrics for {len(simulation_results)} Simulations\n'
                 f'(N={N}, p={inactivation_probability:.3f}, overlap={overlap}, bias={bias_factors}, cell_size={cell_size})',
                 fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    def safe_stats(data, name):
        if len(data) == 0:
            print(f"\n{name}:")
            print(f"  Mean: N/A (No valid data)")
            print(f"  Std:  N/A (No valid data)")
            print(f"  Min:  N/A (No valid data)")
            print(f"  Max:  N/A (No valid data)")
        else:
            print(f"\n{name}:")
            print(f"  Mean: {np.mean(data):.4f}")
            print(f"  Std:  {np.std(data):.4f}")
            print(f"  Min:  {np.min(data):.4f}")
            print(f"  Max:  {np.max(data):.4f}")
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total valid simulations:")
    print(f"  Shape Factor: {len(shape_factors_clean)} / {len(simulation_results)}")
    print(f"  df_v2 (Structure Factor): {len(df_v2_clean)} / {len(simulation_results)}")
    print(f"  Porosity: {len(porosities_clean)} / {len(simulation_results)}")
    print(f"  Rg: {len(Rg_clean)} / {len(simulation_results)}")
    
    safe_stats(shape_factors_clean, "Shape Factor")
    safe_stats(df_v2_clean, "df_v2 (Structure Factor)")
    safe_stats(porosities_clean, "Porosity")
    safe_stats(Rg_clean, "Rg")
    
    if inactivation_probability == 0.0 and len(df_v2_clean) > 0:
        print(f"\nExpected theoretical df for Eden model (p=0): ~3.0")
        print(f"Mean df_v2 deviates by {(np.mean(df_v2_clean) - 3.0):.4f} from expected.")
    
    # Save data if requested
    if save:
        metadata = {
            'function': 'generate_simulation_results_by_seed',
            'parameters': {
                'N': N,
                'bias_factors': bias_factors,
                'overlap': overlap,
                'inactivation_probability': inactivation_probability,
                'primary_radius' : primary_radius,
                'cell_size': cell_size,
                'num_simulations': num_simulations
            }
        }
        filename = f"simulation_results_N{N}_p{inactivation_probability:.3f}"
        save_simulation_results(simulation_results, filename, metadata)
    
    return simulation_results

def generate_simulation_results_by_p(
    N=43,
    bias_factors=(1, 1, 1),
    overlap=0.0,
    max_particles_for_spheres=3000,
    p_range=(0.0, 0.99),
    p_spacing=0.1,
    cell_size=4.0,
    M=100,
    figsize=(18, 6),
    save=False
):
    """
    Generate simulation results by looping over a range of inactivation probabilities p.
    Generates M aggregates for each value of p within the specified range.
    
    Parameters:
    - N: int, number of particles per aggregate
    - bias_factors: tuple, growth bias factors (default: isotropic)
    - overlap: float, particle overlap parameter (default: 0.0)
    - max_particles_for_spheres: int, max particles to render spheres
    - p_range: tuple (min_p, max_p), inclusive range of p values (must be in [0, 1))
    - p_spacing: float, step size between p values (linear spacing)
    - cell_size: float, linked cell grid size (default: 4.0)
    - M: int, number of aggregates to generate per p value
    - figsize: tuple, figure size for the histograms
    - save: bool, if True, saves all simulation data to a text file
    
    Returns:
    - simulation_results: list of dictionaries with structure identical to generate_simulation_results_by_seed()
    """
    
    min_p, max_p = p_range
    if min_p < 0 or max_p >= 1:
        raise ValueError("p_range must be within [0, 1)")
    if min_p > max_p:
        raise ValueError("min_p must be <= max_p")
    
    # Generate evenly spaced p values
    p_values = np.arange(min_p, max_p + p_spacing/2, p_spacing)
    p_values = p_values[p_values < 1.0]
    print(f"Generating aggregates for {len(p_values)} p values: {p_values}")
    print(f"M = {M} aggregates per p value.")
    print(f"cell_size = {cell_size}")
    
    simulation_results = []
    total_simulations = len(p_values) * M
    print(f"Total simulations: {total_simulations}")
    
    np.random.seed(42)
    
    for i_p, p in enumerate(tqdm(p_values, desc="p values")):
        for m in range(M):
            seed = np.random.randint(0, 10**9)
            
            try:
                result = generate_fractal_aggregate(
                    N=N,
                    bias_factors=bias_factors,
                    overlap=overlap,
                    random_seed=seed,
                    max_particles_for_spheres=max_particles_for_spheres,
                    inactivation_probability=p,
                    cell_size=cell_size,  # NEW: pass cell_size
                    visualize=False
                )
                particles = result['particles']  # NEW: extract particles
            except Exception as e:
                print(f"Warning: Failed to generate aggregate for p={p:.3f}, seed={seed}: {e}")
                continue
            
            try:
                sf = calculate_shape_factor(result)
            except Exception as e:
                print(f"Warning: Failed to compute shape factor for p={p:.3f}, seed={seed}: {e}")
                sf = np.nan
            
            try:
                _, _, df_v1, _, _, _ = calculate_and_plot_running_rg(
                    result,
                    fit_start_N=5,
                    fit_end_N=None,
                    visualize=False
                )
            except Exception as e:
                print(f"Warning: Failed to compute df_v1 for p={p:.3f}, seed={seed}: {e}")
                df_v1 = np.nan
            
            try:
                q_array, S_array, Rg, df_v2, alpha_err, _, _ = calculate_structure_factor(
                    result,
                    q_min=0.01,
                    q_max=10.0,
                    n_q=100,
                    R_particle=1.0,
                    fit_range_factor=2.0,
                    visualize=False
                )
            except Exception as e:
                print(f"Warning: Failed to compute df_v2 or Rg for p={p:.3f}, seed={seed}: {e}")
                df_v2 = np.nan
                Rg = np.nan
            
            try:
                porosity, _, _ = calculate_porosity(result, particle_radius=1.0)
            except Exception as e:
                print(f"Warning: Failed to compute porosity for p={p:.3f}, seed={seed}: {e}")
                porosity = np.nan
            
            simulation_results.append({
                'particles': particles,
                'shape_factor': sf,
                'df_v1': df_v1,
                'df_v2': df_v2,
                'porosity': porosity,
                'Rg': Rg,
                'parameters': {
                    'N': N,
                    'bias_factors': bias_factors,
                    'overlap': overlap,
                    'inactivation_probability': p,
                    'cell_size': cell_size,  # NEW: save cell_size
                    'random_seed': seed
                }
            })
    
    print(f"\nGenerated {len(simulation_results)} total simulation results out of {total_simulations} attempts.")
    
    # --- HISTOGRAM VISUALIZATION ---
    shape_factors_clean = np.array([res['shape_factor'] for res in simulation_results if not np.isnan(res['shape_factor'])])
    df_v2_clean = np.array([res['df_v2'] for res in simulation_results if not np.isnan(res['df_v2'])])
    porosities_clean = np.array([res['porosity'] for res in simulation_results if not np.isnan(res['porosity'])])
    Rg_clean = np.array([res['Rg'] for res in simulation_results if not np.isnan(res['Rg'])])
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Shape Factor
    if len(shape_factors_clean) > 0:
        axes[0].hist(shape_factors_clean, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].set_title('Shape Factor Distribution')
        axes[0].set_xlabel('Shape Factor (sqrt(λ_max/λ_min))')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, 'No valid Shape Factor data', horizontalalignment='center',
                     verticalalignment='center', transform=axes[0].transAxes, fontsize=12, color='red')
        axes[0].set_title('Shape Factor Distribution')
    
    # Fractal Dimension
    if len(df_v2_clean) > 0:
        axes[1].hist(df_v2_clean, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
        axes[1].set_title('Fractal Dimension (Structure Factor)')
        axes[1].set_xlabel('Mass Fractal Dimension (df)')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No valid df_v2 data', horizontalalignment='center',
                     verticalalignment='center', transform=axes[1].transAxes, fontsize=12, color='red')
        axes[1].set_title('Fractal Dimension (Structure Factor)')
    
    # Porosity
    if len(porosities_clean) > 0:
        axes[2].hist(porosities_clean, bins=50, color='orange', edgecolor='black', alpha=0.7)
        axes[2].set_title('Porosity Distribution')
        axes[2].set_xlabel('Porosity (ϵ)')
        axes[2].set_ylabel('Frequency')
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'No valid Porosity data', horizontalalignment='center',
                     verticalalignment='center', transform=axes[2].transAxes, fontsize=12, color='red')
        axes[2].set_title('Porosity Distribution')
    
    # Radius of Gyration
    if len(Rg_clean) > 0:
        axes[3].hist(Rg_clean, bins=50, color='purple', edgecolor='black', alpha=0.7)
        axes[3].set_title('Radius of Gyration (Rg) Distribution')
        axes[3].set_xlabel('Radius of Gyration (Rg)')
        axes[3].set_ylabel('Frequency')
        axes[3].grid(True, alpha=0.3)
    else:
        axes[3].text(0.5, 0.5, 'No valid Rg data', horizontalalignment='center',
                     verticalalignment='center', transform=axes[3].transAxes, fontsize=12, color='red')
        axes[3].set_title('Radius of Gyration (Rg) Distribution')
    
    plt.suptitle(f'Distribution of Metrics for {len(simulation_results)} Simulations\n'
                 f'(N={N}, p ∈ [{p_range[0]:.2f}, {p_range[1]:.2f}], Δp={p_spacing}, M={M}, cell_size={cell_size})',
                 fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    def safe_stats(data, name):
        if len(data) == 0:
            print(f"\n{name}:")
            print(f"  Mean: N/A (No valid data)")
            print(f"  Std:  N/A (No valid data)")
            print(f"  Min:  N/A (No valid data)")
            print(f"  Max:  N/A (No valid data)")
        else:
            print(f"\n{name}:")
            print(f"  Mean: {np.mean(data):.4f}")
            print(f"  Std:  {np.std(data):.4f}")
            print(f"  Min:  {np.min(data):.4f}")
            print(f"  Max:  {np.max(data):.4f}")
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total valid simulations:")
    print(f"  Shape Factor: {len(shape_factors_clean)} / {len(simulation_results)}")
    print(f"  df_v2 (Structure Factor): {len(df_v2_clean)} / {len(simulation_results)}")
    print(f"  Porosity: {len(porosities_clean)} / {len(simulation_results)}")
    print(f"  Rg: {len(Rg_clean)} / {len(simulation_results)}")
    
    safe_stats(shape_factors_clean, "Shape Factor")
    safe_stats(df_v2_clean, "df_v2 (Structure Factor)")
    safe_stats(porosities_clean, "Porosity")
    safe_stats(Rg_clean, "Rg")
    
    if p_range[0] == 0.0 and len(df_v2_clean) > 0:
        print(f"\nExpected theoretical df for Eden model (p=0): ~3.0")
        print(f"Mean df_v2 deviates by {(np.mean(df_v2_clean) - 3.0):.4f} from expected.")
    
    # Save data if requested
    if save:
        metadata = {
            'function': 'generate_simulation_results_by_p',
            'parameters': {
                'N': N,
                'bias_factors': bias_factors,
                'overlap': overlap,
                'inactivation_probability': inactivation_probability,
                'cell_size': cell_size,
                'num_simulations': num_simulations
            }
        }
        filename = f"simulation_results_N{N}_p{inactivation_probability:.3f}"
        save_simulation_results(simulation_results, filename, metadata)
    
    return simulation_results


def generate_simulation_results_by_N(
    N_range=(10, 100),
    bias_factors=(1, 1, 1),
    overlap=0.0,
    max_particles_for_spheres=3000,
    inactivation_probability=0.0,
    cell_size=4.0,
    M=50,
    num_N_points=20,
    N_spacing='linear',
    figsize=(18, 6),
    save=False
):
    """
    Generate simulation results by looping over a range of particle numbers N.
    Generates M aggregates for each value of N within the specified range.
    
    Parameters:
    - N_range: tuple (min_N, max_N), inclusive range of N values
    - bias_factors: tuple, growth bias factors (default: isotropic)
    - overlap: float, particle overlap parameter (default: 0.0)
    - max_particles_for_spheres: int, max particles to render spheres
    - inactivation_probability: float, probability p (fixed across all simulations)
    - cell_size: float, linked cell grid size (default: 4.0)
    - M: int, number of aggregates to generate per N value
    - num_N_points: int, number of distinct N values to sample
    - N_spacing: str, 'linear' or 'log' spacing type for N values
    - figsize: tuple, figure size for histograms
    - save: bool, if True, saves all simulation data
    
    Returns:
    - simulation_results: list of dictionaries with structure identical to other generate_simulation_results_by_* functions
    """
    
    min_N, max_N = N_range
    if min_N < 1 or max_N < min_N:
        raise ValueError("N_range must have min_N >= 1 and max_N >= min_N")
    if N_spacing not in ['linear', 'log']:
        raise ValueError("N_spacing must be 'linear' or 'log'")
    
    # Generate N values
    if N_spacing == 'linear':
        N_values = np.linspace(min_N, max_N, num_N_points, dtype=int)
    else:  # log spacing
        N_values_log = np.logspace(np.log10(min_N), np.log10(max_N), num_N_points)
        N_values = np.round(N_values_log).astype(int)
        N_values = np.unique(N_values)
        if len(N_values) < num_N_points:
            print(f"Warning: Log spacing resulted in {len(N_values)} unique N values instead of {num_N_points}.")
    
    print(f"Generating {M} aggregates for each of {len(N_values)} N values: {N_values}")
    print(f"Fixed parameters: p = {inactivation_probability}, bias={bias_factors}, overlap={overlap}, cell_size={cell_size}")
    
    simulation_results = []
    total_simulations = len(N_values) * M
    print(f"Total simulations: {total_simulations}")
    
    np.random.seed(42)
    
    for i_N, N in enumerate(tqdm(N_values, desc="N values")):
        for m in range(M):
            seed = np.random.randint(0, 10**9)
            
            try:
                result = generate_fractal_aggregate(
                    N=int(N),
                    bias_factors=bias_factors,
                    overlap=overlap,
                    random_seed=seed,
                    max_particles_for_spheres=max_particles_for_spheres,
                    inactivation_probability=inactivation_probability,
                    cell_size=cell_size,  # NEW: pass cell_size
                    visualize=False
                )
                particles = result['particles']  # NEW: extract particles
            except Exception as e:
                print(f"Warning: Failed to generate aggregate for N={N}, seed={seed}: {e}")
                continue
            
            try:
                sf = calculate_shape_factor(result)
            except Exception as e:
                print(f"Warning: Failed to compute shape factor for N={N}, seed={seed}: {e}")
                sf = np.nan
            
            try:
                _, _, df_v1, _, _, _ = calculate_and_plot_running_rg(
                    result,
                    fit_start_N=5,
                    fit_end_N=None,
                    visualize=False
                )
            except Exception as e:
                print(f"Warning: Failed to compute df_v1 for N={N}, seed={seed}: {e}")
                df_v1 = np.nan
            
            try:
                q_array, S_array, Rg, df_v2, alpha_err, _, _ = calculate_structure_factor(
                    result,
                    q_min=0.01,
                    q_max=10.0,
                    n_q=100,
                    R_particle=1.0,
                    fit_range_factor=2.0,
                    visualize=False
                )
            except Exception as e:
                print(f"Warning: Failed to compute df_v2 or Rg for N={N}, seed={seed}: {e}")
                df_v2 = np.nan
                Rg = np.nan
            
            try:
                porosity, _, _ = calculate_porosity(result, particle_radius=1.0)
            except Exception as e:
                print(f"Warning: Failed to compute porosity for N={N}, seed={seed}: {e}")
                porosity = np.nan
            
            simulation_results.append({
                'particles': particles,
                'shape_factor': sf,
                'df_v1': df_v1,
                'df_v2': df_v2,
                'porosity': porosity,
                'Rg': Rg,
                'parameters': {
                    'N': N,
                    'bias_factors': bias_factors,
                    'overlap': overlap,
                    'inactivation_probability': inactivation_probability,
                    'cell_size': cell_size,  # NEW: save cell_size
                    'random_seed': seed
                }
            })
    
    print(f"\nGenerated {len(simulation_results)} total simulation results out of {total_simulations} attempts.")
    
    # --- HISTOGRAM VISUALIZATION ---
    shape_factors_clean = np.array([res['shape_factor'] for res in simulation_results if not np.isnan(res['shape_factor'])])
    df_v2_clean = np.array([res['df_v2'] for res in simulation_results if not np.isnan(res['df_v2'])])
    porosities_clean = np.array([res['porosity'] for res in simulation_results if not np.isnan(res['porosity'])])
    Rg_clean = np.array([res['Rg'] for res in simulation_results if not np.isnan(res['Rg'])])
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Shape Factor
    if len(shape_factors_clean) > 0:
        axes[0].hist(shape_factors_clean, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].set_title('Shape Factor Distribution')
        axes[0].set_xlabel('Shape Factor (sqrt(λ_max/λ_min))')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, 'No valid Shape Factor data', horizontalalignment='center',
                     verticalalignment='center', transform=axes[0].transAxes, fontsize=12, color='red')
        axes[0].set_title('Shape Factor Distribution')
    
    # Fractal Dimension
    if len(df_v2_clean) > 0:
        axes[1].hist(df_v2_clean, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
        axes[1].set_title('Fractal Dimension (Structure Factor)')
        axes[1].set_xlabel('Mass Fractal Dimension (df)')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No valid df_v2 data', horizontalalignment='center',
                     verticalalignment='center', transform=axes[1].transAxes, fontsize=12, color='red')
        axes[1].set_title('Fractal Dimension (Structure Factor)')
    
    # Porosity
    if len(porosities_clean) > 0:
        axes[2].hist(porosities_clean, bins=50, color='orange', edgecolor='black', alpha=0.7)
        axes[2].set_title('Porosity Distribution')
        axes[2].set_xlabel('Porosity (ϵ)')
        axes[2].set_ylabel('Frequency')
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'No valid Porosity data', horizontalalignment='center',
                     verticalalignment='center', transform=axes[2].transAxes, fontsize=12, color='red')
        axes[2].set_title('Porosity Distribution')
    
    # Radius of Gyration
    if len(Rg_clean) > 0:
        axes[3].hist(Rg_clean, bins=50, color='purple', edgecolor='black', alpha=0.7)
        axes[3].set_title('Radius of Gyration (Rg) Distribution')
        axes[3].set_xlabel('Radius of Gyration (Rg)')
        axes[3].set_ylabel('Frequency')
        axes[3].grid(True, alpha=0.3)
    else:
        axes[3].text(0.5, 0.5, 'No valid Rg data', horizontalalignment='center',
                     verticalalignment='center', transform=axes[3].transAxes, fontsize=12, color='red')
        axes[3].set_title('Radius of Gyration (Rg) Distribution')
    
    plt.suptitle(f'Distribution of Metrics for {len(simulation_results)} Simulations\n'
                 f'(p={inactivation_probability:.3f}, N ∈ [{min_N}, {max_N}], {N_spacing} spaced, {num_N_points} points, M={M}, cell_size={cell_size})',
                 fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    def safe_stats(data, name):
        if len(data) == 0:
            print(f"\n{name}:")
            print(f"  Mean: N/A (No valid data)")
            print(f"  Std:  N/A (No valid data)")
            print(f"  Min:  N/A (No valid data)")
            print(f"  Max:  N/A (No valid data)")
        else:
            print(f"\n{name}:")
            print(f"  Mean: {np.mean(data):.4f}")
            print(f"  Std:  {np.std(data):.4f}")
            print(f"  Min:  {np.min(data):.4f}")
            print(f"  Max:  {np.max(data):.4f}")
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total valid simulations:")
    print(f"  Shape Factor: {len(shape_factors_clean)} / {len(simulation_results)}")
    print(f"  df_v2 (Structure Factor): {len(df_v2_clean)} / {len(simulation_results)}")
    print(f"  Porosity: {len(porosities_clean)} / {len(simulation_results)}")
    print(f"  Rg: {len(Rg_clean)} / {len(simulation_results)}")
    
    safe_stats(shape_factors_clean, "Shape Factor")
    safe_stats(df_v2_clean, "df_v2 (Structure Factor)")
    safe_stats(porosities_clean, "Porosity")
    safe_stats(Rg_clean, "Rg")
    
    if inactivation_probability == 0.0 and len(df_v2_clean) > 0:
        print(f"\nExpected theoretical df for Eden model (p=0): ~3.0")
        print(f"Mean df_v2 deviates by {(np.mean(df_v2_clean) - 3.0):.4f} from expected.")
    
    # Save data if requested

    if save:
        metadata = {
            'function': 'generate_simulation_results_by_N',
            'parameters': {
                'N': N,
                'bias_factors': bias_factors,
                'overlap': overlap,
                'inactivation_probability': inactivation_probability,
                'cell_size': cell_size,
                'num_simulations': num_simulations
            }
        }
        filename = f"simulation_results_N{N}_p{inactivation_probability:.3f}"
        save_simulation_results(simulation_results, filename, metadata)
    
    return simulation_results

from pathlib import Path
from typing import List, Dict, Union


def load_simulation_results(filepath: Union[str, Path]) -> List[Dict]:
    """
    Load simulation results from a text file created by save_simulation_results().
    Returns EXACTLY the same data structure as generate_simulation_results_by_seed().
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    print(f"Loading simulation results from {filepath}...")
    
    # First pass: read metadata from comment lines
    metadata = {}
    with open(filepath, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                break
            if 'Saved:' in line:
                metadata['timestamp'] = line.split('Saved:')[-1].strip()
            elif 'Total Simulations:' in line:
                try:
                    metadata['num_simulations'] = int(line.split('Total Simulations:')[-1].strip())
                except:
                    metadata['num_simulations'] = 0
            elif 'Parameters:' in line:
                params_str = line.replace('# Parameters:', '').strip()
                params = {}
                for part in params_str.split(','):
                    if '=' in part:
                        key, value = part.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        try:
                            if value.startswith('(') and value.endswith(')'):
                                tuple_vals = value[1:-1].split(',')
                                value = tuple(float(v.strip()) for v in tuple_vals if v.strip())
                            else:
                                if '.' in value or 'e' in value.lower():
                                    val_float = float(value)
                                    value = int(val_float) if val_float.is_integer() else val_float
                                else:
                                    value = int(value)
                        except:
                            pass
                        params[key] = value
                metadata['parameters'] = params
    
    # Second pass: read the data using pandas
    import pandas as pd
    data_df = pd.read_csv(filepath, sep='\t', comment='#')
    
    simulation_results = []
    
    # Process each row
    for _, row in data_df.iterrows():
        # Extract metrics
        shape_factor = float(row['shape_factor'])
        df_v1 = float(row['df_v1'])
        df_v2 = float(row['df_v2'])
        porosity = float(row['porosity'])
        Rg = float(row['Rg'])
        random_seed = int(float(row['random_seed']))
        
        # Extract particles
        particles = []
        particle_index = 0
        
        # Extract particles until we run out of valid coordinates
        while True:
            x_col = f'particle_{particle_index+1}_x'
            y_col = f'particle_{particle_index+1}_y'
            z_col = f'particle_{particle_index+1}_z'
            
            # Check if these columns exist in the DataFrame
            if x_col not in data_df.columns or y_col not in data_df.columns or z_col not in data_df.columns:
                break
                
            try:
                x_val = row[x_col]
                y_val = row[y_col]
                z_val = row[z_col]
                
                # Skip if coordinates are missing or marked as N/A
                if pd.isna(x_val) or pd.isna(y_val) or pd.isna(z_val) or str(x_val).lower() == 'n/a':
                    particle_index += 1
                    continue
                
                # Create particle with EXACT same structure as generator
                particle = {
                    'position': np.array([float(x_val), float(y_val), float(z_val)], dtype=np.float64),
                    'added_step': particle_index,
                    'inactive': False
                }
                particles.append(particle)
                particle_index += 1
            except (ValueError, TypeError, KeyError):
                break
        
        # Reconstruct parameters dictionary EXACTLY as generator produces
        default_params = metadata.get('parameters', {})
        parameters = {
            'N': len(particles),
            'random_seed': random_seed,
            'bias_factors': default_params.get('bias_factors', (1, 1, 1)),
            'overlap': default_params.get('overlap', 0.0),
            'primary_radius': default_params.get('primary_radius', 1.0),
            'inactivation_probability': default_params.get('inactivation_probability', 0.0),
            'cell_size': default_params.get('cell_size', 4.0)
        }
        
        # Build result dict with EXACT same structure as generator
        result = {
            'particles': particles,
            'shape_factor': shape_factor,
            'df_v1': df_v1,
            'df_v2': df_v2,
            'porosity': porosity,
            'Rg': Rg,
            'parameters': parameters
        }
        
        simulation_results.append(result)
    
    print(f"✓ Successfully loaded {len(simulation_results)} simulations with EXACT structure matching generator output")
    return simulation_results

def print_simulation_summary(
    simulation_results: List[Dict],
    metadata: Dict
) -> None:
    """
    Print a comprehensive summary of loaded simulation results.
    
    Parameters:
    -----------
    simulation_results : list of dicts
        Results from load_simulation_results()
    
    metadata : dict
        Metadata from load_simulation_results()
    """
    
    print("\n" + "="*70)
    print("SIMULATION RESULTS SUMMARY")
    print("="*70)
    
    if 'Saved' in metadata:
        print(f"Saved on: {metadata['Saved']}")
    
    print(f"Number of simulations: {len(simulation_results)}")
    
    if 'Parameters' in metadata:
        print(f"Parameters: {metadata['Parameters']}")
    
    # Extract metrics
    shape_factors = np.array([r['shape_factor'] for r in simulation_results if not np.isnan(r['shape_factor'])])
    df_v1_vals = np.array([r['df_v1'] for r in simulation_results if not np.isnan(r['df_v1'])])
    df_v2_vals = np.array([r['df_v2'] for r in simulation_results if not np.isnan(r['df_v2'])])
    porosities = np.array([r['porosity'] for r in simulation_results if not np.isnan(r['porosity'])])
    Rg_vals = np.array([r['Rg'] for r in simulation_results if not np.isnan(r['Rg'])])
    
    print(f"\nMetric Coverage:")
    print(f"  Shape Factor: {len(shape_factors)}/{len(simulation_results)} valid")
    print(f"  df_v1 (Running Rg): {len(df_v1_vals)}/{len(simulation_results)} valid")
    print(f"  df_v2 (Structure Factor): {len(df_v2_vals)}/{len(simulation_results)} valid")
    print(f"  Porosity: {len(porosities)}/{len(simulation_results)} valid")
    print(f"  Rg: {len(Rg_vals)}/{len(simulation_results)} valid")
    
    # Print statistics
    def print_stats(data, name):
        if len(data) > 0:
            print(f"\n{name}:")
            print(f"  Mean: {np.mean(data):.6f}")
            print(f"  Std:  {np.std(data):.6f}")
            print(f"  Min:  {np.min(data):.6f}")
            print(f"  Max:  {np.max(data):.6f}")
        else:
            print(f"\n{name}: No valid data")
    
    print_stats(shape_factors, "Shape Factor")
    print_stats(df_v1_vals, "df_v1 (Running Rg)")
    print_stats(df_v2_vals, "df_v2 (Structure Factor)")
    print_stats(porosities, "Porosity")
    print_stats(Rg_vals, "Radius of Gyration (Rg)")
    
    print("\n" + "="*70)

def _extract_particles_for_filtering(particles_or_result):
    """
    Extract particles from either dict (new format) or list (old format).
    Returns particles list and result dict if available.
    """
    if isinstance(particles_or_result, dict) and 'particles' in particles_or_result:
        # NEW FORMAT: result dict
        particles = particles_or_result['particles']
        result_dict = particles_or_result
    elif isinstance(particles_or_result, list):
        # OLD FORMAT: particles list only
        particles = particles_or_result
        result_dict = None
    else:
        raise TypeError("Must be list or result dict")
    
    return particles, result_dict


def visualize_filtered_aggregates(
    simulation_results,
    min_shape_factor=None,
    max_shape_factor=None,
    min_df_v2=None,
    max_df_v2=None,
    min_porosity=None,
    max_porosity=None,
    min_rg=None,
    max_rg=None,
    plot_type='scatter',
    color_by='addition_step',
    point_size_par=15,
    figsize=(12, 12),
    max_particles_for_spheres=50,
    base_margin_factor=1.5
):
    """
    Filter simulation results and visualize up to 9 randomly selected aggregates.
    
    Accepts result dicts from generate_simulation_results_by_* batch functions
    or simple particles lists for backward compatibility.
    
    Parameters:
    -----------
    simulation_results : list
        List of result dicts from batch generation functions, or list of particles lists
    
    min/max_shape_factor : float or None
        Filter range for shape factor
    
    min/max_df_v2 : float or None
        Filter range for fractal dimension (structure factor method)
    
    min/max_porosity : float or None
        Filter range for porosity (void fraction)
    
    min/max_rg : float or None
        Filter range for radius of gyration
    
    plot_type : str
        'scatter' or 'convex_hull'
    
    color_by : str
        'active_inactive', 'addition_step', or specific color name
    
    figsize : tuple
        Figure size (width, height)
    
    max_particles_for_spheres : int
        Render spheres for aggregates with N <= this value
    
    base_margin_factor : float
        Zoom control (1.5 = default, <1.0 = tighter, >1.0 = looser)
    
    Returns:
    --------
    filtered_results : list
        List of result dicts that passed the filtering criteria
    """
    
    # Filter simulation results based on criteria
    filtered_results = simulation_results.copy()
    
    if min_shape_factor is not None:
        filtered_results = [
            res for res in filtered_results 
            if res.get('shape_factor', np.nan) >= min_shape_factor
        ]
    if max_shape_factor is not None:
        filtered_results = [
            res for res in filtered_results 
            if res.get('shape_factor', np.nan) <= max_shape_factor
        ]
    
    if min_df_v2 is not None:
        filtered_results = [
            res for res in filtered_results 
            if res.get('df_v2', np.nan) >= min_df_v2
        ]
    if max_df_v2 is not None:
        filtered_results = [
            res for res in filtered_results 
            if res.get('df_v2', np.nan) <= max_df_v2
        ]
    
    if min_porosity is not None:
        filtered_results = [
            res for res in filtered_results 
            if res.get('porosity', np.nan) >= min_porosity
        ]
    if max_porosity is not None:
        filtered_results = [
            res for res in filtered_results 
            if res.get('porosity', np.nan) <= max_porosity
        ]
    
    if min_rg is not None:
        filtered_results = [
            res for res in filtered_results 
            if res.get('Rg', np.nan) >= min_rg
        ]
    if max_rg is not None:
        filtered_results = [
            res for res in filtered_results 
            if res.get('Rg', np.nan) <= max_rg
        ]
    
    if len(filtered_results) == 0:
        raise ValueError("No simulations match the given filtering criteria.")
    
    # Randomly sample up to 9 aggregates
    n_samples = min(9, len(filtered_results))
    sampled_indices = np.random.choice(len(filtered_results), size=n_samples, replace=False)
    sampled_aggregates = [filtered_results[i] for i in sampled_indices]
    
    # Determine subplot grid layout dynamically
    n_cols = min(3, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    # Create subplot grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, subplot_kw={'projection': '3d'})
    
    # Handle case where only one row or column
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Build dynamic title with filtering conditions
    title_parts = ["Filtered Aggregates"]
    if min_shape_factor is not None:
        title_parts.append(f"SF ≥ {min_shape_factor:.3f}")
    if max_shape_factor is not None:
        title_parts.append(f"SF ≤ {max_shape_factor:.3f}")
    if min_df_v2 is not None:
        title_parts.append(f"df ≥ {min_df_v2:.3f}")
    if max_df_v2 is not None:
        title_parts.append(f"df ≤ {max_df_v2:.3f}")
    if min_porosity is not None:
        title_parts.append(f"ϵ ≥ {min_porosity:.3f}")
    if max_porosity is not None:
        title_parts.append(f"ϵ ≤ {max_porosity:.3f}")
    if min_rg is not None:
        title_parts.append(f"Rg ≥ {min_rg:.3f}")
    if max_rg is not None:
        title_parts.append(f"Rg ≤ {max_rg:.3f}")
    
    title_str = " | ".join(title_parts)
    fig.suptitle(title_str, fontsize=16)
    
    # Precompute sphere mesh once for efficiency
    u = np.linspace(0, 2 * np.pi, 15)
    v = np.linspace(0, np.pi, 15)
    cos_u = np.cos(u)
    sin_u = np.sin(u)
    cos_v = np.cos(v)
    sin_v = np.sin(v)
    x_sphere = np.outer(cos_u, sin_v)
    y_sphere = np.outer(sin_u, sin_v)
    z_sphere = np.outer(np.ones_like(cos_u), cos_v)
    
    # Flatten axes array for easy iteration
    axes_flat = axes.flatten()
    
    for idx, result in enumerate(sampled_aggregates):
        ax = axes_flat[idx]
        
        # Extract particles from result dict
        particles, _ = _extract_particles_for_filtering(result)
        positions = np.array([p['position'] for p in particles])
        N = len(positions)
        
        # Compute cubic bounding box with dynamic zoom
        max_extent = np.max(np.abs(positions))
        
        # Dynamic margin based on N
        if N < 100:
            margin_factor = base_margin_factor * 0.5
        elif N < 500:
            margin_factor = base_margin_factor
        else:
            margin_factor = base_margin_factor * 1.2
        
        margin = margin_factor * max_extent
        
        ax.set_xlim(-max_extent - margin, max_extent + margin)
        ax.set_ylim(-max_extent - margin, max_extent + margin)
        ax.set_zlim(-max_extent - margin, max_extent + margin)
        
        if plot_type == 'scatter':
            # Scatter plot with coloring options
            if color_by == 'active_inactive':
                active_mask = np.array([not p['inactive'] for p in particles])
                inactive_mask = ~active_mask
                
                if np.any(active_mask):
                    ax.scatter(positions[active_mask, 0], positions[active_mask, 1], positions[active_mask, 2],
                               c='orange', s=15, alpha=0.9, label='Active')
                    
                    if len(particles) <= max_particles_for_spheres:
                        for pos in positions[active_mask]:
                            x = 1.0 * x_sphere + pos[0]
                            y = 1.0 * y_sphere + pos[1]
                            z = 1.0 * z_sphere + pos[2]
                            ax.plot_surface(x, y, z, color='orange', alpha=0.9, rstride=2, cstride=2, linewidth=0)
                
                if np.any(inactive_mask):
                    ax.scatter(positions[inactive_mask, 0], positions[inactive_mask, 1], positions[inactive_mask, 2],
                               c='gray', s=10, alpha=0.9, label='Inactive')
                    
                    if len(particles) <= max_particles_for_spheres:
                        for pos in positions[inactive_mask]:
                            x = 1.0 * x_sphere + pos[0]
                            y = 1.0 * y_sphere + pos[1]
                            z = 1.0 * z_sphere + pos[2]
                            ax.plot_surface(x, y, z, color='gray', alpha=0.9, rstride=2, cstride=2, linewidth=0)
            
            elif color_by == 'addition_step':
                added_steps = np.array([p['added_step'] for p in particles])
                viridis_cmap = plt.cm.viridis
                colors = viridis_cmap(added_steps / np.max(added_steps))
                
                ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                           c=colors, s=15, alpha=1)
                
                if len(particles) <= max_particles_for_spheres:
                    for pos, color in zip(positions, colors):
                        x = 1.0 * x_sphere + pos[0]
                        y = 1.0 * y_sphere + pos[1]
                        z = 1.0 * z_sphere + pos[2]
                        ax.plot_surface(x, y, z, color=color, alpha=0.9, rstride=2, cstride=2, linewidth=0)
            
            else:
                # Use specified color
                ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                           c=color_by, s=point_size_par, alpha=0.5)
        
        elif plot_type == 'convex_hull':
            # Plot convex hull surface
            try:
                hull = ConvexHull(positions)
                for simplex in hull.simplices:
                    triangle = positions[simplex]
                    ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2],
                                   color='gray', alpha=0.3, linewidth=0.5)
            except Exception as e:
                print(f"Warning: Could not compute convex hull for aggregate {idx}: {e}")
                ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                          c='gray', s=10, alpha=0.7)
        
        # Remove visual elements
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_axis_off()
        
        # Set viewing angle
        ax.view_init(elev=20, azim=45)
        ax.set_box_aspect([1, 1, 1])
    
    # Hide unused subplots
    for idx in range(n_samples, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Filtered {len(filtered_results)} simulations. Displaying {n_samples} random samples.")
    
    return filtered_results

def generate_simulation_results_by_seed_PDX(
    N_dist_params={'type': 'lognormal', 'mean': 4.6, 'sigma': 0.8},  # Default ~100 particles mean
    bias_factors=(1, 1, 1),
    overlap=0.0,
    max_particles_for_spheres=3000,
    inactivation_probability=0.0,
    primary_radius=1.0,
    cell_size=4.0,
    num_simulations=1000,
    figsize=(18, 6),
    save=False,
    random_seed_base=1241434
):
    """
    Generate simulation results by looping over random seeds with polydisperse aggregate sizes.
    Each simulation uses a unique random seed and samples N (particle count) from a distribution.
    Generates histograms for Shape Factor, Fractal Dimension (df_v2), Porosity, and Radius of Gyration (Rg).
    
    Parameters:
    -----------
    N_dist_params : dict
        Distribution parameters for number of particles per aggregate.
        Format: {'type': 'lognormal', 'mean': X, 'sigma': Y} OR
                {'type': 'lognormal', 'target_mean': X, 'cv': Y}
                
        Examples:
        - For mean ~100 particles with long tail: {'type': 'lognormal', 'mean': 4.6, 'sigma': 0.8}
        - For mean ~100 particles with tight distribution: {'type': 'lognormal', 'mean': 4.6, 'sigma': 0.3}
        - For mostly small aggregates (mean ~30): {'type': 'lognormal', 'mean': 3.4, 'sigma': 0.6}
        - To specify by target mean and coefficient of variation:
          {'type': 'lognormal', 'target_mean': 100, 'cv': 0.5}
    
    bias_factors : tuple
        Growth bias factors (default: isotropic)
    overlap : float
        Particle overlap parameter (default: 0.0)
    max_particles_for_spheres : int
        Max particles to render spheres for visualization
    inactivation_probability : float
        Probability p for particle inactivation (fixed across all simulations)
    primary_radius : float
        Size of primary particle (default: 1.0)
    cell_size : float
        Linked cell grid size (default: 4.0)
    num_simulations : int
        Number of aggregates to generate (each with a unique random seed)
    figsize : tuple
        Figure size for the histograms
    save : bool
        If True, saves all simulation data to a text file
    random_seed_base : int
        Base seed for generating unique random seeds (default: 1241434)
    
    Returns:
    --------
    simulation_results : list of dictionaries with structure:
        'particles', 'shape_factor', 'df_v1', 'df_v2', 'porosity', 'Rg', 'parameters'
    
    Notes:
    ------
    This function is fully compatible with:
    - save_simulation_results() and load_simulation_results()
    - visualize_filtered_aggregates()
    - xyz_saver()
    - generate_agglomerate()
    
    The polydispersity is at the aggregate level (different N per aggregate), not primary particle level.
    """
    # Validate N_dist_params
    if not isinstance(N_dist_params, dict) or 'type' not in N_dist_params:
        raise ValueError("N_dist_params must be a dict with 'type' key")
    
    if N_dist_params['type'] == 'lognormal':
        # Handle both parameterization methods
        if 'target_mean' in N_dist_params and 'cv' in N_dist_params:
            # Convert target mean and coefficient of variation to lognormal parameters
            target_mean = N_dist_params['target_mean']
            cv = N_dist_params['cv']
            sigma = np.sqrt(np.log(1 + cv**2))
            mu = np.log(target_mean) - 0.5 * sigma**2
            N_dist_params['mean'] = mu
            N_dist_params['sigma'] = sigma
        elif 'mean' not in N_dist_params or 'sigma' not in N_dist_params:
            raise ValueError("For lognormal distribution, need 'mean' and 'sigma' or 'target_mean' and 'cv'")
    else:
        raise ValueError(f"Unsupported distribution type: {N_dist_params['type']}")
    
    # Set base seed for generating unique random seeds
    np.random.seed(random_seed_base)
    
    print(f"Generating {num_simulations} aggregates with polydisperse sizes:")
    print(f"  N distribution: {N_dist_params}")
    print(f"  bias_factors = {bias_factors}")
    print(f"  overlap = {overlap}")
    print(f"  inactivation_probability (p) = {inactivation_probability}")
    print(f"  cell_size = {cell_size}")
    
    simulation_results = []
    N_values = []
    generation_times = []
    
    print(f"Running {num_simulations} simulations...")
    for i in tqdm(range(num_simulations), desc="Simulations"):
        # Generate a unique random seed for each simulation
        seed = np.random.randint(0, 10**9)
        
        # Sample N from the specified distribution
        if N_dist_params['type'] == 'lognormal':
            N_sample = int(np.round(np.random.lognormal(
                mean=N_dist_params['mean'],
                sigma=N_dist_params['sigma']
            )))
            # Ensure N is at least 3 for meaningful calculations
            N_sample = max(3, N_sample)
        else:
            raise ValueError(f"Unsupported distribution type: {N_dist_params['type']}")
        
        N_values.append(N_sample)
        
        # Generate the aggregate
        try:
            result = generate_fractal_aggregate(
                N=N_sample,
                radius=primary_radius,
                bias_factors=bias_factors,
                overlap=overlap,
                random_seed=seed,
                max_particles_for_spheres=max_particles_for_spheres,
                inactivation_probability=inactivation_probability,
                cell_size=cell_size,
                visualize=False
            )
            particles = result['particles']
        except Exception as e:
            print(f"Warning: Failed to generate aggregate for seed {seed} with N={N_sample}: {e}")
            continue
        
        # Calculate metrics
        try:
            sf = calculate_shape_factor(result)
        except Exception as e:
            print(f"Warning: Failed to compute shape factor for seed {seed}: {e}")
            sf = np.nan
        
        try:
            _, _, df_v1, _, _, _ = calculate_and_plot_running_rg(
                result,
                fit_start_N=5,
                fit_end_N=None,
                visualize=False
            )
        except Exception as e:
            print(f"Warning: Failed to compute df_v1 for seed {seed}: {e}")
            df_v1 = np.nan
        
        try:
            q_array, S_array, Rg, df_v2, alpha_err, _, _ = calculate_structure_factor(
                result,
                q_min=0.01,
                q_max=10.0,
                n_q=100,
                R_particle=primary_radius,
                fit_range_factor=2.0,
                visualize=False
            )
        except Exception as e:
            print(f"Warning: Failed to compute df_v2 or Rg for seed {seed}: {e}")
            df_v2 = np.nan
            Rg = np.nan
        
        try:
            porosity, _, _ = calculate_porosity(result, particle_radius=primary_radius)
        except Exception as e:
            print(f"Warning: Failed to compute porosity for seed {seed}: {e}")
            porosity = np.nan
        
        # Store results with consistent structure
        simulation_results.append({
            'particles': particles,
            'shape_factor': sf,
            'df_v1': df_v1,
            'df_v2': df_v2,
            'porosity': porosity,
            'Rg': Rg,
            'parameters': {
                'N': N_sample,  # Actual N used for this simulation
                'bias_factors': bias_factors,
                'overlap': overlap,
                'inactivation_probability': inactivation_probability,
                'primary_radius': primary_radius,
                'cell_size': cell_size,
                'random_seed': seed,
                'N_distribution': N_dist_params  # Store distribution info
            }
        })
    
    print(f"\nGenerated {len(simulation_results)} total simulation results out of {num_simulations} attempts.")
    
    # --- CLEAN METRICS DATA (REMOVE NaNs) ---
    shape_factors_clean = np.array([res['shape_factor'] for res in simulation_results if not np.isnan(res['shape_factor'])])
    df_v2_clean = np.array([res['df_v2'] for res in simulation_results if not np.isnan(res['df_v2'])])
    porosities_clean = np.array([res['porosity'] for res in simulation_results if not np.isnan(res['porosity'])])
    Rg_clean = np.array([res['Rg'] for res in simulation_results if not np.isnan(res['Rg'])])
    N_values_clean = np.array([res['parameters']['N'] for res in simulation_results])
    
    # # --- N DISTRIBUTION PLOT ---
    # plt.figure(figsize=(10, 6))
    # plt.hist(N_values, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    # plt.axvline(x=np.mean(N_values), color='red', linestyle='--', label=f'Mean N = {np.mean(N_values):.1f}')
    # plt.title('Distribution of Aggregate Sizes (Number of Particles)')
    # plt.xlabel('Number of Particles (N)')
    # plt.ylabel('Frequency')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.show()
    
    # --- QUARTILE-BASED VISUALIZATION WITH STACKED HISTOGRAMS ---
    # Extract all N values
    N_values = [res['parameters']['N'] for res in simulation_results]
    
    # Calculate quartile boundaries
    N_quartiles = np.percentile(N_values, [25, 50, 75])
    quartile_labels = [
        f'Q1: N ≤ {N_quartiles[0]:.0f}',
        f'Q2: {N_quartiles[0]:.0f} < N ≤ {N_quartiles[1]:.0f}',
        f'Q3: {N_quartiles[1]:.0f} < N ≤ {N_quartiles[2]:.0f}',
        f'Q4: N > {N_quartiles[2]:.0f}'
    ]
    quartile_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']  # Blue, green, orange, red
    
    # Assign each result to a quartile
    quartile_assignments = np.zeros(len(simulation_results), dtype=int)
    for i, N in enumerate(N_values):
        if N <= N_quartiles[0]:
            quartile_assignments[i] = 0
        elif N <= N_quartiles[1]:
            quartile_assignments[i] = 1
        elif N <= N_quartiles[2]:
            quartile_assignments[i] = 2
        else:
            quartile_assignments[i] = 3
    
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    axes = axes.flatten()
    
    # Function to create stacked histogram with shared bins
    def create_stacked_histogram(ax, values_list, labels, colors, title, xlabel, normalize=False):
        # Remove NaNs and combine all data to determine global bins
        all_data = []
        for values in values_list:
            valid_vals = [v for v in values if not np.isnan(v)]
            all_data.extend(valid_vals)
        
        if len(all_data) == 0:
            ax.text(0.5, 0.5, 'No valid data', horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes, fontsize=12, color='red')
            return
        
        # Create shared bins for all quartiles
        min_val = min(all_data)
        max_val = max(all_data)
        num_bins = 30
        
        # Handle special case for Rg vs N plot
        if title == 'Radius of Gyration vs. Aggregate Size':
            bins = np.logspace(np.log10(min_val), np.log10(max_val), num_bins + 1)
        else:
            bins = np.linspace(min_val, max_val, num_bins + 1)
        
        # Calculate histogram values for each quartile using the same bins
        hist_values = []
        for values in values_list:
            valid_vals = [v for v in values if not np.isnan(v)]
            if normalize and len(valid_vals) > 0:
                # Normalize by quartile size for density comparison
                counts, _ = np.histogram(valid_vals, bins=bins)
                counts = counts / len(valid_vals) * 100  # Convert to percentage
            else:
                counts, _ = np.histogram(valid_vals, bins=bins)
            hist_values.append(counts)
        
        # Create stacked bar chart
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_width = bins[1] - bins[0]
        bottom = np.zeros(len(bin_centers))
        
        for i, counts in enumerate(hist_values):
            if np.sum(counts) > 0:  # Only plot if there's data
                if title == 'Radius of Gyration vs. Aggregate Size':
                    # For log-scale Rg vs N plot
                    ax.bar(bin_centers, counts, width=bin_width*0.8, bottom=bottom, 
                           color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5, 
                           label=labels[i], log=True)
                else:
                    ax.bar(bin_centers, counts, width=bin_width*0.8, bottom=bottom, 
                           color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5, 
                           label=labels[i])
                bottom += counts
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        if normalize:
            ax.set_ylabel('Percentage (%)')
        else:
            ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    # 1. Shape Factor Distribution by N Quartile
    shape_factors_by_quartile = []
    for q in range(4):
        q_indices = [i for i, q_idx in enumerate(quartile_assignments) if q_idx == q]
        shape_factors_by_quartile.append([
            simulation_results[i]['shape_factor'] 
            for i in q_indices if not np.isnan(simulation_results[i]['shape_factor'])
        ])
    create_stacked_histogram(
        axes[0], 
        shape_factors_by_quartile,
        quartile_labels, 
        quartile_colors,
        'Shape Factor Distribution by Aggregate Size',
        'Shape Factor (sqrt(λ_max/λ_min))',
        normalize=False
    )
    
    # 2. Fractal Dimension (df_v2) Distribution by N Quartile
    df_v2_by_quartile = []
    for q in range(4):
        q_indices = [i for i, q_idx in enumerate(quartile_assignments) if q_idx == q]
        df_v2_by_quartile.append([
            simulation_results[i]['df_v2'] 
            for i in q_indices if not np.isnan(simulation_results[i]['df_v2'])
        ])
    create_stacked_histogram(
        axes[1], 
        df_v2_by_quartile,
        quartile_labels, 
        quartile_colors,
        'Fractal Dimension Distribution by Aggregate Size',
        'Mass Fractal Dimension (df)',
        normalize=False
    )
    
    # 3. Porosity Distribution by N Quartile
    porosity_by_quartile = []
    for q in range(4):
        q_indices = [i for i, q_idx in enumerate(quartile_assignments) if q_idx == q]
        porosity_by_quartile.append([
            simulation_results[i]['porosity'] 
            for i in q_indices if not np.isnan(simulation_results[i]['porosity'])
        ])
    create_stacked_histogram(
        axes[2], 
        porosity_by_quartile,
        quartile_labels, 
        quartile_colors,
        'Porosity Distribution by Aggregate Size',
        'Porosity (ϵ)',
        normalize=False
    )
    
    # 4. Radius of Gyration Distribution by N Quartile
    Rg_by_quartile = []
    for q in range(4):
        q_indices = [i for i, q_idx in enumerate(quartile_assignments) if q_idx == q]
        Rg_by_quartile.append([
            simulation_results[i]['Rg'] 
            for i in q_indices if not np.isnan(simulation_results[i]['Rg'])
        ])
    create_stacked_histogram(
        axes[3], 
        Rg_by_quartile,
        quartile_labels, 
        quartile_colors,
        'Radius of Gyration Distribution by Aggregate Size',
        'Radius of Gyration (Rg)',
        normalize=False
    )
    
    # 5. Aggregate Size (N) Distribution
    N_by_quartile = []
    for q in range(4):
        q_indices = [i for i, q_idx in enumerate(quartile_assignments) if q_idx == q]
        N_by_quartile.append([
            simulation_results[i]['parameters']['N'] 
            for i in q_indices
        ])
    create_stacked_histogram(
        axes[4], 
        N_by_quartile,
        quartile_labels, 
        quartile_colors,
        'Aggregate Size Distribution',
        'Number of Particles (N)',
        normalize=False
    )
    
    # 6. Rg vs N Scatter Plot with Quartile Coloring
    ax = axes[5]
    for q in range(4):
        q_indices = [i for i, q_idx in enumerate(quartile_assignments) if q_idx == q]
        q_N = [simulation_results[i]['parameters']['N'] for i in q_indices]
        q_Rg = [
            simulation_results[i]['Rg'] 
            for i in q_indices if not np.isnan(simulation_results[i]['Rg'])
        ]
        
        # Match lengths by filtering
        valid_pairs = [(n, r) for n, r in zip(q_N, q_Rg) if not np.isnan(r)]
        if valid_pairs:
            q_N_valid = [p[0] for p in valid_pairs]
            q_Rg_valid = [p[1] for p in valid_pairs]
            ax.scatter(q_N_valid, q_Rg_valid, 
                       color=quartile_colors[q], alpha=0.6, s=20,
                       label=quartile_labels[q])
    
    # Add power-law fit
    valid_N = [res['parameters']['N'] for res in simulation_results]
    valid_Rg = [res['Rg'] for res in simulation_results if not np.isnan(res['Rg'])]
    if len(valid_Rg) > 2:
        # Filter out NaNs
        valid_pairs = [(n, r) for n, r in zip(valid_N, valid_Rg) if not np.isnan(r)]
        if len(valid_pairs) > 2:
            fit_N = [p[0] for p in valid_pairs]
            fit_Rg = [p[1] for p in valid_pairs]
            log_N = np.log(fit_N)
            log_Rg = np.log(fit_Rg)
            coeffs = np.polyfit(log_N, log_Rg, 1)
            slope = coeffs[0]
            intercept = coeffs[1]
            fit_line_N = np.logspace(np.log10(min(fit_N)), np.log10(max(fit_N)), 100)
            fit_line_Rg = np.exp(slope * np.log(fit_line_N) + intercept)
            ax.plot(fit_line_N, fit_line_Rg, 'k--', linewidth=2, 
                    label=f'Power-law fit: Rg ∝ N^{slope:.3f}')
    
    ax.set_title('Radius of Gyration vs. Aggregate Size')
    ax.set_xlabel('Number of Particles (N)')
    ax.set_ylabel('Radius of Gyration (Rg)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')
    
    plt.suptitle(f'Distribution of Metrics for {len(simulation_results)} Simulations\n'
                 f'(Colored by Aggregate Size Quartiles, p={inactivation_probability:.3f}, '
                 f'overlap={overlap}, bias={bias_factors})',
                 fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
    # Print summary statistics
    def safe_stats(data, name):
        if len(data) == 0:
            print(f"\n{name}:")
            print(f"  Mean: N/A (No valid data)")
            print(f"  Std:  N/A (No valid data)")
            print(f"  Min:  N/A (No valid data)")
            print(f"  Max:  N/A (No valid data)")
        else:
            print(f"\n{name}:")
            print(f"  Mean: {np.mean(data):.4f}")
            print(f"  Std:  {np.std(data):.4f}")
            print(f"  Min:  {np.min(data):.4f}")
            print(f"  Max:  {np.max(data):.4f}")
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total valid simulations: {len(simulation_results)}")
    print(f"Aggregate size statistics (N):")
    print(f"  Mean: {np.mean(N_values):.1f}")
    print(f"  Std:  {np.std(N_values):.1f}")
    print(f"  Min:  {np.min(N_values)}")
    print(f"  Max:  {np.max(N_values)}")
    
    safe_stats(shape_factors_clean, "Shape Factor")
    safe_stats(df_v2_clean, "df_v2 (Structure Factor)")
    safe_stats(porosities_clean, "Porosity")
    safe_stats(Rg_clean, "Rg")
    
    # Save data if requested
    if save:
        metadata = {
            'function': 'generate_simulation_results_by_seed_PDX',
            'parameters': {
                'N_distribution': N_dist_params,
                'bias_factors': bias_factors,
                'overlap': overlap,
                'inactivation_probability': inactivation_probability,
                'primary_radius': primary_radius,
                'cell_size': cell_size,
                'num_simulations': num_simulations,
                'random_seed_base': random_seed_base
            }
        }
        # Create filename based on distribution parameters
        if N_dist_params['type'] == 'lognormal':
            if 'target_mean' in N_dist_params:
                filename = f"simulation_results_PDX_targetmean{N_dist_params['target_mean']}_cv{N_dist_params['cv']}_p{inactivation_probability}"
            else:
                filename = f"simulation_results_PDX_mean{N_dist_params['mean']}_sigma{N_dist_params['sigma']}_p{inactivation_probability}"
        else:
            filename = f"simulation_results_PDX_{N_dist_params['type']}_p{inactivation_probability}"
        
        save_simulation_results(simulation_results, filename, metadata)
    
    return simulation_results

def visualizer_subset_stats(simulation_results_list, figsize=(8, 12), bins=50):
    """
    Visualize distribution of key metrics for one or two simulation datasets.
    
    Creates 4 histograms in a single column:
    1. Shape Factor
    2. Mass Fractal Dimension (df_v2, structure factor method)
    3. Porosity
    4. Radius of Gyration (Rg)
    
    Accepts result dicts from batch generation or lists of particles lists.
    
    Parameters:
    -----------
    simulation_results_list : list of lists
        Each inner list is a dataset (list of result dicts or particles lists).
        1 dataset: Plot parent only
        2 datasets: Plot parent (alpha=0.5) and subset (alpha=1.0)
    
    figsize : tuple
        Figure size (width, height)
    
    bins : int
        Number of bins for histograms
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    
    # Validate input
    if len(simulation_results_list) == 0:
        raise ValueError("simulation_results_list must contain at least one dataset.")
    if len(simulation_results_list) > 2:
        raise ValueError("simulation_results_list can contain at most two datasets.")
    
    # Define metrics
    metrics = [
        ('shape_factor', 'Shape Factor'),
        ('df_v2', 'Mass Fractal Dimension'),
        ('porosity', 'Porosity'),
        ('Rg', 'Radius of Gyration')
    ]
    
    # Extract data from each dataset
    datasets = []
    for sim_results in simulation_results_list:
        data = {}
        for metric_key, _ in metrics:
            # Extract non-NaN values
            values = []
            for res in sim_results:
                # Handle both dict format and list format
                if isinstance(res, dict):
                    val = res.get(metric_key, np.nan)
                else:
                    # Shouldn't happen, but handle gracefully
                    val = np.nan
                
                if not np.isnan(val):
                    values.append(val)
            
            data[metric_key] = np.array(values)
        datasets.append(data)
    
    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=False)
    
    # Define colors and alpha values
    colors = ['skyblue', 'lightcoral', 'orange', 'purple']
    alphas = [0.5, 1.0]  # Parent, Subset
    
    # Plot histograms
    for i, (metric_key, metric_name) in enumerate(metrics):
        ax = axes[i]
        
        # Plot parent dataset
        parent_data = datasets[0][metric_key]
        ax.hist(parent_data, bins=bins, color=colors[i], alpha=alphas[0],
                label='Parent', edgecolor='black', linewidth=0.5)
        
        # Plot subset if provided
        if len(datasets) > 1:
            subset_data = datasets[1][metric_key]
            subset_bins = int(bins * (len(subset_data) / len(parent_data)))
            ax.hist(subset_data, bins=subset_bins, color=colors[i], alpha=alphas[1],
                    label='Subset', edgecolor='black', linewidth=0.5)
        
        # Labels and formatting
        ax.set_title(metric_name, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_xlabel('Value', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if len(datasets) > 1:
            ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    return fig, axes


#%%
# shared 
N_particles = 100
bias = (1,1,1)
overlap_par = 0.0
max_viz = 150
inactivation_probability = 0.95
primary_radius = 3.7
num_simulations = 10000
cell_size=12.0

#%% RUN SIMULATIONS (RANDOM SEEDS, CONSTANT PARAMETERS)

# sim_random = generate_simulation_results_by_seed(
#                                             N=N_particles, 
#                                             bias_factors=bias, 
#                                             overlap=overlap_par, 
#                                             max_particles_for_spheres=max_viz,
#                                             inactivation_probability=inactivation_probability, 
#                                             primary_radius=primary_radius,
#                                             cell_size=cell_size,
#                                             num_simulations=num_simulations,
#                                             save=True
#                                             )

sim_random_pdx = generate_simulation_results_by_seed_PDX(
    N_dist_params={'type': 'lognormal', 'mean': 4.0, 'sigma': 0.8},  # Default ~100 particles mean
    bias_factors=bias,
    overlap=overlap_par,
    max_particles_for_spheres=max_viz,
    inactivation_probability=0.95,
    primary_radius=primary_radius,
    cell_size=12.0,
    num_simulations=10000,
    figsize=(18, 6),
    save=True,
    random_seed_base=601)

#%% 
# p095 = load_simulation_results('simulation_results_N100_p0.950')
# p000 = load_simulation_results('simulation_results_N100_p0.000')
# p095_big = load_simulation_results('simulation_results_N10000_p0.950')
# p000_big = load_simulation_results('simulation_results_N10000_p0.000')
pdx000 = load_simulation_results('simulation_results_PDX_mean4.0_sigma0.8')
pdx095 = load_simulation_results('simulation_results_PDX_mean4.0_sigma0.8_p0.95')
#%%
pdx000sub1700 = visualize_filtered_aggregates(pdx000, 
                                     min_shape_factor=None,
                                     max_shape_factor=None,
                                     min_df_v2=2.70,
                                     max_df_v2=3.00,
                                     min_porosity=None,
                                     max_porosity=None,
                                     min_rg=9.52,
                                     max_rg=32.6,
                                     plot_type='scatter',
                                     color_by='deepskyblue',
                                     point_size_par=15,
                                     figsize=(12, 12),
                                     max_particles_for_spheres=100,
                                     base_margin_factor=0.1
                                 )

save_simulation_results(pdx000sub1700, 'pdx000sub1700_aggregates')
#%% 

pdx000sub1702 = visualize_filtered_aggregates(pdx000, 
                                     min_shape_factor=None,
                                     max_shape_factor=None,
                                     min_df_v2=2.28,
                                     max_df_v2=2.42,
                                     min_porosity=None,
                                     max_porosity=None,
                                     min_rg=22.3,
                                     max_rg=43.5,
                                     plot_type='scatter',
                                     color_by='magenta',
                                     point_size_par=15,
                                     figsize=(12, 12),
                                     max_particles_for_spheres=50,
                                     base_margin_factor=0.1
                                 )

save_simulation_results(pdx000sub1702, 'pdx000sub1702_aggregates')
#%% 

pdx095sub1701 = visualize_filtered_aggregates(pdx095, 
                                     min_shape_factor=None,
                                     max_shape_factor=None,
                                     min_df_v2=2,
                                     max_df_v2=2.15,
                                     min_porosity=None,
                                     max_porosity=None,
                                     min_rg=19.4,
                                     max_rg=36.6,
                                     plot_type='scatter',
                                     color_by='orange',
                                     point_size_par=15,
                                     figsize=(12, 12),
                                     max_particles_for_spheres=50,
                                     base_margin_factor=0.1
                                 )

save_simulation_results(pdx095sub1701, 'pdx095sub1701_aggregates')
#%%
import matplotlib.colors as mcolors
from matplotlib import colormaps
from matplotlib.colors import to_rgba
import numpy as np

def xyz_saver(
    simulation_results,
    radius = 1.0,
    xyz_type='extended',
    particle_name='ThO2',
    N_xyz=3,
    color='addition_step',
    colormap='gnuplot2',
    base_name=None
):
    """
    Export randomly selected aggregates to XYZ files.
    
    Accepts result dicts from batch generation functions or direct particles lists.
    Works with both old and new data formats. Now supports polydisperse aggregates
    with different numbers of particles.
    
    Parameters:
    -----------
    simulation_results : list
        List of result dicts from batch generation, or list of particles lists
    
    xyz_type : str
        'simple' - Just positions (x, y, z)
        'extended' - Positions + colors + active status + radius (8 columns)
    
    particle_name : str
        Element name for XYZ file (default: 'ThO2')
    
    N_xyz : int
        Number of random aggregates to export (default: 3)
    
    color : str
        'addition_step' - Rainbow gradient by particle order
        'color_name' - Uniform color (e.g., 'red', 'blue')
    
    colormap : str
        Matplotlib colormap for 'addition_step' coloring (default: 'gnuplot2')
    
    base_name : str or None
        Base name for output files. If None, prompts user.
    
    Returns:
    --------
    arrays : list
        List of exported numpy arrays
    
    Writes files:
        {base_name}_{xyz_type}XYZ_#{index}.xyz
    """
    
    # ========== INPUT VALIDATION ==========
    
    if not simulation_results:
        raise ValueError("simulation_results must not be empty")
    
    if xyz_type not in ['simple', 'extended']:
        raise ValueError(f"xyz_type must be 'simple' or 'extended', got '{xyz_type}'")
    
    if N_xyz < 1:
        raise ValueError(f"N_xyz must be >= 1, got {N_xyz}")
    
    # ========== PROMPT FOR BASE NAME ==========
    
    if base_name is None:
        base_name = input(
            "Enter base name for output files (e.g., 'simulation'): "
        ).strip()
        if not base_name:
            base_name = "Fractal_Aggregate_XYZ"
    
    # ========== SETUP COLORMAPS & COLORS ==========
    
    named_colors = mcolors.get_named_colors_mapping()
    colmap = None
    
    # Get colormap if needed
    if color == 'addition_step':
        try:
            colmap = colormaps[colormap]
        except KeyError:
            print(f"Warning: colormap '{colormap}' not found, using 'viridis'")
            colmap = colormaps['viridis']
    
    # ========== MAIN LOOP ==========
    
    arrays = []
    number_of_aggs = len(simulation_results)
    N_xyz_actual = min(N_xyz, number_of_aggs)
    
    print(f"Found {number_of_aggs} aggregates with varying sizes")
    print(f"Exporting {N_xyz_actual} random aggregates to XYZ format")
    
    # Randomly select indices
    selected_indices = np.random.choice(
        number_of_aggs,
        size=N_xyz_actual,
        replace=False
    )
    
    for idx_count, entry in enumerate(selected_indices):
        result = simulation_results[entry]
        
        # Extract particles list (handles both formats)
        if isinstance(result, dict) and 'particles' in result:
            particles_list = result['particles']
        else:
            particles_list = result
        
        # Get current aggregate size (varies per aggregate)
        current_size = len(particles_list)
        
        # Initialize arrays with current size
        add_step = np.zeros(current_size)
        active_status = np.zeros(current_size)
        radii = np.ones(current_size) * radius
        elements = np.repeat(particle_name, current_size)
        
        # ========== EXTENDED FORMAT ==========
        
        if xyz_type == 'extended':
            xyz_array = np.zeros((current_size, 8))
            
            for j in range(current_size):
                particle = particles_list[j]
                
                # Position
                xyz_array[j, :3] = particle['position']
                
                # Metadata
                add_step[j] = particle['added_step']
                active_status[j] = float(particle.get('inactive', False))
            
            # Normalize addition step to [0, 1] (if there are steps)
            if np.max(add_step) > 0:
                add_step_norm = add_step / np.max(add_step)
            else:
                add_step_norm = np.zeros(current_size)
            
            # Apply coloring
            if color == 'addition_step' and colmap is not None:
                # Rainbow gradient based on addition order
                colors_rgba = colmap(add_step_norm)
                xyz_array[:, 3:6] = colors_rgba[:, :3]  # RGB only
            elif color in named_colors:
                # Uniform color
                rgb = to_rgba(color)[:3]
                xyz_array[:, 3:6] = rgb
            else:
                # Default: use 'addition_step' if color not recognized
                print(f"Warning: color '{color}' not recognized, using default gray")
                xyz_array[:, 3:6] = [0.5, 0.5, 0.5]  # Gray
            
            # Active status and radius
            xyz_array[:, 6] = active_status
            xyz_array[:, 7] = radii
            
            arrays.append(xyz_array)
        
        # ========== SIMPLE FORMAT ==========
        
        elif xyz_type == 'simple':
            xyz_array = np.zeros((current_size, 3))
            
            for j in range(current_size):
                particle = particles_list[j]
                xyz_array[j, :3] = particle['position']
                add_step[j] = particle['added_step']
                active_status[j] = float(particle.get('inactive', False))
            
            arrays.append(xyz_array)
        
        # ========== BUILD OUTPUT ==========
        
        # Combine element names with data
        elements = elements.reshape(-1, 1)
        combined_data = np.hstack([elements, xyz_array])
        
        # Format output lines
        output_lines = []
        
        # Header: number of atoms
        output_lines.append(f"{current_size}")
        
        # Header: comment line with metadata
        if xyz_type == 'extended':
            output_lines.append(
                'Lattice="100 0.0 0.0 0.0 100 0.0 0.0 0.0 100" '
                'Properties=species:S:1:pos:R:3:color:R:3:flagged:I:1:radius:R:1 pbc="F F F"'
            )
        elif xyz_type == 'simple':
            max_step = np.max(add_step) if current_size > 0 else 0
            output_lines.append(
                f"Fractal aggregate #{entry} (N={current_size} particles, "
                f"addition_step_max={max_step:.0f})"
            )
        
        # Data rows
        for row in combined_data:
            # Format: element x y z [optional columns]
            formatted_row = [row[0]]  # Element (string)
            for val in row[1:]:
                formatted_row.append(f"{float(val):.6f}")
            output_lines.append(" ".join(formatted_row))
        
        # ========== WRITE FILE ==========
        
        filename = f"{base_name}_{xyz_type}XYZ_#{entry}.xyz"
        
        try:
            with open(filename, 'w') as f:
                f.write("\n".join(output_lines))
            print(f"✓ Exported: {filename} (N={current_size} particles)")
        except IOError as e:
            print(f"✗ Error writing {filename}: {e}")
    
    print(f"\nExported {len(arrays)} aggregates successfully")
    return arrays

#%%
rad_export = xyz_saver(
                        pdx095sub1701, radius=3.7,
                        xyz_type='extended',
                        particle_name='ThO2',
                        N_xyz=10,
                        color='deepskyblue',
                        colormap='gnuplot2',
                        base_name=None)


#%%
"""
Fixed agglomerate generation and XYZ saving functions.

generate_agglomerate() - Creates hierarchical agglomerates from aggregates
xyz_saver_agglo() - Saves agglomerates to XYZ format with coloring options
"""

import numpy as np
from pathlib import Path
from matplotlib import colormaps, colors as mcolors
from matplotlib.colors import to_rgba


def calculate_aggregate_properties(particles):
    """
    Calculate Rg and center of mass for an aggregate.
    
    Parameters:
    -----------
    particles : list of dicts
        Each dict must have 'position' key with numpy array [x, y, z]
    
    Returns:
    --------
    Rg : float
        Radius of gyration
    center : numpy array
        Center of mass
    """
    positions = np.array([p['position'] for p in particles])
    center = np.mean(positions, axis=0)
    Rg = np.sqrt(np.mean(np.sum((positions - center)**2, axis=1)))
    return Rg, center


class LinkedCellGrid:
    """Spatial grid for fast collision detection."""
    
    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.cells = {}
    
    def add_particle(self, particle_idx, position):
        """Add particle to grid."""
        cell_key = tuple(np.floor(position / self.cell_size).astype(int))
        if cell_key not in self.cells:
            self.cells[cell_key] = []
        self.cells[cell_key].append(particle_idx)
    
    def get_neighbors(self, position, radius=1.0):
        """Get particle indices near position within radius."""
        cell_key = tuple(np.floor(position / self.cell_size).astype(int))
        neighbors = []
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    neighbor_key = (cell_key[0]+dx, cell_key[1]+dy, cell_key[2]+dz)
                    if neighbor_key in self.cells:
                        neighbors.extend(self.cells[neighbor_key])
        
        return neighbors


def generate_agglomerate(
    aggregates_data,
    N_sub=None,
    contact_scaling_factor=1.0,
    macro_cell_size_beta=2.5,
    random_seed=42,
    visualize=False,
    max_particles_for_spheres=200
):
    """
    Generate an agglomerate by placing aggregates using Porous Eden algorithm.
    
    Parameters:
    -----------
    aggregates_data : list
        List of aggregates (either particle dicts or result dicts)
    
    N_sub : int or None
        If int: randomly select N_sub aggregates WITH REPLACEMENT from input
        If None: use all aggregates (default)
    
    contact_scaling_factor : float
        Scaling for contact distance between aggregates (default: 1.0)
    
    macro_cell_size_beta : float
        Cell size = beta * mean_Rg for macro-scale grid (default: 2.5)
    
    random_seed : int
        Random seed for reproducibility (default: 42)
    
    visualize : bool
        Whether to show 3D plot (default: False)
    
    max_particles_for_spheres : int
        Max particles to render as spheres (default: 200)
    
    Returns:
    --------
    dict with keys:
        'particles' : list of particle dicts with tracking info
        'macro_level' : dict with macro aggregate info
        'parameters' : dict of generation parameters
        'metadata' : dict with statistics
    """
    
    np.random.seed(random_seed)
    
    # ========== EXTRACT AGGREGATES ==========
    aggregates = []
    for agg_data in aggregates_data:
        if isinstance(agg_data, dict) and 'particles' in agg_data:
            aggregates.append(agg_data['particles'])
        elif isinstance(agg_data, list):
            aggregates.append(agg_data)
        else:
            aggregates.append([agg_data])
    
    original_num = len(aggregates)
    
    # ========== N_sub SUBSAMPLING ==========
    if N_sub is not None:
        selected_indices = np.random.choice(original_num, size=N_sub, replace=True)
        aggregates = [aggregates[i] for i in selected_indices]
        print(f"N_sub: selected {N_sub} aggregates from {original_num} (with replacement)")
    
    # ========== CALCULATE AGGREGATE PROPERTIES ==========
    agg_properties = []
    for agg in aggregates:
        Rg, center = calculate_aggregate_properties(agg)
        agg_properties.append({'Rg': Rg, 'center': center})
    
    mean_Rg = np.mean([p['Rg'] for p in agg_properties])
    
    # ========== MACRO-SCALE PLACEMENT (Porous Eden) ==========
    macro_positions = []  # Centers of aggregates in macro space
    macro_cell_size = macro_cell_size_beta * mean_Rg
    spatial_grid = LinkedCellGrid(macro_cell_size)
    
    # Place first aggregate at origin
    macro_positions.append(np.array([0.0, 0.0, 0.0]))
    spatial_grid.add_particle(0, macro_positions[0])
    
    # Place remaining aggregates
    active_indices = [0]
    step = 1
    
    while len(macro_positions) < len(aggregates):
        if len(active_indices) == 0:
            # Start new cluster
            active_indices = [len(macro_positions) - 1]
        
        # Pick random active aggregate
        selected = active_indices[np.random.randint(len(active_indices))]
        
        # Random direction
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)
        
        # Place new aggregate
        sum_Rg = agg_properties[selected]['Rg'] + agg_properties[len(macro_positions)]['Rg']
        distance = contact_scaling_factor * sum_Rg * 2
        
        new_pos = macro_positions[selected] + direction * distance
        
        # Check for collisions
        collision = False
        neighbors = spatial_grid.get_neighbors(new_pos, radius=sum_Rg * 2)
        
        for neighbor_idx in neighbors:
            dist_to_neighbor = np.linalg.norm(new_pos - macro_positions[neighbor_idx])
            min_dist = contact_scaling_factor * (
                agg_properties[len(macro_positions)]['Rg'] + 
                agg_properties[neighbor_idx]['Rg']
            ) * 2
            
            if dist_to_neighbor < min_dist:
                collision = True
                break
        
        if not collision:
            macro_positions.append(new_pos)
            spatial_grid.add_particle(len(macro_positions) - 1, new_pos)
            active_indices.append(len(macro_positions) - 1)
            step += 1
        
        # Randomly deactivate
        if np.random.random() < 0.1:
            if len(active_indices) > 0:
                active_indices.pop(np.random.randint(len(active_indices)))
    
    # ========== TRANSFORM COORDINATES TO GLOBAL SYSTEM ==========
    output_particles = []
    for agg_idx, (agg, macro_pos) in enumerate(zip(aggregates, macro_positions)):
        Rg, local_center = agg_properties[agg_idx]['Rg'], agg_properties[agg_idx]['center']
        
        for particle in agg:
            # Local position relative to aggregate center
            local_pos = particle['position'] - local_center
            
            # Global position
            global_pos = local_pos + macro_pos
            
            output_particles.append({
                'position': global_pos,
                'aggregate_id': agg_idx,
                'added_step': step,
                'parent_aggregate_Rg': Rg,
                'inactive': particle.get('inactive', False),
                **{k: v for k, v in particle.items() 
                   if k not in ['position', 'inactive']}
            })
    
    # ========== BUILD RETURN DICT ==========
    result = {
        'particles': output_particles,
        'macro_level': {
            'positions': macro_positions,
            'Rg_values': [p['Rg'] for p in agg_properties],
            'num_aggregates': len(aggregates)
        },
        'parameters': {
            'contact_scaling_factor': contact_scaling_factor,
            'macro_cell_size_beta': macro_cell_size_beta,
            'mean_Rg': mean_Rg,
            'random_seed': random_seed,
            'N_sub': N_sub,
            'original_num_aggregates': original_num
        },
        'metadata': {
            'total_primary_particles': len(output_particles),
            'num_aggregates': len(aggregates)
        }
    }
    
    # ========== VISUALIZATION ==========
    if visualize:
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            
            positions = np.array([p['position'] for p in output_particles])
            agg_ids = np.array([p['aggregate_id'] for p in output_particles])
            
            scatter = ax.scatter(
                positions[:, 0], positions[:, 1], positions[:, 2],
                c=agg_ids, cmap='tab20', s=20, alpha=0.6
            )
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Agglomerate ({len(aggregates)} aggregates, '
                        f'{len(output_particles)} particles)')
            plt.colorbar(scatter, ax=ax, label='Aggregate ID')
            plt.show()
        except ImportError:
            print("Matplotlib not available for visualization")
    
    return result

#%%
agglomerate_1700 = generate_agglomerate(pdx000sub1700, 
                                        150, 0.5, 2.5, 601, False)

agglomerate_1701 = generate_agglomerate(pdx095sub1701, 
                                        150, 0.5, 2.5, 601, False)

agglomerate_1702 = generate_agglomerate(pdx000sub1702, 
                                        150, 0.5, 2.5, 601, False)

#%%
def xyz_saver_agglo(
    agglomerate,
    radius = 1.0,
    xyz_type='extended',
    particle_name='X',
    N_xyz=3,
    color='aggregate_id',
    colormap='gnuplot2',
    base_name=None
):
    """
    Export agglomerate(s) to XYZ files with coloring options.
    
    Analogous to xyz_saver(), works with generate_agglomerate() output.
    
    Parameters:
    -----------
    agglomerate : dict or list
        Single agglomerate dict from generate_agglomerate(), or
        list of agglomerate dicts
    
    xyz_type : str
        'simple' - Just positions (x, y, z)
        'extended' - Positions + colors + inactive status + radius (8 columns)
    
    particle_name : str
        Element name for XYZ file (default: 'X')
    
    N_xyz : int
        Number of random agglomerates to export from list (default: 3)
    
    color : str
        'aggregate_id' - Color by which aggregate (default)
        'added_step' - Color by assembly order
        'single' or other - Uniform color
    
    colormap : str
        Matplotlib colormap for gradient coloring (default: 'gnuplot2')
    
    base_name : str or None
        Base name for output files. If None, prompts user.
    
    Returns:
    --------
    arrays : list
        List of exported numpy arrays
    
    Writes files:
        {base_name}_{xyz_type}XYZ_#{index}.xyz
    """
    
    # ========== INPUT VALIDATION ==========
    if base_name is None:
        base_name = input(
            "Enter base name for output files (e.g., 'agglom'): "
        ).strip()
        if not base_name:
            base_name = "Agglomerate_XYZ"
    
    if xyz_type not in ['simple', 'extended']:
        raise ValueError(f"xyz_type must be 'simple' or 'extended', got '{xyz_type}'")
    
    # Handle single agglomerate or list
    if isinstance(agglomerate, dict) and 'particles' in agglomerate:
        agglomerates = [agglomerate]
    elif isinstance(agglomerate, list):
        agglomerates = agglomerate
    else:
        raise TypeError("agglomerate must be dict or list of dicts")
    
    # ========== SETUP COLORS ==========
    named_colors = mcolors.get_named_colors_mapping()
    
    if color == 'aggregate_id':
        colormap_obj = colormaps[colormap] if colormap in colormaps else colormaps['viridis']
    elif color == 'added_step':
        colormap_obj = colormaps[colormap] if colormap in colormaps else colormaps['viridis']
    else:
        colormap_obj = None
    
    # ========== PROCESS AGGLOMERATES ==========
    arrays = []
    N_xyz_actual = min(N_xyz, len(agglomerates))
    
    print(f"Found {len(agglomerates)} agglomerate(s)")
    print(f"Exporting {N_xyz_actual} to XYZ format")
    
    selected_indices = np.random.choice(
        len(agglomerates), size=N_xyz_actual, replace=False
    )
    
    for idx_num, agg_idx in enumerate(selected_indices):
        agglom = agglomerates[agg_idx]
        particles_list = agglom['particles']
        N_particles = len(particles_list)
        
        # ========== BUILD XYZ ARRAY ==========
        if xyz_type == 'extended':
            xyz_array = np.zeros((N_particles, 8))
        else:
            xyz_array = np.zeros((N_particles, 3))
        
        # ========== EXTRACT COLORING DATA ==========
        agg_ids = np.zeros(N_particles)
        added_steps = np.zeros(N_particles)
        inactive_status = np.zeros(N_particles)
        
        for j, particle in enumerate(particles_list):
            xyz_array[j, :3] = particle['position']
            agg_ids[j] = particle.get('aggregate_id', 0)
            added_steps[j] = particle.get('added_step', 0)
            inactive_status[j] = float(particle.get('inactive', False))
        
        # ========== APPLY COLORING ==========
        if xyz_type == 'extended':
            if color == 'aggregate_id':
                # Normalize by max aggregate ID
                max_agg_id = np.max(agg_ids) + 1e-10
                agg_ids_norm = agg_ids / max_agg_id
                colors_rgba = colormap_obj(agg_ids_norm)
                xyz_array[:, 3:6] = colors_rgba[:, :3]
            
            elif color == 'added_step':
                # Normalize by max added_step
                max_step = np.max(added_steps) + 1e-10
                added_steps_norm = added_steps / max_step
                colors_rgba = colormap_obj(added_steps_norm)
                xyz_array[:, 3:6] = colors_rgba[:, :3]
            
            else:
                # Uniform color
                if color in named_colors:
                    rgb = to_rgba(color)[:3]
                else:
                    rgb = to_rgba('gray')[:3]
                xyz_array[:, 3:6] = rgb
            
            xyz_array[:, 6] = inactive_status
            xyz_array[:, 7] = 1.0 * radius # radius
        
        arrays.append(xyz_array)
        
        # ========== BUILD OUTPUT ==========
        elements = np.repeat(particle_name, N_particles).reshape(-1, 1)
        combined_data = np.hstack([elements, xyz_array])
        
        output_lines = []
        output_lines.append(f"{N_particles}")
        
        if xyz_type == 'extended':
            output_lines.append(
                'Lattice="100 0.0 0.0 0.0 100 0.0 0.0 0.0 100" '
                'Properties=species:S:1:pos:R:3:color:R:3:flagged:I:1:radius:R:1 pbc="F F F"'
            )
        else:
            output_lines.append(
                f"Agglomerate #{agg_idx} (N={N_particles} particles, "
                f"color_by={color})"
            )
        
        for row in combined_data:
            formatted_row = [row[0]]
            for val in row[1:]:
                formatted_row.append(f"{float(val):.6f}")
            output_lines.append(" ".join(formatted_row))
        
        # ========== WRITE FILE ==========
        filename = f"{base_name}_{xyz_type}XYZ_#{agg_idx}.xyz"
        
        try:
            with open(filename, 'w') as f:
                f.write("\n".join(output_lines))
            print(f"✓ Exported: {filename} (color_by={color})")
        except IOError as e:
            print(f"✗ Error writing {filename}: {e}")
    
    print(f"\nExported {len(arrays)} agglomerate(s) successfully")
    return arrays
#%%
xyz_saver_agglo(
    agglomerate_1702,
    radius = 3.7,
    xyz_type='extended',
    particle_name='ThO2',
    N_xyz=1,
    color='aggregate_id',
    colormap='Dark2',
    base_name=None
)
#%%
for i in range(10):
    seed = np.random.randint(0, 10**9)
    print(seed)
    intermediate = generate_agglomerate(pdx000sub1700, 
                                        150, 0.5, 2.5, seed, False)
    xyz_saver_agglo(
        intermediate,
        radius = 3.7,
        xyz_type='simple',
        particle_name='ThO2',
        N_xyz=1,
        color='aggregate_id',
        colormap='Dark2',
        base_name=f'agglomerate_comp_1700_{i}'
    )
    
#%%
import os
import numpy as np
import glob
from pathlib import Path

def fuzz_xyz_3d(
    file_path,
    sigma=0.1,
    file_pattern="*.xyz",
    random_seed=None
):
    """
    Add random thermal noise to XYZ coordinate files.
    
    Reads XYZ files (simple or extended format), adds Gaussian noise to x, y, z coordinates,
    and exports modified files with '_fuzz{sigma}' appended to filenames.
    
    Parameters:
    -----------
    file_path : str or Path
        Path to a specific XYZ file or directory containing XYZ files
    sigma : float
        Standard deviation of Gaussian noise to add to coordinates (default: 0.1)
    file_pattern : str
        Glob pattern for file selection when file_path is a directory (default: "*.xyz")
    random_seed : int or None
        Random seed for reproducibility (default: None)
    
    Returns:
    --------
    list of str
        List of paths to the generated fuzzed files
    
    Notes:
    ------
    - Preserves exact file format (simple/extended), headers, and non-coordinate data
    - Output files are named: {original_name}_fuzz{sigma:.3f}.xyz
    - Works with both simple XYZ format (element + x,y,z) and extended format (with colors, flags, etc.)
    - Coordinates are modified as: x_new ~ N(x_original, sigma)
    """
    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
    
    file_path = Path(file_path)
    generated_files = []
    
    # Determine if file_path is a file or directory
    if file_path.is_file():
        xyz_files = [file_path]
    elif file_path.is_dir():
        xyz_files = list(file_path.glob(file_pattern))
        if not xyz_files:
            raise ValueError(f"No files matching pattern '{file_pattern}' found in directory: {file_path}")
    else:
        raise ValueError(f"Path does not exist or is neither file nor directory: {file_path}")
    
    print(f"Processing {len(xyz_files)} XYZ file(s) with sigma={sigma}...")
    
    for xyz_file in xyz_files:
        try:
            # Read the original file
            with open(xyz_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < 3:
                print(f"Skipping {xyz_file.name}: file too short to be valid XYZ format")
                continue
            
            # Parse header
            num_atoms = int(lines[0].strip())
            comment_line = lines[1].strip()
            
            # Determine format type from comment line or number of columns
            xyz_type = 'simple'
            if 'Properties=species' in comment_line or 'color' in comment_line.lower():
                xyz_type = 'extended'
            
            # Parse data rows
            data_lines = lines[2:]
            if len(data_lines) < num_atoms:
                print(f"Warning: {xyz_file.name} has {len(data_lines)} data lines but header claims {num_atoms} atoms. Using actual count.")
                num_atoms = len(data_lines)
            
            # Extract data
            elements = []
            coords = []
            extra_data = []
            
            for i in range(num_atoms):
                parts = data_lines[i].strip().split()
                if len(parts) < 4:
                    print(f"Skipping invalid line {i+3} in {xyz_file.name}: '{data_lines[i].strip()}'")
                    continue
                
                # First column is element name
                elements.append(parts[0])
                
                # Next three columns are x, y, z coordinates
                try:
                    x, y, z = map(float, parts[1:4])
                    coords.append([x, y, z])
                except ValueError:
                    print(f"Warning: could not parse coordinates in line {i+3} of {xyz_file.name}. Using zeros.")
                    coords.append([0.0, 0.0, 0.0])
                
                # Any remaining columns are preserved as-is
                if len(parts) > 4:
                    extra_data.append(parts[4:])
                else:
                    extra_data.append([])
            
            coords = np.array(coords)
            
            # Apply Gaussian noise to coordinates
            noise = np.random.normal(loc=0.0, scale=sigma, size=coords.shape)
            fuzzed_coords = coords + noise
            
            # Build output lines with fuzzed coordinates
            output_lines = [
                str(num_atoms) + "\n",
                comment_line + "\n"
            ]
            
            for i in range(len(elements)):
                line_parts = [elements[i]]
                # Add fuzzed coordinates
                line_parts.extend([f"{fuzzed_coords[i,0]:.6f}", f"{fuzzed_coords[i,1]:.6f}", f"{fuzzed_coords[i,2]:.6f}"])
                # Add any extra data unchanged
                if extra_data[i]:
                    line_parts.extend(extra_data[i])
                output_lines.append(" ".join(line_parts) + "\n")
            
            # Create output filename with sigma suffix
            sigma_str = f"{sigma:.3f}".replace(".", "_")
            output_filename = xyz_file.stem + f"_fuzz{sigma_str}" + xyz_file.suffix
            output_path = xyz_file.parent / output_filename
            
            # Write the fuzzed file
            with open(output_path, 'w') as f:
                f.writelines(output_lines)
            
            generated_files.append(str(output_path))
            print(f"✓ Created: {output_path.name} (sigma={sigma})")
            
        except Exception as e:
            print(f"✗ Error processing {xyz_file.name}: {str(e)}")
            continue
    
    print(f"\nSuccessfully generated {len(generated_files)} fuzzed XYZ file(s)")
    return generated_files

# Fuzz all XYZ files in a directory
fuzzed_files = fuzz_xyz_3d(".", sigma=2, random_seed=601)