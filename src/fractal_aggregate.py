import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # Or 'Qt5Agg' if you have Qt installed
import matplotlib.pyplot as plt
plt.rcParams['figure.autolayout'] = True
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from tqdm import tqdm
from scipy.spatial.distance import pdist
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull

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
        print(f"  - Overlap: {overlap} -> center-to-center distance = {2.0 * (1 - overlap):.3f}")
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
        Rg = np.sqrt(sum_sq_distances) / N
        
        return Rg
    
    def calculate_structure_factor(self, particles, q_min=0.01, q_max=10.0, n_q=100, 
                                  R_particle=1.0, fit_range_factor=2.0, visualize=False):
        """
        Calculate the structure factor S(q) for a 3D particle aggregate as defined in Guesnet et al. (2019).
        
        Parameters:
        - particles: list of dictionaries with 'position' keys (numpy arrays of 3D coordinates)
        - q_min: float, minimum q value for calculation
        - q_max: float, maximum q value for calculation
        - n_q: int, number of q points in log space
        - R_particle: float, radius of individual particles (default=1.0)
        - fit_range_factor: float, multiplier for Rg to define fitting range: [1/Rg, fit_range_factor / R_particle]
        - visualize: bool, whether to display the plot
        
        Returns:
        - q_array: array of q values
        - S_array: array of S(q) values
        - Rg: float, radius of gyration of the aggregate
        - alpha: float, fitted slope of ln(S(q)) vs ln(q)
        - alpha_err: float, standard error of the fit
        """
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
        q_fit_max = min(1.0 / R_particle, q_max) * fit_range_factor
        
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
        
        # Step 7: Plot if requested
        if visualize:
            fig, ax = plt.subplots(figsize=(5, 4))
            
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
        
        return q_array, S_array, Rg, alpha, alpha_err
    def calculate_and_plot_running_rg(self, particles, figsize=(5, 4), title=None, fit_start_N=5, fit_end_N=None, visualize=False):
        """
        Calculate the running radius of gyration (Rg) and optionally plot it against the number of particles N
        on a log-log scale, following the methodology in Guesnet et al. (2019).
        
        The mass fractal dimension df is estimated from the slope of the linear region in this plot:
        Rg ∝ N^(1/df)
        
        Parameters:
        - particles: list of dictionaries returned by generate_aggregate()
                    Must contain 'position' and 'added_step' keys.
        - figsize: tuple, figure size. Default (5, 4)
        - title: str, optional custom title for the plot
        - fit_start_N: int, first N value to include in the linear fit. Default 5.
        - fit_end_N: int or None, last N value to include in the linear fit. If None, uses all data up to N_total.
        - visualize: bool, whether to display the plot. Default False.
        
        Returns:
        - N_list: list of integers, number of particles at each step (from 1 to N_total)
        - Rg_list: list of floats, radius of gyration at each step
        - df: float, estimated mass fractal dimension from linear fit
        - df_err: float, standard error of the df estimate from fit uncertainty
        """
        # Sort particles by their addition step to ensure chronological order
        sorted_particles = sorted(particles, key=lambda p: p['added_step'])
        
        # Extract positions in order of addition
        positions = [p['position'] for p in sorted_particles]
        N_total = len(positions)
        
        # Initialize lists to store running N and Rg
        N_list = list(range(1, N_total + 1))
        Rg_list = []
        
        # Compute Rg incrementally using efficient vectorized operations
        print("Computing running radius of gyration (Rg)...")
        for n in range(1, N_total + 1):
            if n == 1:
                Rg = 0.0
            else:
                # Get positions of first n particles
                current_positions = np.array(positions[:n])
                
                # Calculate all pairwise distances efficiently using pdist
                # pdist returns condensed form: distances between pairs (i,j) where i < j
                pairwise_distances = pdist(current_positions, metric='euclidean')
                # Square all distances
                sq_distances = pairwise_distances ** 2
                # Sum all squared distances
                sum_sq_distances = np.sum(sq_distances)
                # Calculate Rg according to Eq. (2): Rg = sqrt( (1/N) * sum(r_ij^2) )
                Rg = np.sqrt(sum_sq_distances) / n
            
            Rg_list.append(Rg)
        
        # Convert to numpy arrays for fitting
        N_array = np.array(N_list)
        Rg_array = np.array(Rg_list)
        
        # Define the range for fitting (start from fit_start_N, end at fit_end_N or N_total)
        start_idx = max(fit_start_N - 1, 0)  # Convert to zero-based index
        end_idx = min(fit_end_N, N_total) if fit_end_N is not None else N_total
        end_idx -= 1  # Convert to zero-based index
        
        if start_idx >= end_idx:
            raise ValueError(f"fit_start_N ({fit_start_N}) must be less than fit_end_N ({fit_end_N})")
        
        # Select data points for fitting
        N_fit = N_array[start_idx:end_idx]
        Rg_fit = Rg_array[start_idx:end_idx]
        
        # Filter out any zero or negative values to avoid log(0)
        valid_mask = (N_fit > 0) & (Rg_fit > 0)
        N_fit = N_fit[valid_mask]
        Rg_fit = Rg_fit[valid_mask]
        
        if len(N_fit) < 3:
            raise ValueError("Not enough valid data points for fitting after filtering.")
        
        # Log-transform the data for linear regression
        log_N = np.log(N_fit)
        log_Rg = np.log(Rg_fit)
        
        # Define linear model: log(Rg) = m * log(N) + c
        def linear_model(x, m, c):
            return m * x + c
        
        # Fit the model using scipy.optimize.curve_fit
        popt, pcov = curve_fit(linear_model, log_N, log_Rg)
        m, c = popt
        m_err, c_err = np.sqrt(np.diag(pcov))
        
        # Mass fractal dimension df = 1/m since Rg ∝ N^(1/df) => log(Rg) = (1/df) * log(N) + const
        df = 1.0 / m
        df_err = m_err / (m**2)  # Error propagation: d(df)/dm = -1/m^2
        
        # Only create plot if visualize=True
        if visualize:
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot full running Rg
            ax.loglog(N_list, Rg_list, 'b-', linewidth=1.0, alpha=0.7, label='Running Rg')
            
            # Highlight the fitted region
            ax.loglog(N_fit, Rg_fit, 'ro', markersize=4, label=f'Fit Region (N={fit_start_N}-{end_idx+1})')
            
            # Plot the fitted line
            log_N_full = np.log(N_array[start_idx:])
            log_Rg_pred = linear_model(log_N_full, m, c)
            Rg_pred = np.exp(log_Rg_pred)
            ax.loglog(N_array[start_idx:], Rg_pred, 'r--', linewidth=2, 
                    label=f'Fit: df = {df:.3f} ± {df_err:.3f}')
            
            # Add labels and title
            ax.set_xlabel('Number of Particles (N)', fontsize=10)
            ax.set_ylabel('Radius of Gyration (Rg)', fontsize=10)
            if title is None:
                title = f'Running Rg vs N\nFinal N={N_total}, Final Rg={Rg_list[-1]:.3f}, df={df:.3f}±{df_err:.3f}'
            ax.set_title(title, fontsize=12)
            
            # Enable grid for better readability
            ax.grid(True, which="both", ls="-", alpha=0.2)
            
            # Set limits to make the plot look cleaner
            ax.set_xlim(left=1)
            ax.set_ylim(bottom=1e-3)  # Avoid zero on log scale
            
            # Add legend
            ax.legend(loc='lower right', fontsize=9)
            
            # Show plot
            plt.tight_layout()
            plt.show()
        
        print(f"\nFractal Dimension Analysis (Running Rg):")
        print(f"  - Fitting range: N = {fit_start_N} to {end_idx+1}")
        print(f"  - Slope (m) of log(Rg) vs log(N): {m:.5f} ± {m_err:.5f}")
        print(f"  - Estimated mass fractal dimension df = 1/m = {df:.5f} ± {df_err:.5f}")
        
        return N_list, Rg_list, df, df_err

    def visualize_aggregate(self, particles, max_particles_for_spheres=200, figsize=(5, 4)):
        """
        Visualize a fractal aggregate in 3D.
        
        Parameters:
        - particles: list of particle dictionaries
        - max_particles_for_spheres: int, max particles to render with spheres
        - figsize: tuple, figure size (smaller default: 5, 4)
        
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
    def calculate_porosity(self, particles, particle_radius=1.0):
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
        - particles: list of dictionaries returned by generate_aggregate()
                    Must contain 'position' key with numpy array of 3D coordinates.
        - particle_radius: float, radius of each spherical particle. Default = 1.0.
        
        Returns:
        - porosity: float, the calculated porosity (between 0 and 1)
        - convex_hull_volume: float, volume of the convex hull enclosing all particles
        - total_particle_volume: float, total volume occupied by all particles
        """
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

    def run_parameter_sweep_p(self, N=100, p_range=(0.0, 0.95), p_steps=10, samples_per_p=10, 
                         bias_factors=(1, 1, 1), overlap=0.0, max_particles_for_spheres=200):
        """
        Run a parameter sweep over inactivation probability p.
        
        Generates multiple aggregates for each value of p in the specified range,
        then calculates and stores the average metrics for each p value.
        
        Parameters:
        - N: int, number of particles per aggregate
        - p_range: tuple (min_p, max_p), range of inactivation probabilities to sweep
        - p_steps: int, number of p values to sample in the range
        - samples_per_p: int, number of aggregates to generate per p value
        - bias_factors: tuple, growth bias factors (default: isotropic)
        - overlap: float, particle overlap parameter
        - max_particles_for_spheres: int, max particles to render with spheres for visualization
        
        Returns:
        - results: dict containing arrays of p values and corresponding metrics
        """
        # Generate p values
        p_values = np.linspace(p_range[0], p_range[1], p_steps)
        
        # Initialize result arrays
        shape_factors = []
        df_v1_values = []
        df_v2_values = []
        porosities = []
        Rg_values = []
        
        print(f"Running parameter sweep over p values: {p_values}")
        print(f"Generating {samples_per_p} aggregates per p value...")
        
        for p in tqdm(p_values, desc="Parameter sweep (p)"):
            # Temporary storage for this p value
            p_shape_factors = []
            p_df_v1_values = []
            p_df_v2_values = []
            p_porosities = []
            p_Rg_values = []
            
            for _ in range(samples_per_p):
                # Generate aggregate
                particles = self.generate_aggregate(
                    N=N,
                    bias_factors=bias_factors,
                    overlap=overlap,
                    inactivation_probability=p,
                    max_particles_for_spheres=max_particles_for_spheres,
                    visualize=False
                )
                
                # Calculate metrics
                sf = self.calculate_shape_factor(particles)
                df_v1 = self.calculate_and_plot_running_rg(particles, visualize=False)[2]
                q_array, S_array, Rg, alpha, alpha_err = self.calculate_structure_factor(particles, visualize=False)
                porosity = self.calculate_porosity(particles)[0]
                
                # Store results
                p_shape_factors.append(sf)
                p_df_v1_values.append(df_v1)
                p_df_v2_values.append(alpha)
                p_porosities.append(porosity)
                p_Rg_values.append(Rg)
            
            # Store average metrics for this p value
            shape_factors.append(np.mean(p_shape_factors))
            df_v1_values.append(np.mean(p_df_v1_values))
            df_v2_values.append(np.mean(p_df_v2_values))
            porosities.append(np.mean(p_porosities))
            Rg_values.append(np.mean(p_Rg_values))
        
        # Store results in the class
        results = {
            'p_values': p_values,
            'shape_factors': np.array(shape_factors),
            'df_v1_values': np.array(df_v1_values),
            'df_v2_values': np.array(df_v2_values),
            'porosities': np.array(porosities),
            'Rg_values': np.array(Rg_values),
            'parameters': {
                'N': N,
                'bias_factors': bias_factors,
                'overlap': overlap,
                'samples_per_p': samples_per_p
            }
        }
        
        self.aggregates.append({
            'type': 'parameter_sweep_p',
            'results': results
        })
        
        print("\nParameter sweep complete!")
        return results

    def run_parameter_sweep_N(self, N_range=(10, 1000), N_steps=10, samples_per_N=10, 
                            p=0.0, bias_factors=(1, 1, 1), overlap=0.0, max_particles_for_spheres=200):
        """
        Run a parameter sweep over number of particles N.
        
        Generates multiple aggregates for each value of N in the specified range,
        then calculates and stores the average metrics for each N value.
        
        Parameters:
        - N_range: tuple (min_N, max_N), range of particle counts to sweep
        - N_steps: int, number of N values to sample in the range
        - samples_per_N: int, number of aggregates to generate per N value
        - p: float, inactivation probability (fixed for this sweep)
        - bias_factors: tuple, growth bias factors (default: isotropic)
        - overlap: float, particle overlap parameter
        - max_particles_for_spheres: int, max particles to render with spheres for visualization
        
        Returns:
        - results: dict containing arrays of N values and corresponding metrics
        """
        # Generate N values (logarithmic spacing often makes more sense for particle counts)
        N_values = np.round(np.logspace(np.log10(N_range[0]), np.log10(N_range[1]), N_steps)).astype(int)
        
        # Initialize result arrays
        shape_factors = []
        df_v1_values = []
        df_v2_values = []
        porosities = []
        Rg_values = []
        
        print(f"Running parameter sweep over N values: {N_values}")
        print(f"Generating {samples_per_N} aggregates per N value...")
        
        for N in tqdm(N_values, desc="Parameter sweep (N)"):
            # Temporary storage for this N value
            N_shape_factors = []
            N_df_v1_values = []
            N_df_v2_values = []
            N_porosities = []
            N_Rg_values = []
            
            for _ in range(samples_per_N):
                # Generate aggregate
                particles = self.generate_aggregate(
                    N=int(N),
                    bias_factors=bias_factors,
                    overlap=overlap,
                    inactivation_probability=p,
                    max_particles_for_spheres=max_particles_for_spheres,
                    visualize=False
                )
                
                # Calculate metrics
                sf = self.calculate_shape_factor(particles)
                df_v1 = self.calculate_and_plot_running_rg(particles, visualize=False)[2]
                q_array, S_array, Rg, alpha, alpha_err = self.calculate_structure_factor(particles, visualize=False)
                porosity = self.calculate_porosity(particles)[0]
                
                # Store results
                N_shape_factors.append(sf)
                N_df_v1_values.append(df_v1)
                N_df_v2_values.append(alpha)
                N_porosities.append(porosity)
                N_Rg_values.append(Rg)
            
            # Store average metrics for this N value
            shape_factors.append(np.mean(N_shape_factors))
            df_v1_values.append(np.mean(N_df_v1_values))
            df_v2_values.append(np.mean(N_df_v2_values))
            porosities.append(np.mean(N_porosities))
            Rg_values.append(np.mean(N_Rg_values))
        
        # Store results in the class
        results = {
            'N_values': N_values,
            'shape_factors': np.array(shape_factors),
            'df_v1_values': np.array(df_v1_values),
            'df_v2_values': np.array(df_v2_values),
            'porosities': np.array(porosities),
            'Rg_values': np.array(Rg_values),
            'parameters': {
                'p': p,
                'bias_factors': bias_factors,
                'overlap': overlap,
                'samples_per_N': samples_per_N
            }
        }
        
        self.aggregates.append({
            'type': 'parameter_sweep_N',
            'results': results
        })
        
        print("\nParameter sweep complete!")
        return results