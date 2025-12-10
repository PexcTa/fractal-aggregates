import numpy as np
from scipy.spatial.distance import euclidean
# Add these imports at the top of fractal_generator.py
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull

class LinkedCellGrid:
    def __init__(self, cell_size=8.0):
        self.cell_size = cell_size
        self.cells = {}
        self.particle_to_cell = {}
        self.particle_positions = {}
    
    def position_to_cell_coords(self, position):
        return tuple(np.floor(position / self.cell_size).astype(int))
    
    def add_particle(self, particle_idx, position):
        cell_coords = self.position_to_cell_coords(position)
        if cell_coords not in self.cells:
            self.cells[cell_coords] = []
        self.cells[cell_coords].append(particle_idx)
        self.particle_to_cell[particle_idx] = cell_coords
        self.particle_positions[particle_idx] = position.copy()
    
    def get_nearby_particle_indices(self, position, search_radius=1):
        cell_coords = self.position_to_cell_coords(position)
        nearby_indices = []
        for di in range(-search_radius, search_radius + 1):
            for dj in range(-search_radius, search_radius + 1):
                for dk in range(-search_radius, search_radius + 1):
                    neighbor_cell = (cell_coords[0] + di, cell_coords[1] + dj, cell_coords[2] + dk)
                    if neighbor_cell in self.cells:
                        nearby_indices.extend(self.cells[neighbor_cell])
        return nearby_indices

def _extract_particles(particles_or_result):
    if isinstance(particles_or_result, dict) and 'particles' in particles_or_result:
        particles = particles_or_result['particles']
        parameters = particles_or_result.get('parameters', {})
    elif isinstance(particles_or_result, list):
        particles = particles_or_result
        parameters = {}
    else:
        raise TypeError("Input must be either a list of particle dicts or a result dict")
    return particles, parameters

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
    np.random.seed(random_seed)
    particles = []
    spatial_grid = LinkedCellGrid(cell_size=cell_size)
    effective_distance = 2.0 * radius * (1 - overlap)
    
    seed_particle = {'position': np.array([0.0, 0.0, 0.0]), 'added_step': 0, 'inactive': False}
    particles.append(seed_particle)
    spatial_grid.add_particle(0, seed_particle['position'])
    
    def random_unit_vector():
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.arccos(np.random.uniform(-1, 1))
        base_x = np.sin(phi) * np.cos(theta)
        base_y = np.sin(phi) * np.sin(theta)
        base_z = np.cos(phi)
        biased_components = np.array([base_x * bias_factors[0], base_y * bias_factors[1], base_z * bias_factors[2]])
        norm = np.linalg.norm(biased_components)
        return biased_components / norm
    
    def is_space_available_linked(position):
        nearby_indices = spatial_grid.get_nearby_particle_indices(position, search_radius=1)
        for idx in nearby_indices:
            distance = euclidean(position, spatial_grid.particle_positions[idx])
            if distance < effective_distance:
                return False
        return True
    
    step = 1
    while len(particles) < N:
        active_particles = [p for p in particles if not p['inactive']]
        if len(active_particles) == 0:
            break
        
        selected_particle = np.random.choice(active_particles)
        direction = random_unit_vector()
        target_position = selected_particle['position'] + direction * effective_distance
        
        if is_space_available_linked(target_position):
            new_particle = {'position': target_position, 'added_step': step, 'inactive': False}
            particles.append(new_particle)
            spatial_grid.add_particle(len(particles) - 1, target_position)
            
            if len(active_particles) > 1 and np.random.random() < inactivation_probability:
                selected_particle['inactive'] = True
            step += 1
    
    return {'particles': particles}

def calculate_radius_of_gyration(particles_or_result):
    particles, _ = _extract_particles(particles_or_result)
    positions = np.array([p['position'] for p in particles])
    N = len(positions)
    if N <= 1:
        return 0.0
    
    sum_sq_distances = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            rij = euclidean(positions[i], positions[j])
            sum_sq_distances += rij ** 2
    
    Rg = np.sqrt(sum_sq_distances) / N
    return Rg

def calculate_shape_factor(particles_or_result):
    particles, _ = _extract_particles(particles_or_result)
    positions = np.array([p['position'] for p in particles])
    centroid = np.mean(positions, axis=0)
    centered_positions = positions - centroid
    
    I = np.zeros((3, 3))
    for pos in centered_positions:
        x, y, z = pos
        r_squared = np.dot(pos, pos)
        I[0, 0] += r_squared - x**2
        I[1, 1] += r_squared - y**2
        I[2, 2] += r_squared - z**2
        I[0, 1] -= x * y
        I[1, 0] -= x * y
        I[0, 2] -= x * z
        I[2, 0] -= x * z
        I[1, 2] -= y * z
        I[2, 1] -= y * z
    
    eigenvalues, _ = np.linalg.eigh(I)
    eigenvalues_sorted = np.sort(eigenvalues)
    lambda_min = eigenvalues_sorted[0]
    lambda_max = eigenvalues_sorted[-1]
    
    if lambda_min < 1e-15:
        shape_factor = np.inf
    else:
        shape_factor = np.sqrt(lambda_max / lambda_min)
    return shape_factor

# Add these functions at the bottom of fractal_generator.py

def calculate_structure_factor(
    particles_or_result,
    q_min=0.01,
    q_max=10.0,
    n_q=100,
    R_particle=1.0,
    fit_range_factor=2.0,
    visualize=False
):
    particles, _ = _extract_particles(particles_or_result)
    positions = np.array([p['position'] for p in particles])
    N = len(positions)
    if N <= 1:
        raise ValueError("Need at least 2 particles to compute structure factor.")
    
    # Calculate radius of gyration Rg
    pairwise_distances = pdist(positions, metric='euclidean')
    sum_sq_distances = np.sum(pairwise_distances ** 2)
    Rg = np.sqrt(sum_sq_distances) / N
    
    # Define q range
    q_array = np.logspace(np.log10(q_min), np.log10(q_max), n_q)
    
    # Precompute all pairwise distances
    r_ij = pairwise_distances
    
    # Compute S(q)
    S_array = np.zeros_like(q_array)
    for i, q in enumerate(q_array):
        qr = q * r_ij
        sinc_values = np.where(qr == 0, 1.0, np.sin(qr) / qr)
        S_i = 1.0 + (1.0 / N) * np.sum(sinc_values)
        S_array[i] = S_i
    
    # Determine fitting range
    q_fit_min = max(1.0 / Rg, q_min)
    q_fit_max = min(1.0 / R_particle, q_max)
    
    # Filter data for fitting
    mask = (q_array >= q_fit_min) & (q_array <= q_fit_max)
    q_fit = q_array[mask]
    S_fit = S_array[mask]
    
    # Remove non-positive values
    valid_mask = S_fit > 0
    q_fit = q_fit[valid_mask]
    S_fit = S_fit[valid_mask]
    
    if len(q_fit) < 3:
        alpha = np.nan
        alpha_err = np.nan
    else:
        # Linear fit on log-log scale
        log_q = np.log(q_fit)
        log_S = np.log(S_fit)
        
        def linear_model(log_q, alpha, c):
            return -alpha * log_q + c
        
        try:
            popt, pcov = curve_fit(linear_model, log_q, log_S, p0=[1.0, 0.0])
            alpha, c = popt
            alpha_err = np.sqrt(pcov[0, 0])
        except:
            alpha = np.nan
            alpha_err = np.nan
    
    return q_array, S_array, Rg, alpha, alpha_err, None, None

def calculate_porosity(particles_or_result, particle_radius=1.0):
    particles, _ = _extract_particles(particles_or_result)
    positions = np.array([p['position'] for p in particles])
    N = len(positions)
    if N == 0:
        return 0.0, 0.0, 0.0
    
    # Calculate total volume of all particles
    volume_per_particle = (4.0 / 3.0) * np.pi * (particle_radius ** 3)
    total_particle_volume = N * volume_per_particle
    
    # Compute convex hull
    try:
        hull = ConvexHull(positions)
        convex_hull_volume = hull.volume
    except:
        # Fallback to bounding box
        min_coords = np.min(positions, axis=0)
        max_coords = np.max(positions, axis=0)
        convex_hull_volume = np.prod(max_coords - min_coords)
    
    # Calculate porosity
    if convex_hull_volume <= 0:
        porosity = 0.0
    else:
        porosity = 1.0 - (total_particle_volume / convex_hull_volume)
    
    porosity = max(0.0, min(1.0, porosity))
    return porosity, convex_hull_volume, total_particle_volume