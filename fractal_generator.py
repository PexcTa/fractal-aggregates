import numpy as np
from scipy.spatial.distance import euclidean
# Add these imports at the top of fractal_generator.py
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull

class ParticleLevelGrid:
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
    """Extract particles list and parameters from various input formats"""
    if isinstance(particles_or_result, list):
        # Direct list of particles
        particles = particles_or_result
        parameters = {}
    elif isinstance(particles_or_result, dict):
        if 'particles' in particles_or_result:
            # Standard result format
            particles = particles_or_result['particles']
            parameters = particles_or_result.get('parameters', {})
        elif 'macro_level' in particles_or_result:
            # Agglomerate format
            particles = particles_or_result['particles']
            parameters = particles_or_result.get('parameters', {})
        else:
            raise TypeError("Dictionary input must contain 'particles' or 'macro_level' key")
    else:
        raise TypeError("Input must be either a list of particle dicts or a result dictionary")
    
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
    visualize=False,
    sample_interval=0.05  # Sample every 5% of growth
):
    np.random.seed(random_seed)
    particles = []
    spatial_grid = ParticleLevelGrid(cell_size=cell_size)
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
    # Initialize storage for intermediate states
    intermediate_states = []
    next_sample_point = sample_interval
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

        if len(particles) / N >= next_sample_point:
            # Deep copy particles for this state
            current_state = [{
                'position': p['position'].copy(),
                'added_step': p['added_step'],
                'inactive': p['inactive']
            } for p in particles]
            intermediate_states.append(current_state)
            next_sample_point += sample_interval

        # Final state should always be stored
    final_state = [{
        'position': p['position'].copy(),
        'added_step': p['added_step'],
        'inactive': p['inactive']
    } for p in particles]
    intermediate_states.append(final_state)

    result = {
        'particles': particles,
        'intermediate_states' : intermediate_states,
        'total_N': N  # Store the requested total number of particles
    }
    return result

def calculate_radius_of_gyration(particles_or_result):
    """Calculate radius of gyration for particles or aggregate results"""
    particles, _ = _extract_particles(particles_or_result)
    
    if not particles:
        return 0.0
    
    # Extract positions
    positions = np.array([p['position'] for p in particles])
    N = len(positions)
    
    if N <= 1:
        return 0.0
    
    # Calculate center of mass
    center_of_mass = np.mean(positions, axis=0)
    
    # Calculate squared distances from center of mass
    squared_distances = np.sum((positions - center_of_mass) ** 2, axis=1)
    
    # Calculate radius of gyration
    Rg = np.sqrt(np.sum(squared_distances) / N)
    
    return Rg
    
    sum_sq_distances = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            rij = euclidean(positions[i], positions[j])
            sum_sq_distances += rij ** 2
    
    Rg = np.sqrt(sum_sq_distances) / N
    return Rg

def calculate_shape_factor(particles_or_result):
    """Calculate shape factor (anisotropy) from inertia tensor"""
    particles, _ = _extract_particles(particles_or_result)
    
    if not particles or len(particles) < 3:
        return 1.0  # Spherical by default
    
    positions = np.array([p['position'] for p in particles])
    
    # Calculate center of mass
    center = np.mean(positions, axis=0)
    centered_positions = positions - center
    
    # Calculate inertia tensor
    inertia_tensor = np.zeros((3, 3))
    for pos in centered_positions:
        r2 = np.dot(pos, pos)
        inertia_tensor += r2 * np.eye(3) - np.outer(pos, pos)
    
    # Get eigenvalues (moments of inertia)
    eigenvalues = np.linalg.eigvalsh(inertia_tensor)
    eigenvalues = np.sort(eigenvalues)  # Sort from smallest to largest
    
    # Shape factor = sqrt(max_eigenvalue / min_eigenvalue)
    if eigenvalues[0] == 0:
        return 1.0
    
    shape_factor = np.sqrt(eigenvalues[2] / eigenvalues[0])
    return shape_factor

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

def calculate_aggregate_properties(aggregate_particles):
    """Calculate Rg and center of mass for an aggregate"""
    positions = np.array([p['position'] for p in aggregate_particles])
    center = np.mean(positions, axis=0)
    rg = calculate_radius_of_gyration(positions - center)
    return rg, center

class AggregateLevelGrid:
    """Spatial grid for efficient aggregate-level collision detection"""
    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.grid = {}
    
    def _get_cell(self, position):
        """Get grid cell coordinates for a position"""
        return tuple((np.array(position) / self.cell_size).astype(int))
    
    def add_particle(self, idx, position):
        """Add particle to grid"""
        cell = self._get_cell(position)
        if cell not in self.grid:
            self.grid[cell] = []
        self.grid[cell].append(idx)
    
    def get_neighbors(self, position, radius):
        """Get indices of particles within physical radius of position"""
        cell = self._get_cell(position)
        neighbors = []
        # Determine how many cells to search based on physical radius
        search_dist = int(np.ceil(radius / self.cell_size))
        
        for dx in range(-search_dist, search_dist + 1):
            for dy in range(-search_dist, search_dist + 1):
                for dz in range(-search_dist, search_dist + 1):
                    check_cell = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                    if check_cell in self.grid:
                        neighbors.extend(self.grid[check_cell])
        
        return neighbors
    

def generate_agglomerate(
    aggregates_data,
    N_sub=None,
    contact_scaling_factor=1.0,
    macro_cell_size_beta=2.5,
    random_seed=42,
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
    
    Returns:
    --------
    dict with keys:
        'particles' : list of particle dicts with tracking info
        'macro_level' : dict with macro aggregate info
        'parameters' : dict of generation parameters
        'metadata' : dict with statistics
    """
    
    print("="*50)
    print("STARTING AGGLOMERATE GENERATION")
    print(f"Input type: {type(aggregates_data)}")
    print(f"Number of input aggregates: {len(aggregates_data)}")
    print("-"*50)
    
    np.random.seed(random_seed)
    
    # ========== EXTRACT AGGREGATES ==========
    print("EXTRACTING AGGREGATES FROM INPUT DATA...")
    aggregates = []
    for i, agg_data in enumerate(aggregates_data):
        print(f"  Processing aggregate #{i}:")
        print(f"    Type: {type(agg_data)}")
        
        if isinstance(agg_data, dict):
            print(f"    Dictionary keys: {list(agg_data.keys())}")
            
        if isinstance(agg_data, dict) and 'particles' in agg_data:
            print(f"    Found 'particles' key with {len(agg_data['particles'])} particles")
            aggregates.append(agg_data['particles'])
            print("    ✓ Successfully extracted particles list")
        elif isinstance(agg_data, list):
            print(f"    Direct list with {len(agg_data)} elements")
            aggregates.append(agg_data)
            print("    ✓ Used direct list")
        else:
            print("    ❌ Unexpected format - wrapping in list")
            aggregates.append([agg_data])
    
    original_num = len(aggregates)
    print(f"\nExtracted {original_num} aggregates for assembly")
    print("-"*50)
    
    # ========== N_sub SUBSAMPLING ==========
    if N_sub is not None:
        print(f"SUBSAMPLING: Selecting {N_sub} aggregates from {original_num} (with replacement)")
        selected_indices = np.random.choice(original_num, size=N_sub, replace=True)
        aggregates = [aggregates[i] for i in selected_indices]
        print(f"Selected indices: {selected_indices}")
        print(f"New number of aggregates: {len(aggregates)}")
        print("-"*50)
    
    # ========== CALCULATE AGGREGATE PROPERTIES ==========
    print("CALCULATING PROPERTIES FOR EACH AGGREGATE...")
    agg_properties = []
    for i, agg in enumerate(aggregates):
        print(f"  Processing aggregate #{i} with {len(agg)} particles")
        print(f"    First particle keys: {list(agg[0].keys()) if agg else 'EMPTY'}")
        print(f"    First position type: {type(agg[0]['position']) if agg and agg[0] else 'N/A'}")
        print(f"    First position shape: {agg[0]['position'].shape if agg and agg[0] and hasattr(agg[0]['position'], 'shape') else 'N/A'}")
        
        try:
            Rg, center = calculate_aggregate_properties(agg)
            print(f"    ✓ Calculated Rg={Rg:.4f}, center={center}")
            agg_properties.append({'Rg': Rg, 'center': center})
        except Exception as e:
            print(f"    ❌ ERROR calculating properties: {str(e)}")
            print(f"    Aggregate structure: {type(agg)} with {len(agg) if hasattr(agg, '__len__') else 'unknown'} items")
            raise
    
    mean_Rg = np.mean([p['Rg'] for p in agg_properties])
    print(f"\nMean Rg across all aggregates: {mean_Rg:.4f}")
    print("-"*50)
    
    # ========== MACRO-SCALE PLACEMENT (Porous Eden) ==========
    print("PLACING AGGREGATES AT MACRO SCALE...")
    macro_positions = []  # Centers of aggregates in macro space
    macro_cell_size = macro_cell_size_beta * mean_Rg
    print(f"Macro cell size: {macro_cell_size:.4f} (beta={macro_cell_size_beta}, mean_Rg={mean_Rg:.4f})")
    
    spatial_grid = AggregateLevelGrid(macro_cell_size)
    print("✓ Created spatial grid")
    
    # Place first aggregate at origin
    macro_positions.append(np.array([0.0, 0.0, 0.0]))
    spatial_grid.add_particle(0, macro_positions[0])
    print("✓ Placed first aggregate at origin")
    
    # Place remaining aggregates
    active_indices = [0]
    step = 1
    print(f"Starting placement loop with {len(aggregates)-1} remaining aggregates")
    
    while len(macro_positions) < len(aggregates):
        if len(active_indices) == 0:
            print("  ⚠ No active indices - starting new cluster")
            active_indices = [len(macro_positions) - 1]
        
        # Pick random active aggregate
        selected_idx = np.random.randint(len(active_indices))
        selected = active_indices[selected_idx]
        print(f"  Selected active aggregate #{selected} (index {selected_idx} in active list)")
        
        # Random direction
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)
        
        # Place new aggregate
        current_idx = len(macro_positions)
        print(f"  Placing aggregate #{current_idx}")
        
        if current_idx >= len(agg_properties):
            print(f"  ❌ ERROR: current_idx={current_idx} exceeds agg_properties length={len(agg_properties)}")
            raise IndexError("Index out of bounds in agg_properties")
            
        sum_Rg = agg_properties[selected]['Rg'] + agg_properties[current_idx]['Rg']
        distance = contact_scaling_factor * sum_Rg * 2
        print(f"    Sum Rg: {sum_Rg:.4f}, Distance: {distance:.4f}")
        
        new_pos = macro_positions[selected] + direction * distance
        print(f"    Proposed position: {new_pos}")
        
        # Check for collisions
        collision = False
        neighbors = spatial_grid.get_neighbors(new_pos, radius=sum_Rg * 2)
        print(f"    Checking {len(neighbors)} potential neighbors for collisions")
        
        for neighbor_idx in neighbors:
            dist_to_neighbor = np.linalg.norm(new_pos - macro_positions[neighbor_idx])
            min_dist = contact_scaling_factor * (
                agg_properties[current_idx]['Rg'] + 
                agg_properties[neighbor_idx]['Rg']
            ) * 2
            
            print(f"      Neighbor #{neighbor_idx}: distance={dist_to_neighbor:.4f}, min_dist={min_dist:.4f}")
            
            if dist_to_neighbor < min_dist:
                print("      ❌ Collision detected!")
                collision = True
                break
        
        if not collision:
            print("    ✓ No collision - placing aggregate")
            macro_positions.append(new_pos)
            spatial_grid.add_particle(len(macro_positions) - 1, new_pos)
            active_indices.append(len(macro_positions) - 1)
            step += 1
            print(f"    Updated positions count: {len(macro_positions)}")
        else:
            print("    ⚠ Collision - skipping placement")
        
        # Randomly deactivate
        if np.random.random() < 0.1 and len(active_indices) > 0:
            deactivate_idx = np.random.randint(len(active_indices))
            deactivated = active_indices.pop(deactivate_idx)
            print(f"    Randomly deactivated aggregate #{deactivated}")
    
    print(f"\n✓ Successfully placed all {len(macro_positions)} aggregates")
    print("-"*50)
    
    # ========== TRANSFORM COORDINATES TO GLOBAL SYSTEM ==========
    print("TRANSFORMING COORDINATES TO GLOBAL SYSTEM...")
    output_particles = []
    total_particles = 0
    
    for agg_idx, (agg, macro_pos) in enumerate(zip(aggregates, macro_positions)):
        print(f"  Processing aggregate #{agg_idx} with {len(agg)} particles")
        print(f"    Macro position: {macro_pos}")
        
        Rg, local_center = agg_properties[agg_idx]['Rg'], agg_properties[agg_idx]['center']
        print(f"    Local center: {local_center}, Rg: {Rg:.4f}")
        
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
        
        total_particles += len(agg)
        print(f"    Added {len(agg)} particles from this aggregate")
    
    print(f"\n✓ Created {len(output_particles)} total particles in agglomerate")
    print("-"*50)
    
    # ========== BUILD RETURN DICT ==========
    print("BUILDING RESULT DICTIONARY...")
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
    
    print("✓ Result dictionary created successfully")
    print("="*50)
    
    return result