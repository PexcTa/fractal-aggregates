import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
from io import BytesIO

# Proper imports from fractal_generator
from fractal_generator import (
    LinkedCellGrid,
    generate_fractal_aggregate,
    calculate_radius_of_gyration,
    calculate_shape_factor
)

st.title("Fractal Aggregate Generator")

# Parameters
N = st.slider("Number of particles", 10, 5000, 500)
p = st.slider("Inactivation probability", 0.0, 1.0, 0.05)
overlap = st.slider("Particle overlap", 0.0, 0.9, 0.0)
cell_size = st.slider("Cell size", 2.0, 10.0, 4.0)

if st.button("Generate Aggregate"):
    result = generate_fractal_aggregate(
        N=N,
        inactivation_probability=p,
        overlap=overlap,
        cell_size=cell_size,
        visualize=False
    )
    
    # Calculate metrics
    Rg = calculate_radius_of_gyration(result)
    sf = calculate_shape_factor(result)
    
    # Display metrics
    col1, col2 = st.columns(2)
    col1.metric("Radius of Gyration", f"{Rg:.4f}")
    col2.metric("Shape Factor", f"{sf:.4f}")
    
    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    positions = np.array([p['position'] for p in result['particles']])
    ax.scatter(positions[:,0], positions[:,1], positions[:,2], s=1)
    ax.set_box_aspect([1,1,1])
    st.pyplot(fig)