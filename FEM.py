import numpy as np
import matplotlib.pyplot as plt
from sfepy.discrete import (FieldVariable, Material, Integral, Equation, Equations, Problem)
from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.terms import Term
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
from sfepy.discrete.conditions import EssentialBC
import subprocess
from scipy.integrate import quad
import pandas as pd
from sfepy.mechanics.matcoefs import stiffness_from_lame

# -----------------------------
# Material Properties
# -----------------------------
E = 68000.0       # Young's modulus in MPa for 6060-T6 Aluminum
nu = 0.33         # Poisson's ratio
sigma_a = 70.0    # Fatigue strength in MPa
UTS = 220.0       # Ultimate tensile strength in MPa

# -----------------------------
# Paris Law Constants (Example values for Aluminum)
# -----------------------------
C = 1e-12  # Paris law constant (MPa^-m^-1 cycle^-1)
m = 3.0    # Paris law exponent

# -----------------------------
# Specimen Geometry
# -----------------------------
W = 100.0  # Width of the specimen in mm
H = 50.0   # Height of the specimen in mm
t = 6.0    # Thickness of the specimen in mm
L = 300.0  # Distance between outer supports in mm
a_initial = 1.0  # Initial crack length in mm
a_final = 6.0    # Final crack length in mm (set to thickness)

# -----------------------------
# Load Cases
# -----------------------------
loads = [500.0, 700.0, 900.0, 1100.0]  # Applied loads in N

# -----------------------------
# Control Volume Radius
# -----------------------------
R_c = 0.12  # mm

# -----------------------------
# Tolerance for coordinate-based region definitions
# -----------------------------
tol = 1e-6

# -----------------------------
# Function to Generate Mesh
# -----------------------------
def generate_mesh(crack_length):
    """
    Generate mesh using Gmsh for a given crack length.
    """
    geo_content = f"""
    W = {W};
    H = {H};
    crack_length = {crack_length};

    // Define corner points
    Point(1) = {{0, 0, 0, 1.0}};
    Point(2) = {{{W}, 0, 0, 1.0}};
    Point(3) = {{{W}, {H}, 0, 1.0}};
    Point(4) = {{0, {H}, 0, 1.0}};

    // Define the crack (embedded line)
    Point(5) = {{{W/2 - crack_length/2}, {H/2}, 0, 0.1}};
    Point(6) = {{{W/2 + crack_length/2}, {H/2}, 0, 0.1}};
    Line(5) = {{5, 6}};

    // Define lines (boundary)
    Line(1) = {{1, 2}};
    Line(2) = {{2, 3}};
    Line(3) = {{3, 4}};
    Line(4) = {{4, 1}};

    // Define the loop and surface
    Line Loop(1) = {{1, 2, 3, 4}};
    Plane Surface(1) = {{1}};

    // Embed the crack line in the surface
    Line{{5}} In Surface{{1}};

    // Define physical groups with explicit IDs
    Physical Line("Left") = {{4}};
    Physical Line("Right") = {{2}};
    Physical Line("Top") = {{3}};
    Physical Line("Bottom") = {{1}};
    Physical Line("Crack") = {{5}};
    Physical Surface("Specimen") = {{1}};
    """
    # Write to a temporary .geo file
    with open("geometry.geo", "w") as geo_file:
        geo_file.write(geo_content)

    # Generate mesh using Gmsh
    try:
        subprocess.run(["gmsh", "-2", "geometry.geo", "-format", "msh2", "-o", "geometry.msh"], check=True)
    except subprocess.CalledProcessError as e:
        print("Error during mesh generation with Gmsh:", e)
        return False

    return True

# -----------------------------
# Function to Compute ASED
# -----------------------------
def compute_ased(problem, state, crack_tip, R_c):
    """
    Compute the Average Strain Energy Density (ASED) in the control volume.
    """
    from sfepy.discrete.probes import Probe

    # Create a grid of points around the crack tip within R_c
    num_points = 1000
    angles = np.linspace(0, 2*np.pi, num_points)
    x_coords = crack_tip[0] + R_c * np.cos(angles)
    y_coords = crack_tip[1] + R_c * np.sin(angles)
    points = np.vstack((x_coords, y_coords)).T

    # Evaluate strain and stress at these points
    probe = Probe(points, problem.domain, delta=R_c/100)

    # Evaluate strain
    strain = probe.evaluate('ev_cauchy_strain.2.Omega(u)', u=state())

    # Evaluate stress
    stress = problem.materials['m'].D @ strain

    # Compute strain energy density W = 0.5 * epsilon : sigma
    W = 0.5 * np.einsum('ij...,ij...->...', strain, stress)

    # Average the strain energy density
    ASED = np.mean(W)

    return ASED

# -----------------------------
# Function to Calculate NSIF
# -----------------------------
def calculate_nsif(ASED, E, R_c, c_w=0.5, e_1=0.133):
    """
    Calculate the Nominal Stress Intensity Factor (NSIF).
    """
    Delta_sigma_n = np.sqrt((E * R_c) / (c_w * e_1) * ASED)
    return Delta_sigma_n

# -----------------------------
# Function to Perform FEM Analysis
# -----------------------------
def fem_analysis(crack_length, P):
    """
    Perform FEM analysis for a given crack length and applied load.
    """
    # Generate mesh
    mesh_generated = generate_mesh(crack_length)
    if not mesh_generated:
        print(f"Failed to generate mesh for crack length {crack_length} mm and load {P} N.")
        return None

    # Load the mesh directly in SfePy
    try:
        mesh = Mesh.from_file('geometry.msh')
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return None

    domain = FEDomain('domain', mesh)
    min_coor, max_coor = domain.get_mesh_bounding_box()

    # Define regions based on coordinates instead of physical groups
    omega = domain.create_region('Omega', 'all')

    # Define regions based on coordinates with tolerance
    left = domain.create_region('Left', f'vertices in x < {tol}', 'facet')
    right = domain.create_region('Right', f'vertices in x > {W - tol}', 'facet')
    top = domain.create_region('Top', f'vertices in y > {H - tol}', 'facet')
    bottom = domain.create_region('Bottom', f'vertices in y < {tol}', 'facet')

    # Check regions
    for region in [left, right, top, bottom]:
        print(f"Region '{region.name}' has {region.get_n_cells()} cells and {region.get_n_vertices()} vertices.")

    # Define field
    field = Field.from_args('displacement', np.float64, 'vector', omega, approx_order=2)

    # Define variables
    u = FieldVariable('u', 'unknown', field)
    v = FieldVariable('v', 'test', field, primary_var_name='u')

    # Define material
    D = stiffness_from_lame(2, E=E, nu=nu)
    material = Material('m', D=D)

    # Define integral
    integral = Integral('i', order=3)

    # Define terms
    t1 = Term.new('dw_lin_elastic(m.D, v, u)', integral, omega, m=material, v=v, u=u)

    # Define traction
    traction_magnitude = P / W  # Distribute load over the width
    val = np.array([0.0, -traction_magnitude])

    # Define surface term for traction on the 'Top' region
    ts = Term.new('dw_surface_ltr(v, traction.val)', integral, top, v=v, traction=val)

    # Define equations
    eq = Equation('balance', t1 + ts)
    eqs = Equations([eq])

    # Define problem
    ls = ScipyDirect({})
    nls = Newton({})
    problem = Problem('elasticity', equations=eqs)
    problem.set_solver(nls)
    problem.set_linear_solver(ls)

    # Set boundary conditions
    fix_left = EssentialBC('fix_left', left, {'u.all': 0.0})
    fix_right = EssentialBC('fix_right', right, {'u.all': 0.0})
    problem.set_bcs(ebcs=[fix_left, fix_right])

    # Solve the problem
    try:
        state = problem.solve()
    except Exception as e:
        print(f"Error during FEM solve: {e}")
        return None

    # Post-processing to compute ASED
    crack_tip = np.array([W/2, H/2])  # Crack tip is at (W/2, H/2)
    ASED = compute_ased(problem, state, crack_tip, R_c)

    return ASED

# -----------------------------
# Function to Calculate Number of Cycles
# -----------------------------
def calculate_cycles(a_initial, a_final, C, m, P):
    """
    Calculate the number of cycles to grow the crack from a_initial to a_final.
    """
    # Define delta_K as a function of a
    def delta_K(a):
        # Calculate stress range for given load
        M = P * L / 4
        S = (W * t**2) / 6
        delta_sigma = (2 * M) / S  # MPa
        Y = 1.12
        return Y * delta_sigma * np.sqrt(np.pi * a)
    
    # Define the integrand for Paris Law
    def integrand(a):
        K = delta_K(a)
        if K == 0:
            return 0
        return 1 / (C * (K)**m)
    
    # Integrate from a_initial to a_final
    N, _ = quad(integrand, a_initial, a_final)
    return N

# -----------------------------
# Perform Parametric Analysis
# -----------------------------
results = []

for P in loads:
    print(f"\nAnalyzing Load: {P} N")
    # Generate crack lengths up to t, ensuring they do not exceed thickness
    crack_lengths = np.linspace(a_initial, a_final, 4)  # 1.0, 3.0, 5.0, 6.0 mm
    for crack_length in crack_lengths:
        print(f"  Crack Length: {crack_length:.1f} mm")
        # Perform FEM analysis to get ASED
        ASED = fem_analysis(crack_length, P)
        
        if ASED is None:
            print(f"  Skipping due to FEM analysis failure.")
            continue
        
        # Calculate NSIF
        NSIF = calculate_nsif(ASED, E, R_c)
        
        # Estimate Number of Cycles
        N_cycles = calculate_cycles(a_initial, crack_length, C, m, P)
        
        # Store results
        results.append({
            'Load (N)': P,
            'Crack Length (mm)': crack_length,
            'ASED (mJ/mm^3)': ASED,
            'NSIF (MPa√mm)': NSIF,
            'Cycles': N_cycles
        })

# -----------------------------
# Convert results to DataFrame and Output
# -----------------------------
df_results = pd.DataFrame(results)
print("\nSimulation Results:")
print(df_results)

# Save results to CSV
df_results.to_csv('simulation_results.csv', index=False)

# -----------------------------
# Plot Number of Cycles vs Crack Length for Different Loads
# -----------------------------
plt.figure(figsize=(10, 7))
for P in loads:
    subset = df_results[df_results['Load (N)'] == P]
    plt.plot(subset['Crack Length (mm)'], subset['Cycles'], marker='o', label=f'Load={P} N')

plt.xlabel('Crack Length (mm)')
plt.ylabel('Number of Cycles (N)')
plt.title('Number of Cycles vs Crack Length for Different Loads')
plt.yscale('log')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig('cycles_vs_crack_length.png')
plt.show()
