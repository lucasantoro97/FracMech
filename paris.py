import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Material Properties
E = 68000  # Young's modulus in MPa for 6060-T6 Aluminum
nu = 0.33  # Poisson's ratio
sigma_a = 70  # Fatigue strength in MPa
UTS = 220  # Ultimate tensile strength in MPa

# Paris Law Constants (Example values for Aluminum)
C = 1e-12  # Paris law constant (MPa^-m^-1 cycle^-1)
m = 3.0    # Paris law exponent

# Specimen Geometry
W = 100.0  # Width of the specimen in mm
t = 6.0    # Thickness of the specimen in mm
L = 300.0  # Distance between outer supports in mm
a_initial = 1.0  # Initial crack length in mm
a_final = 10.0    # Final crack length in mm

# Four-Point Bending Configuration
def four_point_bending_stress(P, W, L, a):
    """
    Calculate the stress range (Δσ) in four-point bending.

    Parameters:
    P : Load applied in N
    W : Width of the specimen in mm
    L : Span between outer supports in mm
    a : Crack length in mm

    Returns:
    Δσ : Stress range in MPa
    """
    # Moment in the middle region
    M = P * L / 4  # in N*mm

    # Section modulus for rectangular cross-section
    S = (W * t**2) / 6  # in mm^3

    # Stress range
    delta_sigma = (2 * M) / S  # in MPa

    return delta_sigma

# Stress Intensity Factor Calculation
def stress_intensity_factor(delta_sigma, a, W, t):
    """
    Calculate the Stress Intensity Factor (ΔK) for a crack in four-point bending.

    Parameters:
    delta_sigma : Stress range in MPa
    a : Crack length in mm
    W : Width of the specimen in mm
    t : Thickness of the specimen in mm

    Returns:
    Delta_K : Stress Intensity Factor in MPa*sqrt(mm)
    """
    # Geometric factor Y for four-point bending with central crack
    # For simplicity, assuming Y = 1.12 (approximate value for edge cracks)
    # More accurate Y can be determined using finite element analysis or more detailed formulas
    Y = 1.12

    Delta_K = Y * delta_sigma * np.sqrt(np.pi * a)

    return Delta_K

# Paris Law for Crack Growth
def paris_law(a, C, m):
    """
    Paris law differential equation.

    Parameters:
    a : Current crack length in mm
    C : Paris law constant
    m : Paris law exponent

    Returns:
    da_dN : Crack growth per cycle in mm/cycle
    """
    Delta_K = stress_intensity_factor(delta_sigma=delta_sigma(a), a=a, W=W, t=t)
    da_dN = C * (Delta_K)**m
    return da_dN

# Differential Function for Integration
def da_dN(a, C, m):
    Delta_K = stress_intensity_factor(delta_sigma=delta_sigma(a), a=a, W=W, t=t)
    return 1 / (C * (Delta_K)**m)

# Load Function (can be made more complex if needed)
def load_function():
    """
    Define the load function P as a function of crack length a.
    For simplicity, assuming a constant load.
    """
    return P  # in N

# Stress Range as a Function of Crack Length
def delta_sigma(a):
    """
    Calculate stress range Δσ as a function of crack length a.

    Parameters:
    a : Crack length in mm

    Returns:
    Δσ : Stress range in MPa
    """
    P_current = load_function()
    return four_point_bending_stress(P_current, W, L, a)

# Calculate Number of Cycles using Paris Law Integration
def calculate_cycles(a_initial, a_final, C, m):
    """
    Calculate the number of cycles to grow the crack from a_initial to a_final.

    Parameters:
    a_initial : Initial crack length in mm
    a_final : Final crack length in mm
    C : Paris law constant
    m : Paris law exponent

    Returns:
    N : Number of cycles
    """
    # Integrate da/dN from a_initial to a_final
    N, _ = quad(lambda a: 1 / (C * (stress_intensity_factor(delta_sigma(a), a, W, t))**m), a_initial, a_final)
    return N

# Example Load
P = 900.0  # Applied load in N

# Calculate Stress Range for initial crack
delta_sigma_initial = four_point_bending_stress(P, W, L, a_initial)
print(f"Initial Stress Range (Δσ): {delta_sigma_initial:.2f} MPa")

# Calculate Stress Intensity Factor for initial crack
Delta_K_initial = stress_intensity_factor(delta_sigma_initial, a_initial, W, t)
print(f"Initial Stress Intensity Factor (ΔK): {Delta_K_initial:.2f} MPa√mm")

# Calculate Number of Cycles for Crack Growth
N_cycles = calculate_cycles(a_initial, a_final, C, m)
print(f"Number of Cycles to grow crack from {a_initial} mm to {a_final} mm: {N_cycles:.2e} cycles")

# Parametric Analysis: Different Crack Lengths
crack_lengths = np.linspace(a_initial, a_final, 50)
Delta_K_values = stress_intensity_factor(delta_sigma(crack_lengths), crack_lengths, W, t)

# Plot Stress Intensity Factor vs Crack Length
plt.figure(figsize=(8,6))
plt.plot(crack_lengths, Delta_K_values, label='ΔK vs Crack Length')
plt.xlabel('Crack Length (mm)')
plt.ylabel('Stress Intensity Factor ΔK (MPa√mm)')
plt.title('Stress Intensity Factor vs Crack Length')
plt.legend()
plt.grid(True)
plt.show()

# Optional: Plot Number of Cycles for Different Initial Loads
loads = [500, 700, 900, 1100]  # Different applied loads in N
a_final_list = [10, 12, 15]  # Different final crack lengths in mm

for P in loads:
    N_cycles = calculate_cycles(a_initial, a_final, C, m)
    print(f"Applied Load: {P} N, Number of Cycles: {N_cycles:.2e} cycles")

# Advanced: Implementing Crack Growth Simulation
def simulate_crack_growth(a_initial, a_final, C, m, P, step=0.1):
    """
    Simulate crack growth and calculate cumulative number of cycles.

    Parameters:
    a_initial : Initial crack length in mm
    a_final : Final crack length in mm
    C : Paris law constant
    m : Paris law exponent
    P : Applied load in N
    step : Step size for crack growth in mm

    Returns:
    a_values : Array of crack lengths
    N_values : Array of cumulative number of cycles
    """
    a_values = [a_initial]
    N_values = [0]
    a = a_initial
    N = 0
    while a < a_final:
        da = step
        Delta_K = stress_intensity_factor(delta_sigma(a), a, W, t)
        da_dN_val = C * (Delta_K)**m
        dN = da / da_dN_val
        N += dN
        a += da
        a_values.append(a)
        N_values.append(N)
    return np.array(a_values), np.array(N_values)

# Run Crack Growth Simulation
a_vals, N_vals = simulate_crack_growth(a_initial, a_final, C, m, P, step=0.1)

# Plot Number of Cycles vs Crack Length
plt.figure(figsize=(8,6))
plt.plot(a_vals, N_vals, label='Crack Growth Simulation')
plt.xlabel('Crack Length (mm)')
plt.ylabel('Number of Cycles (N)')
plt.title('Crack Growth Simulation using Paris Law')
plt.legend()
plt.grid(True)
plt.show()
