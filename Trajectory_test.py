import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# === Constants ===
q = 1.6e-19                     # Charge [C]
V0 = 100                        # Voltage [V]
m = 1.6605e-27 * 100            # Mass [kg]
b = 1e-1                        # Electrode spacing [m]
R = 1e-2                        # Geometric factor
omega = 1e6                     # Angular frequency [rad/s]

# === Compute dimensionless q_tilde ===
q_tilde = (8 * q * V0) / (m * omega**2 * b**2 * np.log(R))
print(f"q_tilde = {q_tilde:.8e}")

# === Define the ODE system ===
def coupled_mathieu(tau, Z):
    x, dx, y, dy = Z
    ddx = 2 * q_tilde * np.cos(2 * tau) * x
    ddy = -2 * q_tilde * np.cos(2 * tau) * y
    return [dx, ddx, dy, ddy]

# === Initial condition: near origin, at rest ===
IC = [1e-6, 0, 1e-6, 0]

# === Time span in τ (dimensionless time) ===
tau_max = 1000000 * np.pi
tau_eval = np.linspace(0, tau_max, 10000000)

# === Integrate the system ===
sol = solve_ivp(coupled_mathieu, [0, tau_max], IC, t_eval=tau_eval, rtol=1e-9, atol=1e-12)

# === Extract results ===
x = sol.y[0]
y = sol.y[2]
tau = sol.t

# === Plot time-domain trajectories ===
plt.figure(figsize=(10, 4))
plt.plot(tau, x, label="x(τ)")
plt.plot(tau, y, label="y(τ)")
plt.xlabel("τ (dimensionless time)")
plt.ylabel("Displacement")
plt.title("Trajectory of Ion in x and y under Mathieu Equations")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === Optional: 2D Phase Portrait ===
plt.figure(figsize=(5, 5))
plt.plot(x, y, color='purple')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Phase Portrait: y vs x")
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.show()
