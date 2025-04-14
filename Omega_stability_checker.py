import numpy as np
from scipy.integrate import solve_ivp
from numpy.linalg import eig

# Constants
q = 1.6e-19
V0 = 100
m = 1.6605e-27 * 100
b = 1e-1
R = 1e-2

# Define the 4D coupled Mathieu system
def coupled_mathieu(tau, Z, qx, qy):
    x, dx, y, dy = Z
    ddx = 2 * qx * np.cos(2 * tau) * x
    ddy = -2 * qy * np.cos(2 * tau) * y
    return [dx, ddx, dy, ddy]

def test_stability_for_omega(omega):
    T = np.pi  # Dimensionless period in τ
    q_tilde = (8 * q * V0) / (m * omega**2 * b**2 * np.log(R))
    qx = qy = q_tilde

    ICs = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]

    Phi_T = np.zeros((4, 4))

    for i, ic in enumerate(ICs):
        sol = solve_ivp(
            coupled_mathieu,
            [0, T],
            ic,
            args=(qx, qy),
            method='RK45',
            t_eval=[T],
            rtol=1e-9,
            atol=1e-12
        )

        if sol.success:
            Phi_T[:, i] = np.array(sol.y).reshape(4, -1)[:, -1]
        else:
            print(f"Integration failed for IC{i+1}")
            return

    eigenvals = eig(Phi_T)[0]
    abs_vals = np.abs(eigenvals)
    sorted_abs = np.sort(abs_vals)

    stable_x = np.all(sorted_abs[:2] <= 1)
    stable_y = np.all(sorted_abs[2:] <= 1)

    print(f"\nω = {omega:.2e} rad/s")
    print(f"q_tilde = {q_tilde:.15f}")
    print("Floquet multipliers:")
    for idx, val in enumerate(eigenvals):
        print(f"λ{idx+1}: {val:.10f}  | |λ| = {abs(val):.10f}")
    print(f"Stable in x-direction: {'Yes' if stable_x else 'No'}")
    print(f"Stable in y-direction: {'Yes' if stable_y else 'No'}")

# === Example usage ===
omega_input = 1e5  # rad/s — very low
test_stability_for_omega(omega_input)
