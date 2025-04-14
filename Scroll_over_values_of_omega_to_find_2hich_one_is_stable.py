import numpy as np
from scipy.integrate import solve_ivp
from numpy.linalg import eig

# Constants
q = 1.6e-19
V0 = 100
m = 1.6605e-27 * 100
b = 1e-1
R = 1e-2

def coupled_mathieu(tau, Z, qx, qy):
    x, dx, y, dy = Z
    ddx = 2 * qx * np.cos(2 * tau) * x
    ddy = -2 * qy * np.cos(2 * tau) * y
    return [dx, ddx, dy, ddy]

def test_stability_for_omega(omega):
    T = np.pi  # τ ∈ [0, π]
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
            t_eval=[T],
            rtol=1e-9,
            atol=1e-12
        )
        if sol.success:
            Phi_T[:, i] = sol.y[:, 0]
        else:
            print(f"Integration failed for ω = {omega:.2e}")
            return None

    eigenvals = eig(Phi_T)[0]
    abs_vals = np.abs(eigenvals)
    sorted_abs = np.sort(abs_vals)

    stable_x = np.all(sorted_abs[:2] <= 1)
    stable_y = np.all(sorted_abs[2:] <= 1)

    return q_tilde, eigenvals, stable_x, stable_y

# Log scale omega values: 10^2 to 10^12
omega_values = [10**i for i in range(2, 13)]

print(f"{'ω (rad/s)':>12} | {'q_tilde':>12} | {'Stable X':>9} | {'Stable Y':>9}")
print("-" * 80)

for omega in omega_values:
    result = test_stability_for_omega(omega)
    if result is not None:
        q_tilde, eigenvals, stable_x, stable_y = result
        print(f"{omega:12.2e} | {q_tilde:12.4e} | {str(stable_x):>9} | {str(stable_y):>9}")
        for idx, val in enumerate(eigenvals):
            print(f"    λ{idx+1}: {val.real:+.10f} {val.imag:+.10f}j   | |λ| = {abs(val):.10f}")
        print("-" * 80)
