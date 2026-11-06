#!/usr/bin/env python3
"""One-dimensional base state of the two-layer system, eqs (2.5)-(2.7), (2.13).

Independent finite-volume solver in the dimensionless variables of the paper:
a cross-check of the OpenFOAM results and the source of K_rho, eq (2.3).
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

DB, DC = 0.64, 0.48
RB, RC = 2.44, 2.28
VM_PAPER = 0.045  # (mu/rho)_H2O + (mu/rho)_NaCl, L/mol


@dataclass
class BaseState:
    z: np.ndarray
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    W: float
    Da: float
    history: np.ndarray  # t, omega, dVw, dVw/Vrz

    @property
    def rho(self):
        return self.A + RB * self.B + RC * self.W * self.C

    @property
    def R(self):
        return self.Da * self.A * self.B

    @property
    def krho(self):
        """K_rho: density at the interface over the initial lower layer."""
        return self.rho[0] / (RB * self.B[-1])


def _tridiagonal(n, c, sink):
    """Implicit diffusion with zero flux on both ends, c = D*dt/dz^2."""
    ab = np.zeros((3, n))
    ab[0, 1:] = -c
    ab[2, :-1] = -c
    ab[1] = 1 + 2 * c + sink
    ab[1, 0] -= c
    ab[1, -1] -= c
    return ab


def solve(gamma, t_end=5.0, A0=1.0, Da=1000.0, Vm=VM_PAPER, H=20.0, n=1000,
          dt0=1e-6, dt_max=1e-3, water=True, outer=3):
    """March the base state to t_end.

    gamma = B0/A0, A0 in mol/L, Vm in L/mol, H is the depth of each layer.
    The acid is solved in the two layers together, the interface is in the
    middle of the array.
    """
    dz = H / n
    z = (np.arange(n) + 0.5) * dz
    A = np.concatenate((np.ones(n), np.zeros(n)))  # upper layer, lower layer
    B = np.full(n, float(gamma))
    C = np.zeros(n)

    t, dt, W = 0.0, dt0, 1.0
    history = []
    while t < t_end - 1e-12:
        dt = min(dt, t_end - t)
        A_old, B_old, C_old = A.copy(), B.copy(), C.copy()

        for _ in range(outer):
            sink = np.concatenate((np.zeros(n), Da * B * dt))
            A = solve_banded((1, 1), _tridiagonal(2 * n, dt / dz**2, sink), A_old)
            A1 = A[n:]
            B = solve_banded(
                (1, 1), _tridiagonal(n, DB * dt / dz**2, Da * A1 * dt), B_old
            )

        C = solve_banded(
            (1, 1),
            _tridiagonal(n, DC * dt / dz**2, np.zeros(n)),
            C_old + dt * Da * A1 * B,
        )

        t += dt
        R = A1 * B
        omega = (z * R).sum() / R.sum() if R.sum() > 0 else 0.0
        dVw = Vm * A0 * C.sum() * dz
        ratio = dVw / omega if omega > 0 else 0.0
        W = 1 / (1 + ratio) if water else 1.0
        history.append((t, omega, dVw, ratio))
        dt = min(dt * 1.03, dt_max)

    return BaseState(z, A[n:].copy(), B, C, W, Da, np.array(history))
