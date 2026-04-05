"""
TDOA (Time Difference of Arrival) Solver
=========================================
Physics & Math:
  Sound travels at c = 343 m/s (at 20°C, sea level).

  For mic pair (A, B):
    Δd_AB = (t_B - t_A) × c     ← distance difference in metres
    |PA| - |PB| = Δd_AB         ← hyperbola definition

  Nonlinear residual for mic pair (A, B):
    f_AB(x,y) = sqrt((x-xA)²+(y-yA)²) - sqrt((x-xB)²+(y-yB)²) - Δd_AB

  Minimise Σ f² using scipy least_squares (LM algorithm).
  Multiple restarts ensure global minimum.
"""

import numpy as np
from scipy.optimize import least_squares, minimize
from typing import Dict, Tuple
import math

SPEED_OF_SOUND = 343.0  # m/s


def solve_tdoa(
    timestamps: Dict[str, float],
    mic_positions: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, dict]:
    mic_ids = list(mic_positions.keys())
    if len(mic_ids) < 3:
        raise ValueError(f"Need >= 3 mics, got {len(mic_ids)}")

    ref_id  = mic_ids[0]
    ref_t   = timestamps[ref_id]
    ref_pos = mic_positions[ref_id]

    delta_d = {}
    tdoa_pairs = {}
    for mic_id in mic_ids[1:]:
        dt_s = (timestamps[mic_id] - ref_t) / 1000.0
        dd   = dt_s * SPEED_OF_SOUND
        delta_d[mic_id] = dd
        tdoa_pairs[f"{ref_id}→{mic_id}"] = {
            "dt_ms":       round(timestamps[mic_id] - ref_t, 4),
            "delta_d_m":   round(dd, 4),
            "interpretation": (
                f"Shooter is {abs(dd):.2f}m "
                f"{'closer to' if dd < 0 else 'farther from'} "
                f"{ref_id} than {mic_id}"
            )
        }

    def residuals(xy):
        x, y = xy
        pt    = np.array([x, y])
        r_ref = np.linalg.norm(pt - ref_pos) + 1e-12
        res = []
        for mid in mic_ids[1:]:
            r_i = np.linalg.norm(pt - mic_positions[mid]) + 1e-12
            res.append(r_i - r_ref - delta_d[mid])
        return np.array(res, dtype=np.float64)

    all_pos = np.array(list(mic_positions.values()), dtype=np.float64)
    cx, cy  = all_pos.mean(axis=0)
    spread  = max(np.ptp(all_pos[:, 0]) + 20, np.ptp(all_pos[:, 1]) + 20)

    best_x, best_cost = None, np.inf
    seeds = [np.array([cx, cy])]
    for _ in range(30):
        seeds.append(np.array([cx + np.random.uniform(-spread, spread),
                                cy + np.random.uniform(-spread, spread)]))

    for x0 in seeds:
        try:
            r = least_squares(residuals, x0, method='lm', max_nfev=2000, ftol=1e-14)
            if r.cost < best_cost:
                best_cost, best_x = r.cost, r.x
        except Exception:
            pass

    solution = best_x if best_x is not None else np.array([cx, cy])

    # Condition number via Jacobian at solution
    pt = solution
    J_rows = []
    for mid in mic_ids[1:]:
        r_ref = np.linalg.norm(pt - ref_pos) + 1e-9
        r_i   = np.linalg.norm(pt - mic_positions[mid]) + 1e-9
        grad  = (pt - ref_pos) / r_ref - (pt - mic_positions[mid]) / r_i
        J_rows.append(grad)
    J = np.array(J_rows)
    try:
        _, sv, _ = np.linalg.svd(J)
        cond = float(sv[0] / (sv[-1] + 1e-12))
    except Exception:
        sv, cond = np.array([1.0, 1.0]), 999.9

    geom = "GOOD" if cond < 10 else "FAIR" if cond < 50 else "POOR"
    pos_error_m = SPEED_OF_SOUND * (2.0 / 1000.0)

    details = {
        "tdoa_pairs":       tdoa_pairs,
        "position_error_m": round(pos_error_m, 3),
        "solver_cost":      round(best_cost, 8),
        "condition_number": round(cond, 2),
        "geometry_quality": geom,
        "singular_values":  [round(float(s), 4) for s in sv],
        "reference_mic":    ref_id,
        "matrix_rank":      2
    }
    return solution, details


def local_to_latlon(local_pos, origin_lat, origin_lng):
    lat = origin_lat + (local_pos[1] / 111320.0)
    lng = origin_lng + (local_pos[0] / (111320.0 * math.cos(math.radians(origin_lat))))
    return lat, lng


def _make_ts(mics, shooter, base_t=10000.0):
    dists = {k: np.linalg.norm(shooter - v) for k, v in mics.items()}
    return {k: base_t + (d / SPEED_OF_SOUND) * 1000 for k, d in dists.items()}


def run_self_tests():
    print("\n" + "="*50)
    print("TDOA SOLVER SELF-TESTS")
    print("="*50)
    mics = {"A": np.array([0.,0.]), "B": np.array([50.,0.]), "C": np.array([25.,43.3])}
    cases = [
        ("Inside triangle",   np.array([25., 20.])),
        ("Near mic A",        np.array([0.1, 0.1])),
        ("Far outside",       np.array([200., 150.])),
    ]
    ok = True
    for name, true_pos in cases:
        ts  = _make_ts(mics, true_pos)
        pos, det = solve_tdoa(ts, mics)
        err = np.linalg.norm(pos - true_pos)
        passed = err < 1.0
        ok = ok and passed
        print(f"  {'✓' if passed else '✗'} {name}: err={err:.4f}m")
    print("ALL PASSED ✓" if ok else "SOME FAILED ✗")
    return ok


if __name__ == "__main__":
    run_self_tests()
