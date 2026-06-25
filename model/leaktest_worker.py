"""leaktest_worker.py -- importable small-return worker for the PARALLEL leak test.

Exact fig4 config (the chaotic `maranon_ward` stable construct, years=60, the real
grazing/GGE params and solver floor), but returns ONLY the small `_clim` dict
(return_traj=False) so the parent doesn't accumulate trajectories -- that isolates
the per-worker leak and makes maxtasksperchild (which recycles workers, not the
parent) a fair test.

Place this beside seasonal_scan_harness.py / fig4_panels.py so spawn workers can
import it by reference.
"""
import seasonal_scan_harness as ssh

_SK = {**ssh.SEASONAL_SOLVER_KWARGS, 'instability_neg_threshold': -1e-2}
_P = dict(GGE=0.31, mP=0.0015, m_Z=0.10, KsZ=0.23, sigma_log=0.20)


def run_small(forc_era, fish):
    """One fig4 cell -> small _clim dict (no trajectory)."""
    return ssh.run_one(
        ssh.allometry('maranon_ward'), forc_era, fish_rate=fish, years=60, spinup=15,
        mP=_P['mP'], m_Z=_P['m_Z'],
        grazing={'KsZ': _P['KsZ'], 'sigma_log': _P['sigma_log']},
        iv_overrides={'GrazingRouter': {'gge': _P['GGE']}},
        solver_kwargs=_SK, return_traj=False,
    )