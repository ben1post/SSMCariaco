"""
resize_cariaco_setup.py
=======================

Build a fresh ``xso.setup`` that mirrors a previously-saved setup
(e.g. one persisted to netCDF from a parameter scan), but with
different numbers of phytoplankton and zooplankton size classes.

Typical use
-----------

>>> import xarray as xr, numpy as np
>>> from cariaco_ssm_setup import (
...     model, generate_size_classes,
...     compute_K_s, compute_mu_max_maranon, compute_I_max,
...     compute_fish_kernel_vdl_joint,
... )
>>> from resize_cariaco_setup import make_resized_setup, build_cariaco_regenerators
>>>
>>> loaded = xr.open_dataset('best_cell_stability.nc')
>>>
>>> n_P_new, n_Z_new = 30, 30
>>> P_new = generate_size_classes(n_P_new, esd_min=0.5, esd_max=200)
>>> Z_new = generate_size_classes(n_Z_new, esd_min=5,   esd_max=2000)
>>>
>>> regen = build_cariaco_regenerators(
...     compute_K_s, compute_mu_max_maranon,
...     compute_I_max, compute_fish_kernel_vdl_joint,
... )
>>>
>>> new_setup = make_resized_setup(
...     loaded_setup=loaded,
...     model=model,
...     phyto_esd=P_new,
...     zoo_esd=Z_new,
...     regenerators=regen,
...     time=np.arange(0, 5000, 1),   # optional; otherwise inherited
... )

Design notes
------------
* All size-independent inputs (foreign refs, scalar parameters, labels)
  are auto-extracted from ``loaded_setup`` and copied verbatim. Add or
  remove a scalar parameter in the model and it gets transferred without
  any edit here.
* Size-dependent inputs (anything with a ``phyto``, ``zoo`` or ``full``
  dim) are recomputed from ``phyto_esd``/``zoo_esd`` via callables in
  the user-supplied ``regenerators`` dict. This is the one piece you
  must keep in sync with the model structure — and the validator below
  warns when it drifts.
* Framework variables (``Core__*``, ``Time__*``, ``Solver__*``) are
  skipped during extraction; they are re-supplied via
  ``xso.setup(solver=..., time=...)``.
* The validator emits ``warnings.warn`` (never raises) when:
    - the model expects an input nothing provides,
    - ``loaded_setup`` contains a var the model no longer expects,
    - a regenerator targets a var that is not in ``loaded_setup``.
"""

from __future__ import annotations

import warnings
from typing import Callable, Mapping, Optional

import numpy as np
import xso
import xarray as xr


# Dim names that mark a variable as size-dependent (must be regenerated,
# not inherited verbatim from loaded_setup).
SIZE_DIM_NAMES = ('phyto', 'zoo', 'full')

# Process-name prefixes that belong to the xso framework rather than
# to user components. These are skipped during extraction; their values
# are re-supplied through xso.setup(solver=..., time=...).
FRAMEWORK_PROCESSES = ('Core', 'Time', 'Solver')


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_size_dependent(da: xr.DataArray) -> bool:
    return any(d in SIZE_DIM_NAMES for d in da.dims)


def _is_framework(flat_name: str) -> bool:
    proc = flat_name.split('__', 1)[0]
    return proc in FRAMEWORK_PROCESSES


def _scalarize(da: xr.DataArray):
    """Convert a DataArray input value into a plain Python / numpy value
    suitable for handing to ``xso.setup(input_vars=...)``.
    """
    arr = da.values
    if arr.ndim == 0:
        v = arr.item()
        if isinstance(v, bytes):
            return v.decode()
        return v
    if arr.dtype.kind in ('U', 'O', 'S'):
        return arr.tolist()
    return np.asarray(arr)


def _model_expected_inputs(model) -> set[str]:
    """Return the set of ``'Process__var'`` strings the model expects.

    Tries several xsimlab/xso conventions and returns whatever it can
    find. Returns an empty set if nothing introspectable is available
    (in which case the validator will silently skip the missing/extra
    checks).
    """
    expected: set[str] = set()

    iv = getattr(model, 'input_vars', None)
    if iv is None:
        return expected

    # input_vars-dict shape: {proc: [vars]}
    if isinstance(iv, Mapping):
        for proc, varlist in iv.items():
            for var in varlist:
                expected.add(f'{proc}__{var}')
        return expected

    # iterable of (proc, var) tuples or 'proc__var' strings
    try:
        for entry in iv:
            if isinstance(entry, tuple) and len(entry) == 2:
                proc, var = entry
                expected.add(f'{proc}__{var}')
            elif isinstance(entry, str):
                if '__' in entry:
                    expected.add(entry)
                # else: bare process name — not a flat input id, ignore
    except TypeError:
        pass

    return expected


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_resized_setup(
    loaded_setup: xr.Dataset,
    model,
    phyto_esd: np.ndarray,
    zoo_esd: np.ndarray,
    regenerators: Mapping[str, Callable[[np.ndarray, np.ndarray], object]],
    *,
    solver: Optional[str] = None,
    time: Optional[np.ndarray] = None,
    output_vars=None,
    extra_overrides: Optional[Mapping[str, object]] = None,
    verbose: bool = True,
) -> xr.Dataset:
    """Build a fresh ``xso.setup`` mirroring ``loaded_setup`` with new size grids.

    Parameters
    ----------
    loaded_setup : xr.Dataset
        Previously-saved xso/xsimlab input dataset (e.g. opened from a
        ``.nc`` file written during a parameter scan).
    model : xsimlab.Model
        The model object the new setup will run against. Used both for
        ``xso.setup`` and for input-var validation.
    phyto_esd, zoo_esd : np.ndarray
        New size-class grids. Their lengths set the new ``n_P`` and
        ``n_Z``; their values become the new ``Phytoplankton__phyto_esd_index``
        / ``Zooplankton__zoo_esd_index``.
    regenerators : dict[str, callable]
        Map of ``'Component__var'`` to a function ``f(phyto_esd, zoo_esd)
        -> value`` for every size-dependent input. See
        ``build_cariaco_regenerators`` for an example.
    solver : str, optional
        Solver string passed to ``xso.setup``. If ``None``, inherited
        from ``loaded_setup['Core__solver_type']``.
    time : np.ndarray, optional
        Time array passed to ``xso.setup``. If ``None``, inherited from
        ``loaded_setup['Time__time_input']``.
    output_vars : dict | set | "ALL", optional
        Forwarded to ``xso.setup``.
    extra_overrides : dict, optional
        Flat-form ``{'Component__var': value}`` overrides that take
        precedence over both regenerators and the loaded-setup
        extraction. Useful for one-off experiments without editing
        either the model or the regenerator dict.
    verbose : bool
        Print a one-line summary of what was inherited / regenerated /
        flagged.

    Returns
    -------
    xr.Dataset
        Output of ``xso.setup(...)``, ready for ``.xsimlab.run()``.
    """
    extra_overrides = dict(extra_overrides or {})

    # ---- 1. Resolve solver / time, inheriting from loaded_setup if not given.
    if solver is None:
        if 'Core__solver_type' in loaded_setup:
            solver = _scalarize(loaded_setup['Core__solver_type'])
        else:
            raise ValueError(
                "solver= not provided and 'Core__solver_type' not in loaded_setup"
            )
    if time is None:
        if 'Time__time_input' in loaded_setup:
            time = np.asarray(loaded_setup['Time__time_input'].values)
        else:
            raise ValueError(
                "time= not provided and 'Time__time_input' not in loaded_setup"
            )

    # ---- 2. Run regenerators -> size-dependent values for the new grids.
    regen_values: dict[str, object] = {}
    for name, func in regenerators.items():
        try:
            regen_values[name] = func(phyto_esd, zoo_esd)
        except Exception as exc:
            raise RuntimeError(
                f"regenerator for '{name}' raised {type(exc).__name__}: {exc}"
            ) from exc

    # ---- 3. Auto-extract size-independent vars from loaded_setup.
    inherited: dict[str, object] = {}
    skipped_size_dep: list[str] = []
    for name, da in loaded_setup.data_vars.items():
        if _is_framework(name):
            continue
        if _is_size_dependent(da):
            skipped_size_dep.append(name)
            continue
        inherited[name] = _scalarize(da)

    # ---- 4. Merge: inherited < regenerators < extra_overrides
    input_vars: dict[str, object] = {}
    input_vars.update(inherited)
    input_vars.update(regen_values)
    input_vars.update(extra_overrides)

    # ---- 5. Validate against model expectations.
    expected = _model_expected_inputs(model)
    framework_inputs = {'Core__solver_type', 'Time__time_input'}
    provided = set(input_vars.keys()) | framework_inputs

    if expected:
        missing = expected - provided
        unexpected = provided - expected - framework_inputs
        for name in sorted(missing):
            warnings.warn(
                f"[resize] Model expects '{name}' but neither loaded_setup "
                f"nor regenerators provide it. The model may use a default "
                f"or fail at run time.",
                stacklevel=2,
            )
        for name in sorted(unexpected):
            warnings.warn(
                f"[resize] '{name}' was provided (from loaded_setup or "
                f"regenerators) but the model does not expect it. Possibly "
                f"stale — consider removing.",
                stacklevel=2,
            )
    else:
        if verbose:
            print("[resize] (model.input_vars was not introspectable; "
                  "skipping missing/extra-input validation)")

    # Regenerators that target vars not in loaded_setup are likely obsolete.
    loaded_names = set(loaded_setup.data_vars.keys())
    obsolete_regen = set(regenerators.keys()) - loaded_names
    for name in sorted(obsolete_regen):
        warnings.warn(
            f"[resize] Regenerator listed for '{name}' but loaded_setup "
            f"does not contain it. The regenerator may be obsolete.",
            stacklevel=2,
        )

    if verbose:
        print(
            f"[resize] inherited={len(inherited)} "
            f"regenerated={len(regen_values)} "
            f"skipped_size_dep={len(skipped_size_dep)} "
            f"overrides={len(extra_overrides)}"
        )

    # ---- 6. Build the new xso setup.
    return xso.setup(
        solver=solver,
        model=model,
        time=time,
        input_vars=input_vars,
        output_vars=output_vars,
    )


# ---------------------------------------------------------------------------
# Convenience: regenerator dict for the cariaco NPxZxF model.
# ---------------------------------------------------------------------------

def build_cariaco_regenerators(
    compute_K_s,
    compute_mu_max_maranon,
    compute_I_max,
    compute_fish_kernel_vdl_joint,
    *,
    phyto_init_value: float = 1e-3,
    zoo_init_value: float = 1e-4,
    mort_phyto_factor: float = 0.1,
):
    """Build the size-dependent-regenerator dict for the cariaco NPxZxF model.

    The allometric functions are passed in (rather than imported) to
    keep this file decoupled from the cariaco model module — that way
    you can edit either side without circular-import headaches.

    Parameters
    ----------
    compute_K_s, compute_mu_max_maranon, compute_I_max,
    compute_fish_kernel_vdl_joint :
        The allometric helpers from ``cariaco_ssm_setup``.
    phyto_init_value, zoo_init_value : float
        Uniform initial-condition values for the new size classes.
        Defaults match the current ``cariaco_ssm_setup`` convention
        (1e-3 mmol N m-3 for phyto, 1e-4 for zoo).
    mort_phyto_factor : float
        ``PhytoMortality__rate = mort_phyto_factor * mu_max(P)``. The
        cariaco script uses 0.1; expose it here so you can override
        without rewriting the regenerator.
    """
    def _kernel_P(P, Z):
        return compute_fish_kernel_vdl_joint(P, Z)[0]
    def _kernel_Z(P, Z):
        return compute_fish_kernel_vdl_joint(P, Z)[1]

    return {
        'Phytoplankton__biomass_init':
            lambda P, Z: np.full(len(P), phyto_init_value),
        'Phytoplankton__phyto_esd_index':
            lambda P, Z: np.asarray(P),
        'Zooplankton__biomass_init':
            lambda P, Z: np.full(len(Z), zoo_init_value),
        'Zooplankton__zoo_esd_index':
            lambda P, Z: np.asarray(Z),
        'Growth__halfsat':
            lambda P, Z: compute_K_s(P),
        'Growth__mu_max':
            lambda P, Z: compute_mu_max_maranon(P),
        'Grazing__Imax':
            lambda P, Z: compute_I_max(Z),
        'PhytoMortality__rate':
            lambda P, Z: mort_phyto_factor * compute_mu_max_maranon(P),
        'FishGrazing__kernel_P': _kernel_P,
        'FishGrazing__kernel_Z': _kernel_Z,
    }


# ---------------------------------------------------------------------------
# Snapshot a single parscan cell as a reusable setup (and optionally save it).
# ---------------------------------------------------------------------------

def save_cell_as_setup(
    best_cell: xr.Dataset,
    model,
    path: Optional[str] = None,
    *,
    solver: str = 'stability',
    time=(0, 1),
    verbose: bool = True,
) -> xr.Dataset:
    """Build a fresh ``xso.setup`` from a single parscan cell's inputs.

    Walks the model's declared input variables and, for each one, pulls
    the value from ``best_cell.data_vars`` (or, failing that,
    ``best_cell.coords``). Inputs missing from ``best_cell`` are skipped
    silently — useful when ``best_cell`` is a sliced parscan result
    that doesn't carry every framework variable.

    Unlike :func:`make_resized_setup`, this does **not** change the
    number of size classes — it just snapshots the cell verbatim. Use
    it to persist a single parameter combination (e.g. the best-fitting
    cell of a sweep) as a self-contained setup that can be reloaded
    and rerun with a different solver later.

    Parameters
    ----------
    best_cell : xr.Dataset
        Single cell (or any Dataset) whose ``data_vars`` / ``coords``
        carry input values keyed by the flat ``'Process__var'``
        convention.
    model : xsimlab.Model
        Model the new setup will run against.
    path : str, optional
        If given, the resulting setup is written to this path via
        ``setup.to_netcdf(path)``. If ``None`` (default), nothing is
        written and only the in-memory setup is returned.
    solver : str
        Solver string for ``xso.setup``. Default ``'stability'``.
    time : array-like
        Time argument for ``xso.setup``. Default ``(0, 1)`` — appropriate
        for the stability solver. Pass an explicit ``np.arange(...)`` for
        ``solve_ivp``.
    verbose : bool
        Print a one-line summary of what was extracted and where it was
        written.

    Returns
    -------
    xr.Dataset
        Output of ``xso.setup(...)``, ready for ``.xsimlab.run()``.

    Examples
    --------
    >>> save_cell_as_setup(best_cell, model, 'best_cell_stability.nc',
    ...                    solver='stability', time=[0, 1])
    >>> save_cell_as_setup(best_cell, model, 'best_cell_ivp.nc',
    ...                    solver='solve_ivp', time=np.arange(0, 5000, 1))
    """
    input_vars_dict: dict[str, dict[str, object]] = {}
    n_extracted = 0
    n_missing = 0

    iv = getattr(model, 'input_vars', None)
    if iv is None:
        raise ValueError(
            "model has no .input_vars attribute — cannot determine which "
            "inputs to snapshot."
        )

    for entry in iv:
        # Tolerate both (proc, var) tuple form and 'proc__var' string form.
        if isinstance(entry, tuple) and len(entry) == 2:
            proc, var = entry
        elif isinstance(entry, str) and '__' in entry:
            proc, var = entry.split('__', 1)
        else:
            continue

        if proc in FRAMEWORK_PROCESSES:
            # Solver / time get re-supplied via xso.setup kwargs.
            continue

        flat = f"{proc}__{var}"
        if flat in best_cell.data_vars:
            da = best_cell[flat]
        elif flat in best_cell.coords:
            da = best_cell.coords[flat]
        else:
            n_missing += 1
            continue

        input_vars_dict.setdefault(proc, {})[var] = _scalarize(da)
        n_extracted += 1

    setup = xso.setup(
        solver=solver,
        model=model,
        time=np.asarray(time),
        input_vars=input_vars_dict,
    )

    if path is not None:
        setup.to_netcdf(path)

    if verbose:
        suffix = f" -> {path}" if path else " (not written to disk)"
        print(
            f"[snapshot] extracted={n_extracted} missing={n_missing} "
            f"solver={solver!r}{suffix}"
        )

    return setup
