"""
N–P(–Z) spectral model — XSO setups
===================================
Step 1 N–P diagnostic (four variants, MS3 layered construction plan
2026-05-18) plus the NPZ extension for the Ago-meeting bridge figure
(six new NPZ variants, 2026-05-19).

NP variants (Step 1, existing)
------------------------------
Each variant exposes both an IVP setup (`solve_ivp`, 5000 d) and a
stability setup (`fsolve` Jacobian eigenvalue analysis, length-2 time
array) for the IVP-then-stability parscan workflow used in
`run_1d_scan_spectrum.py`.

* `closed_const` — closed (N_T conserved), size-independent Λ
* `closed_allom` — closed,                size-dependent Λ(s) = Λ₀·s^(-0.25)
* `open_const`   — open chemostat (F_N supply + λ-dilution on N & P),
                   size-independent Λ
* `open_allom`   — open chemostat,        size-dependent Λ

NPZ variants (Ago bridge, 2026-05-19)
-------------------------------------
Six new NPZ open-chemostat setups, all built on the same Hansen-1997
zoo parameter defaults; structural variant is set entirely by the
`phiPZ` matrix passed to the grazing component. Grazing type II vs III
is the swap of the grazing component class itself (two model schemas).

* `npz_typeII_matched`,  `npz_typeII_herb`,  `npz_typeII_omni`
* `npz_typeIII_matched`, `npz_typeIII_herb`, `npz_typeIII_omni`

No stability setups are exposed for the NPZ variants — the bridge
figure is time-series only.

Module-level exports
--------------------
* Models:           `model_closed`, `model_open`,
                    `model_npz_typeII`, `model_npz_typeIII`
* NP IVP setups:    `model_setup_closed_const`, `model_setup_closed_allom`,
                    `model_setup_open_const`,   `model_setup_open_allom`
* NP stability:     same names with `_stability` suffix
* NPZ IVP setups:   `model_setup_npz_typeII_matched`,
                    `model_setup_npz_typeII_herb`,
                    `model_setup_npz_typeII_omni`,
                    `model_setup_npz_typeIII_matched`,
                    `model_setup_npz_typeIII_herb`,
                    `model_setup_npz_typeIII_omni`
* Helpers:          `phyto_esd`, `zoo_esd`, `n_classes`, `n_zoo`,
                    `generate_size_classes`, `compute_phiPZ`,
                    `phiPZ_matched`, `phiPZ_herb`, `phiPZ_omni`,
                    `avg_tail` (re-export)

Scan axes
---------
* Closed variants: `'Nutrient__value_init'` (sets N_T at t = 0; closed
  dynamics preserve `N_T = N + ΣP` thereafter — see MS3 Background §2026-05-18).
* Open variants:   `'FN_Forcing__value'` (the F_N supply rate; at steady
  state `N_T = F_N / (λ · d_e)` per Eq. F-N-Nstar-bridge in the LaTeX).

Size grids
----------
Phyto: 40 log-spaced classes from 0.2 µm to 200 µm (Sieburth-compatible
range, ~13 classes per decade — Banas 2011 ballpark per Benny's pref).
Change via module-level `N_CLASSES`.

Zoo: 40 log-spaced classes constructed as `ZOO_ESD_RATIO · phyto_esd`,
i.e. 2 µm to 2000 µm at `ZOO_ESD_RATIO = 10` (matched to `THETA_OPT = 10`
so each predator Z_j has its kernel peak exactly on the matched prey
P_j). Same `n_zoo = n_classes` keeps the pairwise structure clean and
keeps `phiPZ` a square block.
"""

import numpy as np
import xso

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from xso.parscans import avg_tail  # re-export for parscan worker discovery

from np_comps import (
    Nutrient,
    PhytoSizeSpectrum,
    ConstantExternalNutrient,
    LinearForcingInput,
    MonodGrowth_NP,
    PhytoLinearLoss_recycled,
    ChemostatDilution_Scalar,
    ChemostatDilution_PhytoDim,
    # NPZ extension (2026-05-19, updated 2026-05-19 with quadratic closure)
    ZooSizeSpectrum,
    ZooQuadraticLoss_recycled,    # Banas 2011 / MS3 style — buffers Type II overshoot
    ChemostatDilution_ZooDim,
    GrazingMatrix_TypeII,
    GrazingMatrix_TypeIII,
    GrazingRouter,
)


# =============================================================================
# TOP-LEVEL GRID / PARAMETER CONSTANTS
# =============================================================================
N_CLASSES = 40
ESD_MIN   = 0.2
ESD_MAX   = 200.0

# IVP run duration. Restored 2026-05-18 from 2000 d to 5000 d after the
# LSODA + relaxed-atol switch (see IVP_SOLVER_KWARGS below) made each
# cell sub-second; 2000 d was a workaround for RK45 stiffness slowness
# that is no longer relevant. At 5000 d the slowest-decaying loser
# classes have 5+ e-folds below their initial value, so the tail-mean
# represents a more fully resolved steady state.
IVP_TIME_END   = 5000
ivp_time_array = np.arange(0, IVP_TIME_END, 1)

# scipy.integrate.solve_ivp method + tolerances (passed via XSO's
# solver_kwargs hook added 2026-05-18).
#
# LSODA auto-switches between Adams (non-stiff) and BDF (stiff) and
# is the safe default for size-spectrum systems with mixed regimes.
#
# Tolerances loosened from XSO's RK45 defaults (atol=1e-9, rtol=1e-6).
# In competitive-exclusion N-P dynamics, ~40 loser classes spend long
# stretches at <1e-6 mmol N m-3 — biologically noise — and tight atol
# forces the step controller to spend most of its budget tracking
# those values faithfully. The relaxed values match the physically
# meaningful precision: any concentration below ~1e-6 is rounding,
# and 1e-4 relative is finer than every comparison we make in the
# manuscript metrics. Final-state comparison after this change should
# still match RK45/default-tol within fractions of a percent.
IVP_SOLVER_KWARGS = {
    'method': 'LSODA', 'atol': 1e-6, 'rtol': 1e-4,
    # 2026-05-19: configurable instability-event threshold (XSO update of
    # the same date). Loosens XSO's default `-1e-6` negative-state floor
    # to `-1e-3`. Rationale: the Type II NPZ setups have legitimate
    # transient predator-prey overshoots in the `-1e-6` to `-1e-5` range
    # — well within physical noise for biomass values that equilibrate
    # at 0.1–1 mmol N m⁻³ — and the default floor was terminating those
    # runs as if they were genuinely unstable. The `-1e-3` choice gives
    # ~500× head-room over the observed transient excursions and is
    # still 3 orders of magnitude below typical equilibrium biomass,
    # so it catches actual blow-ups without flagging Lotka-Volterra
    # bottoming-out. Applies uniformly to NP and NPZ setups; NP runs
    # don't produce values anywhere near this floor so the change is
    # invisible there.
    #
    # History: an earlier `IVP_SOLVER_KWARGS_STRICT` variant (atol=1e-9 +
    # max_step=1.0) and the switch to Banas-style distributed quadratic
    # closure (`ZooQuadraticLoss_recycled`) both predate this fix. The
    # quadratic closure is kept on scientific grounds (Banas 2011 / MS3
    # structural alignment) even though the threshold change alone would
    # also have solved the symptom — the quadratic closure is the right
    # model, the loose threshold is the right safety net.
    'instability_neg_threshold': -1e-3,
}


def generate_size_classes(n=None, esd_min=None, esd_max=None):
    """Log-spaced phyto ESD grid. Defaults pulled from module-level constants."""
    if n is None:        n        = N_CLASSES
    if esd_min is None:  esd_min  = ESD_MIN
    if esd_max is None:  esd_max  = ESD_MAX
    return np.logspace(np.log10(esd_min), np.log10(esd_max), n)


phyto_esd = generate_size_classes()
n_classes = len(phyto_esd)


# =============================================================================
# ALLOMETRIES — Taniguchi 2014 Table 1 (verbatim for μ, k_s)
# =============================================================================
mu_max_arr = 1.36 * phyto_esd ** (-0.16)
k_s_arr    = 0.33 * phyto_esd ** ( 0.48)

# Two Λ variants:
#   const — size-independent
#   allom — Λ(s) = Λ₀ · s^(-0.25), the MTE prediction Taniguchi explicitly
#           rejected; included here to show that even with this classic
#           size-dependence the system still collapses to a single surviving
#           class. Anchored so Λ(1 µm) = the const value.
#
# 2026-05-18: prefactor raised from Taniguchi's 0.0015 d⁻¹ to a more
# conventional 0.1 d⁻¹ (Stock 2008 background phyto mortality; Banas
# 2011 m_P ≈ 0.1·μ). The Taniguchi value produces decay timescales of
# ~20 000 d for classes near the competitive-exclusion winner, far past
# our 2000 d IVP window; with 0.1 d⁻¹ adjacent-class decay timescales
# are ~330 d so loser classes are 99 %+ decayed by t = 2000. Qualitative
# competitive-exclusion result is unchanged; visualisation becomes clean.
LAMBDA_CONST     = 0.1
LAMBDA_ALLOM_EXP = -0.25
lambda_arr_const = np.full(n_classes, LAMBDA_CONST)
lambda_arr_allom = LAMBDA_CONST * phyto_esd ** LAMBDA_ALLOM_EXP


# =============================================================================
# OPEN-VARIANT PARAMETERS — F_N supply + chemostat dilution
# =============================================================================
D_E              = 50.0     # m, surface-box depth
DILUTION_RATE    = 0.05     # d⁻¹, single λ applied to both N and P
F_N_DEFAULT      = 5.0      # mmol N m⁻² d⁻¹ (scanned at run time)


# =============================================================================
# INITIAL CONDITIONS — N AND P
# =============================================================================
P_INIT_PER_CLASS = 1e-3                 # mmol N m⁻³, small uniform seed
phyto_init       = np.full(n_classes, P_INIT_PER_CLASS)
N_INIT_CLOSED    = 15.0                 # closed: this sets N_T; scanned at run time
N_INIT_OPEN      = 1.0                  # open: irrelevant to SS, just a seed


# =============================================================================
# NPZ EXTENSION (2026-05-19) — ZOO GRID
# =============================================================================
# Zoo grid: same N_CLASSES count as phyto, shifted up by ZOO_ESD_RATIO so
# each predator Z_j sits exactly at theta_opt × prey P_j. With the default
# (phyto 0.2–200 µm, ratio 10) the zoo grid spans 2–2000 µm — meso-zoo
# range without the >2 mm metazoan tail.
ZOO_ESD_RATIO = 10.0
zoo_esd       = ZOO_ESD_RATIO * phyto_esd
n_zoo         = len(zoo_esd)


# =============================================================================
# NPZ EXTENSION — ZOO PARAMETERS
# =============================================================================
# Modern (Hansen / Stock / Banas)-derived defaults, with a deliberate
# departure on I_max documented below.
#   - I_max = 3.0 d⁻¹ UNIFORM (Banas 2011 convention). The first NPZ
#     revision (2026-05-19 morning) used Hansen-allometric
#     I_max(s) = 26·s^(-0.48), which at the smallest predator class
#     (Z at 2 µm with theta_opt=10) gives I_max = 18.65 d⁻¹ and
#     γ·I_max = 5.6 d⁻¹ — 3.2× the matched-prey μ_max(0.2 µm) = 1.73 d⁻¹.
#     That rate mismatch produced violent predator-prey overshoots
#     (transient prey excursions to ~-1e-3) that tripped the XSO
#     instability event even with Banas-style quadratic closure on
#     Z and the relaxed instability_neg_threshold = -1e-3.
#     Reverting to a single Banas-style uniform value removes the
#     extreme low-end Imax without inventing a custom truncation
#     or rescaling of Hansen's allometric form. Side benefit for the
#     bridge figure: the matched/herb/omni comparison is now a pure
#     kernel-structure contrast, no longer confounded with the
#     "smallest predator is also the most aggressive" effect of the
#     allometric form. Hansen's full dataset DOES find a significant
#     trend on I_max (b ≈ -0.48); we are choosing Banas's parsimonious
#     uniform reading for the demo, not contradicting Hansen.
#   - K_sZ = 3.0 mmol N m⁻³ UNIFORM (Hansen 1997 full dataset; settled
#     per `Current Status Briefing.md` "Items closed" 2026-05-09 —
#     "Size-dependent K_sZ as a slope lever: off the table").
#   - Γ = 0.3 scalar (Stock central; uniform per Correction; Hansen
#     full dataset finds no significant size trend on assimilation).
#   - Z closure: distributed quadratic m_Z · Z_j · ΣZ (Banas 2011 / MS3
#     ZooQuadraticMortality_toD). Kept after the linear→quadratic swap
#     even though uniform I_max alone would have stabilised the runs —
#     the quadratic closure is the right structural form per Edwards
#     & Yool 2000 and matches MS3-as-built, so it stays on scientific
#     grounds independent of the I_max change. m_Z = 0.1 is the MS3
#     setup value (Model Equations §5, cariaco_ssm_setup:211) — low
#     end of the Banas defensible range (analytical estimate 0.26;
#     paper 1.0).
IMAX_UNIFORM = 3.0                       # Banas 2011 uniform g_max [d⁻¹]
Imax_arr     = np.full(n_zoo, IMAX_UNIFORM)

KSZ_VAL      = 3.0
GAMMA_VAL    = 0.3

M_Z_VAL      = 0.1                       # Banas-style quadratic closure coefficient
                                         # [(mmol N m-3)^-1 d^-1]


# =============================================================================
# NPZ EXTENSION — GRAZING KERNEL (3 STRUCTURAL VARIANTS)
# =============================================================================
# MS3 log-normal kernel (cariaco_ssm_comps.compute_grazing_kernel) replicated
# inline here to keep np_comps / np_setups self-contained. The kernel is
# computed once at setup time and supplied as a numeric (n_P+n_Z, n_Z) array
# to the GrazingMatrix component's `phiPZ` parameter. Mode dispatch produces
# the three structural variants from the same kernel:
#
#   'matched' — diagonal identity on P-block, zero on Z-block (η=1, P-only).
#               Predator j eats only the matched P_j (Taniguchi M1 structure,
#               extended here to theta_opt > 1 by the zoo-grid shift).
#   'herb'    — log-normal on P-block, zero on Z-block (η>1, P-only).
#               Multi-prey herbivorous (Taniguchi Model 2 / MS3 P-only).
#   'omni'    — log-normal on full (P+Z) prey, Z-on-self diagonal zeroed
#               (Taniguchi Model 3 / MS3-as-built omnivory).
THETA_OPT = 10.0    # predator:prey ESD ratio (matched by ZOO_ESD_RATIO above)
SIGMA_LOG = 0.25    # log10-space kernel width (MS3 default)


def compute_phiPZ(phyto_esd, zoo_esd, mode,
                  theta_opt=THETA_OPT, sigma_log=SIGMA_LOG):
    """Construct the (n_P+n_Z, n_Z) feeding-preference matrix.

    Parameters
    ----------
    phyto_esd, zoo_esd : 1D arrays of ESDs in µm.
    mode : {'matched', 'herb', 'omni'}
        Structural variant — see module docstring.
    theta_opt, sigma_log : kernel peak ratio and log10-space width.

    Returns
    -------
    phiPZ : ndarray, shape (n_P + n_Z, n_Z)
        Prey-by-predator preference matrix. Rows 0..n_P-1 are P prey;
        rows n_P..n_P+n_Z-1 are Z prey. Z-on-self diagonal is zeroed
        in 'omni' mode to prevent within-class cannibalism (cf.
        cariaco_ssm_comps.compute_grazing_kernel:35-36).
    """
    n_P = len(phyto_esd)
    n_Z = len(zoo_esd)
    prey_esd = np.concatenate([phyto_esd, zoo_esd])
    log_ratio = np.log10(zoo_esd[None, :] / prey_esd[:, None])
    log_theta = np.log10(theta_opt)
    kernel = np.exp(-((log_ratio - log_theta) ** 2) / (2 * sigma_log ** 2))

    phiPZ = np.zeros((n_P + n_Z, n_Z))
    if mode == 'matched':
        # With zoo_esd = theta_opt * phyto_esd, the kernel peak (log_ratio =
        # log_theta) lands exactly on the matched class i = j in the P-block.
        # The 'matched' variant collapses the kernel to a true delta at that
        # class, ignoring sigma_log entirely.
        for j in range(n_Z):
            phiPZ[j, j] = 1.0
    elif mode == 'herb':
        phiPZ[:n_P, :] = kernel[:n_P, :]
    elif mode == 'omni':
        phiPZ[:] = kernel
        for j in range(n_Z):
            phiPZ[n_P + j, j] = 0.0
    else:
        raise ValueError(f"Unknown phiPZ mode: {mode!r}")
    return phiPZ


phiPZ_matched = compute_phiPZ(phyto_esd, zoo_esd, 'matched')
phiPZ_herb    = compute_phiPZ(phyto_esd, zoo_esd, 'herb')
phiPZ_omni    = compute_phiPZ(phyto_esd, zoo_esd, 'omni')


# =============================================================================
# NPZ EXTENSION — Z INITIAL CONDITIONS
# =============================================================================
Z_INIT_PER_CLASS = 1e-3                 # mmol N m⁻³, same small seed as P
zoo_init         = np.full(n_zoo, Z_INIT_PER_CLASS)


# =============================================================================
# MODELS
# =============================================================================
model_closed = xso.create({
    'Nutrient':      Nutrient,
    'Phytoplankton': PhytoSizeSpectrum,
    'Growth':        MonodGrowth_NP,
    'PhytoLoss':     PhytoLinearLoss_recycled,
})

model_open = xso.create({
    'Nutrient':      Nutrient,
    'Phytoplankton': PhytoSizeSpectrum,
    'Growth':        MonodGrowth_NP,
    'PhytoLoss':     PhytoLinearLoss_recycled,
    'FN_Forcing':    ConstantExternalNutrient,
    'FN_Input':      LinearForcingInput,
    'DilutionN':     ChemostatDilution_Scalar,
    'DilutionP':     ChemostatDilution_PhytoDim,
})


# =============================================================================
# SETUP — CLOSED, SIZE-INDEPENDENT Λ
# =============================================================================
model_setup_closed_const = xso.setup(
    solver='solve_ivp', model=model_closed,
    time=ivp_time_array,
    input_vars={
        'Nutrient':      {'value_label': 'N', 'value_init': N_INIT_CLOSED},
        'Phytoplankton': {'biomass_label': 'P', 'biomass_init': phyto_init,
                          'phyto_esd_index': phyto_esd.tolist(),
                          'phyto_esd_label': 'phyto_esd'},
        'Growth':        {'resource': 'N', 'consumer': 'P',
                          'mu_max': mu_max_arr, 'halfsat': k_s_arr},
        'PhytoLoss':     {'population': 'P', 'nutrient': 'N',
                          'rate': lambda_arr_const},
    },
    solver_kwargs=IVP_SOLVER_KWARGS,
)

model_setup_closed_const_stability = xso.setup(
    solver='stability', model=model_closed,
    time=[0, 1],
    input_vars={
        'Nutrient':      {'value_label': 'N', 'value_init': N_INIT_CLOSED},
        'Phytoplankton': {'biomass_label': 'P', 'biomass_init': phyto_init,
                          'phyto_esd_index': phyto_esd.tolist(),
                          'phyto_esd_label': 'phyto_esd'},
        'Growth':        {'resource': 'N', 'consumer': 'P',
                          'mu_max': mu_max_arr, 'halfsat': k_s_arr},
        'PhytoLoss':     {'population': 'P', 'nutrient': 'N',
                          'rate': lambda_arr_const},
    },
)


# =============================================================================
# SETUP — CLOSED, SIZE-DEPENDENT Λ(s) = Λ₀ · s^(-0.25)
# =============================================================================
model_setup_closed_allom = xso.setup(
    solver='solve_ivp', model=model_closed,
    time=ivp_time_array,
    input_vars={
        'Nutrient':      {'value_label': 'N', 'value_init': N_INIT_CLOSED},
        'Phytoplankton': {'biomass_label': 'P', 'biomass_init': phyto_init,
                          'phyto_esd_index': phyto_esd.tolist(),
                          'phyto_esd_label': 'phyto_esd'},
        'Growth':        {'resource': 'N', 'consumer': 'P',
                          'mu_max': mu_max_arr, 'halfsat': k_s_arr},
        'PhytoLoss':     {'population': 'P', 'nutrient': 'N',
                          'rate': lambda_arr_allom},
    },
    solver_kwargs=IVP_SOLVER_KWARGS,
)

model_setup_closed_allom_stability = xso.setup(
    solver='stability', model=model_closed,
    time=[0, 1],
    input_vars={
        'Nutrient':      {'value_label': 'N', 'value_init': N_INIT_CLOSED},
        'Phytoplankton': {'biomass_label': 'P', 'biomass_init': phyto_init,
                          'phyto_esd_index': phyto_esd.tolist(),
                          'phyto_esd_label': 'phyto_esd'},
        'Growth':        {'resource': 'N', 'consumer': 'P',
                          'mu_max': mu_max_arr, 'halfsat': k_s_arr},
        'PhytoLoss':     {'population': 'P', 'nutrient': 'N',
                          'rate': lambda_arr_allom},
    },
)


# =============================================================================
# SETUP — OPEN CHEMOSTAT, SIZE-INDEPENDENT Λ
# =============================================================================
model_setup_open_const = xso.setup(
    solver='solve_ivp', model=model_open,
    time=ivp_time_array,
    input_vars={
        'Nutrient':      {'value_label': 'N', 'value_init': N_INIT_OPEN},
        'Phytoplankton': {'biomass_label': 'P', 'biomass_init': phyto_init,
                          'phyto_esd_index': phyto_esd.tolist(),
                          'phyto_esd_label': 'phyto_esd'},
        'Growth':        {'resource': 'N', 'consumer': 'P',
                          'mu_max': mu_max_arr, 'halfsat': k_s_arr},
        'PhytoLoss':     {'population': 'P', 'nutrient': 'N',
                          'rate': lambda_arr_const},
        'FN_Forcing':    {'forcing_label': 'FN_supply', 'value': F_N_DEFAULT},
        'FN_Input':      {'var': 'N', 'forcing': 'FN_supply',
                          'rate': 1.0 / D_E},
        'DilutionN':     {'var': 'N', 'rate': DILUTION_RATE},
        'DilutionP':     {'var': 'P', 'rate': DILUTION_RATE},
    },
    solver_kwargs=IVP_SOLVER_KWARGS,
)

model_setup_open_const_stability = xso.setup(
    solver='stability', model=model_open,
    time=[0, 1],
    input_vars={
        'Nutrient':      {'value_label': 'N', 'value_init': N_INIT_OPEN},
        'Phytoplankton': {'biomass_label': 'P', 'biomass_init': phyto_init,
                          'phyto_esd_index': phyto_esd.tolist(),
                          'phyto_esd_label': 'phyto_esd'},
        'Growth':        {'resource': 'N', 'consumer': 'P',
                          'mu_max': mu_max_arr, 'halfsat': k_s_arr},
        'PhytoLoss':     {'population': 'P', 'nutrient': 'N',
                          'rate': lambda_arr_const},
        'FN_Forcing':    {'forcing_label': 'FN_supply', 'value': F_N_DEFAULT},
        'FN_Input':      {'var': 'N', 'forcing': 'FN_supply',
                          'rate': 1.0 / D_E},
        'DilutionN':     {'var': 'N', 'rate': DILUTION_RATE},
        'DilutionP':     {'var': 'P', 'rate': DILUTION_RATE},
    },
)


# =============================================================================
# SETUP — OPEN CHEMOSTAT, SIZE-DEPENDENT Λ(s) = Λ₀ · s^(-0.25)
# =============================================================================
model_setup_open_allom = xso.setup(
    solver='solve_ivp', model=model_open,
    time=ivp_time_array,
    input_vars={
        'Nutrient':      {'value_label': 'N', 'value_init': N_INIT_OPEN},
        'Phytoplankton': {'biomass_label': 'P', 'biomass_init': phyto_init,
                          'phyto_esd_index': phyto_esd.tolist(),
                          'phyto_esd_label': 'phyto_esd'},
        'Growth':        {'resource': 'N', 'consumer': 'P',
                          'mu_max': mu_max_arr, 'halfsat': k_s_arr},
        'PhytoLoss':     {'population': 'P', 'nutrient': 'N',
                          'rate': lambda_arr_allom},
        'FN_Forcing':    {'forcing_label': 'FN_supply', 'value': F_N_DEFAULT},
        'FN_Input':      {'var': 'N', 'forcing': 'FN_supply',
                          'rate': 1.0 / D_E},
        'DilutionN':     {'var': 'N', 'rate': DILUTION_RATE},
        'DilutionP':     {'var': 'P', 'rate': DILUTION_RATE},
    },
    solver_kwargs=IVP_SOLVER_KWARGS,
)

model_setup_open_allom_stability = xso.setup(
    solver='stability', model=model_open,
    time=[0, 1],
    input_vars={
        'Nutrient':      {'value_label': 'N', 'value_init': N_INIT_OPEN},
        'Phytoplankton': {'biomass_label': 'P', 'biomass_init': phyto_init,
                          'phyto_esd_index': phyto_esd.tolist(),
                          'phyto_esd_label': 'phyto_esd'},
        'Growth':        {'resource': 'N', 'consumer': 'P',
                          'mu_max': mu_max_arr, 'halfsat': k_s_arr},
        'PhytoLoss':     {'population': 'P', 'nutrient': 'N',
                          'rate': lambda_arr_allom},
        'FN_Forcing':    {'forcing_label': 'FN_supply', 'value': F_N_DEFAULT},
        'FN_Input':      {'var': 'N', 'forcing': 'FN_supply',
                          'rate': 1.0 / D_E},
        'DilutionN':     {'var': 'N', 'rate': DILUTION_RATE},
        'DilutionP':     {'var': 'P', 'rate': DILUTION_RATE},
    },
)


# =============================================================================
# NPZ EXTENSION (2026-05-19) — MODELS
# =============================================================================
# Two NPZ model schemas, differing only in the grazing component class.
# Each is the open NP chemostat (model_open above) with three additions:
#   - Zooplankton state variable on dim 'zoo'
#   - ZooLoss (linear Δ on Z, full sum-recycle to N)
#   - DilutionZ (chemostat dilution on Z at the same λ as N and P)
# plus the Grazing component (TypeII or TypeIII).
#
# In both models, the Z state variable label 'Z' is wired to TWO foreign
# refs on the Grazing component (`prey_Z` and `predator`) — Z-as-prey
# (negative flux, omnivory only) and Z-as-predator (positive flux). XSO
# supports multi-role same-label wiring (cf. MS3 GGE_Full_withD).

model_npz_typeII = xso.create({
    'Nutrient':      Nutrient,
    'Phytoplankton': PhytoSizeSpectrum,
    'Zooplankton':   ZooSizeSpectrum,
    'Growth':        MonodGrowth_NP,
    'PhytoLoss':     PhytoLinearLoss_recycled,
    'ZooLoss':       ZooQuadraticLoss_recycled,
    'Grazing':       GrazingMatrix_TypeII,
    'GrazingRouter': GrazingRouter,
    'FN_Forcing':    ConstantExternalNutrient,
    'FN_Input':      LinearForcingInput,
    'DilutionN':     ChemostatDilution_Scalar,
    'DilutionP':     ChemostatDilution_PhytoDim,
    'DilutionZ':     ChemostatDilution_ZooDim,
})

model_npz_typeIII = xso.create({
    'Nutrient':      Nutrient,
    'Phytoplankton': PhytoSizeSpectrum,
    'Zooplankton':   ZooSizeSpectrum,
    'Growth':        MonodGrowth_NP,
    'PhytoLoss':     PhytoLinearLoss_recycled,
    'ZooLoss':       ZooQuadraticLoss_recycled,
    'Grazing':       GrazingMatrix_TypeIII,
    'GrazingRouter': GrazingRouter,
    'FN_Forcing':    ConstantExternalNutrient,
    'FN_Input':      LinearForcingInput,
    'DilutionN':     ChemostatDilution_Scalar,
    'DilutionP':     ChemostatDilution_PhytoDim,
    'DilutionZ':     ChemostatDilution_ZooDim,
})


# =============================================================================
# NPZ EXTENSION — SETUPS (6 = 2 grazing types × 3 phiPZ structural variants)
# =============================================================================
# Per Benny's explicit-setup-over-helper preference, each setup is spelled
# out fully. The six differ only in (a) which `model_npz_typeXX` they
# target and (b) which `phiPZ_*` matrix they pass to the Grazing component.
# All other input_vars are identical (Hansen Imax, K_sZ = 3.0, Γ = 0.3,
# Δ = 0.1, λ = DILUTION_RATE, F_N = F_N_DEFAULT).

# ---------------------------------------------------------------------------
# Type II × matched (η=1, P-only — analogue of Taniguchi Model 1 at theta=10)
# ---------------------------------------------------------------------------
model_setup_npz_typeII_matched = xso.setup(
    solver='solve_ivp', model=model_npz_typeII,
    time=ivp_time_array,
    input_vars={
        'Nutrient':      {'value_label': 'N', 'value_init': N_INIT_OPEN},
        'Phytoplankton': {'biomass_label': 'P', 'biomass_init': phyto_init,
                          'phyto_esd_index': phyto_esd.tolist(),
                          'phyto_esd_label': 'phyto_esd'},
        'Zooplankton':   {'biomass_label': 'Z', 'biomass_init': zoo_init,
                          'zoo_esd_index': zoo_esd.tolist(),
                          'zoo_esd_label': 'zoo_esd'},
        'Growth':        {'resource': 'N', 'consumer': 'P',
                          'mu_max': mu_max_arr, 'halfsat': k_s_arr},
        'PhytoLoss':     {'population': 'P', 'nutrient': 'N',
                          'rate': lambda_arr_const},
        'ZooLoss':       {'population': 'Z', 'nutrient': 'N',
                          'rate': M_Z_VAL},
        'Grazing':       {'resource': 'P', 'consumer': 'Z',
                          'phiPZ': phiPZ_matched,
                          'Imax': Imax_arr, 'KsZ': KSZ_VAL},
        'GrazingRouter': {'grazed_phyto': 'P', 'grazed_zoo': 'Z',
                          'assimilated_consumer': 'Z',
                          'excreted_nutrient': 'N',
                          'gamma': GAMMA_VAL},
        'FN_Forcing':    {'forcing_label': 'FN_supply', 'value': F_N_DEFAULT},
        'FN_Input':      {'var': 'N', 'forcing': 'FN_supply',
                          'rate': 1.0 / D_E},
        'DilutionN':     {'var': 'N', 'rate': DILUTION_RATE},
        'DilutionP':     {'var': 'P', 'rate': DILUTION_RATE},
        'DilutionZ':     {'var': 'Z', 'rate': DILUTION_RATE},
    },
    solver_kwargs=IVP_SOLVER_KWARGS,
)

# ---------------------------------------------------------------------------
# Type II × herb (multi-prey log-normal kernel on P only — Taniguchi Model 2)
# ---------------------------------------------------------------------------
model_setup_npz_typeII_herb = xso.setup(
    solver='solve_ivp', model=model_npz_typeII,
    time=ivp_time_array,
    input_vars={
        'Nutrient':      {'value_label': 'N', 'value_init': N_INIT_OPEN},
        'Phytoplankton': {'biomass_label': 'P', 'biomass_init': phyto_init,
                          'phyto_esd_index': phyto_esd.tolist(),
                          'phyto_esd_label': 'phyto_esd'},
        'Zooplankton':   {'biomass_label': 'Z', 'biomass_init': zoo_init,
                          'zoo_esd_index': zoo_esd.tolist(),
                          'zoo_esd_label': 'zoo_esd'},
        'Growth':        {'resource': 'N', 'consumer': 'P',
                          'mu_max': mu_max_arr, 'halfsat': k_s_arr},
        'PhytoLoss':     {'population': 'P', 'nutrient': 'N',
                          'rate': lambda_arr_const},
        'ZooLoss':       {'population': 'Z', 'nutrient': 'N',
                          'rate': M_Z_VAL},
        'Grazing':       {'resource': 'P', 'consumer': 'Z',
                          'phiPZ': phiPZ_herb,
                          'Imax': Imax_arr, 'KsZ': KSZ_VAL},
        'GrazingRouter': {'grazed_phyto': 'P', 'grazed_zoo': 'Z',
                          'assimilated_consumer': 'Z',
                          'excreted_nutrient': 'N',
                          'gamma': GAMMA_VAL},
        'FN_Forcing':    {'forcing_label': 'FN_supply', 'value': F_N_DEFAULT},
        'FN_Input':      {'var': 'N', 'forcing': 'FN_supply',
                          'rate': 1.0 / D_E},
        'DilutionN':     {'var': 'N', 'rate': DILUTION_RATE},
        'DilutionP':     {'var': 'P', 'rate': DILUTION_RATE},
        'DilutionZ':     {'var': 'Z', 'rate': DILUTION_RATE},
    },
    solver_kwargs=IVP_SOLVER_KWARGS,
)

# ---------------------------------------------------------------------------
# Type II × omni (multi-prey log-normal kernel on P+Z — Taniguchi Model 3 /
# MS3-as-built omnivory)
# ---------------------------------------------------------------------------
model_setup_npz_typeII_omni = xso.setup(
    solver='solve_ivp', model=model_npz_typeII,
    time=ivp_time_array,
    input_vars={
        'Nutrient':      {'value_label': 'N', 'value_init': N_INIT_OPEN},
        'Phytoplankton': {'biomass_label': 'P', 'biomass_init': phyto_init,
                          'phyto_esd_index': phyto_esd.tolist(),
                          'phyto_esd_label': 'phyto_esd'},
        'Zooplankton':   {'biomass_label': 'Z', 'biomass_init': zoo_init,
                          'zoo_esd_index': zoo_esd.tolist(),
                          'zoo_esd_label': 'zoo_esd'},
        'Growth':        {'resource': 'N', 'consumer': 'P',
                          'mu_max': mu_max_arr, 'halfsat': k_s_arr},
        'PhytoLoss':     {'population': 'P', 'nutrient': 'N',
                          'rate': lambda_arr_const},
        'ZooLoss':       {'population': 'Z', 'nutrient': 'N',
                          'rate': M_Z_VAL},
        'Grazing':       {'resource': 'P', 'consumer': 'Z',
                          'phiPZ': phiPZ_omni,
                          'Imax': Imax_arr, 'KsZ': KSZ_VAL},
        'GrazingRouter': {'grazed_phyto': 'P', 'grazed_zoo': 'Z',
                          'assimilated_consumer': 'Z',
                          'excreted_nutrient': 'N',
                          'gamma': GAMMA_VAL},
        'FN_Forcing':    {'forcing_label': 'FN_supply', 'value': F_N_DEFAULT},
        'FN_Input':      {'var': 'N', 'forcing': 'FN_supply',
                          'rate': 1.0 / D_E},
        'DilutionN':     {'var': 'N', 'rate': DILUTION_RATE},
        'DilutionP':     {'var': 'P', 'rate': DILUTION_RATE},
        'DilutionZ':     {'var': 'Z', 'rate': DILUTION_RATE},
    },
    solver_kwargs=IVP_SOLVER_KWARGS,
)

# ---------------------------------------------------------------------------
# Type III × matched
# ---------------------------------------------------------------------------
model_setup_npz_typeIII_matched = xso.setup(
    solver='solve_ivp', model=model_npz_typeIII,
    time=ivp_time_array,
    input_vars={
        'Nutrient':      {'value_label': 'N', 'value_init': N_INIT_OPEN},
        'Phytoplankton': {'biomass_label': 'P', 'biomass_init': phyto_init,
                          'phyto_esd_index': phyto_esd.tolist(),
                          'phyto_esd_label': 'phyto_esd'},
        'Zooplankton':   {'biomass_label': 'Z', 'biomass_init': zoo_init,
                          'zoo_esd_index': zoo_esd.tolist(),
                          'zoo_esd_label': 'zoo_esd'},
        'Growth':        {'resource': 'N', 'consumer': 'P',
                          'mu_max': mu_max_arr, 'halfsat': k_s_arr},
        'PhytoLoss':     {'population': 'P', 'nutrient': 'N',
                          'rate': lambda_arr_const},
        'ZooLoss':       {'population': 'Z', 'nutrient': 'N',
                          'rate': M_Z_VAL},
        'Grazing':       {'resource': 'P', 'consumer': 'Z',
                          'phiPZ': phiPZ_matched,
                          'Imax': Imax_arr, 'KsZ': KSZ_VAL},
        'GrazingRouter': {'grazed_phyto': 'P', 'grazed_zoo': 'Z',
                          'assimilated_consumer': 'Z',
                          'excreted_nutrient': 'N',
                          'gamma': GAMMA_VAL},
        'FN_Forcing':    {'forcing_label': 'FN_supply', 'value': F_N_DEFAULT},
        'FN_Input':      {'var': 'N', 'forcing': 'FN_supply',
                          'rate': 1.0 / D_E},
        'DilutionN':     {'var': 'N', 'rate': DILUTION_RATE},
        'DilutionP':     {'var': 'P', 'rate': DILUTION_RATE},
        'DilutionZ':     {'var': 'Z', 'rate': DILUTION_RATE},
    },
    solver_kwargs=IVP_SOLVER_KWARGS,
)

# ---------------------------------------------------------------------------
# Type III × herb
# ---------------------------------------------------------------------------
model_setup_npz_typeIII_herb = xso.setup(
    solver='solve_ivp', model=model_npz_typeIII,
    time=ivp_time_array,
    input_vars={
        'Nutrient':      {'value_label': 'N', 'value_init': N_INIT_OPEN},
        'Phytoplankton': {'biomass_label': 'P', 'biomass_init': phyto_init,
                          'phyto_esd_index': phyto_esd.tolist(),
                          'phyto_esd_label': 'phyto_esd'},
        'Zooplankton':   {'biomass_label': 'Z', 'biomass_init': zoo_init,
                          'zoo_esd_index': zoo_esd.tolist(),
                          'zoo_esd_label': 'zoo_esd'},
        'Growth':        {'resource': 'N', 'consumer': 'P',
                          'mu_max': mu_max_arr, 'halfsat': k_s_arr},
        'PhytoLoss':     {'population': 'P', 'nutrient': 'N',
                          'rate': lambda_arr_const},
        'ZooLoss':       {'population': 'Z', 'nutrient': 'N',
                          'rate': M_Z_VAL},
        'Grazing':       {'resource': 'P', 'consumer': 'Z',
                          'phiPZ': phiPZ_herb,
                          'Imax': Imax_arr, 'KsZ': KSZ_VAL},
        'GrazingRouter': {'grazed_phyto': 'P', 'grazed_zoo': 'Z',
                          'assimilated_consumer': 'Z',
                          'excreted_nutrient': 'N',
                          'gamma': GAMMA_VAL},
        'FN_Forcing':    {'forcing_label': 'FN_supply', 'value': F_N_DEFAULT},
        'FN_Input':      {'var': 'N', 'forcing': 'FN_supply',
                          'rate': 1.0 / D_E},
        'DilutionN':     {'var': 'N', 'rate': DILUTION_RATE},
        'DilutionP':     {'var': 'P', 'rate': DILUTION_RATE},
        'DilutionZ':     {'var': 'Z', 'rate': DILUTION_RATE},
    },
    solver_kwargs=IVP_SOLVER_KWARGS,
)

# ---------------------------------------------------------------------------
# Type III × omni
# ---------------------------------------------------------------------------
model_setup_npz_typeIII_omni = xso.setup(
    solver='solve_ivp', model=model_npz_typeIII,
    time=ivp_time_array,
    input_vars={
        'Nutrient':      {'value_label': 'N', 'value_init': N_INIT_OPEN},
        'Phytoplankton': {'biomass_label': 'P', 'biomass_init': phyto_init,
                          'phyto_esd_index': phyto_esd.tolist(),
                          'phyto_esd_label': 'phyto_esd'},
        'Zooplankton':   {'biomass_label': 'Z', 'biomass_init': zoo_init,
                          'zoo_esd_index': zoo_esd.tolist(),
                          'zoo_esd_label': 'zoo_esd'},
        'Growth':        {'resource': 'N', 'consumer': 'P',
                          'mu_max': mu_max_arr, 'halfsat': k_s_arr},
        'PhytoLoss':     {'population': 'P', 'nutrient': 'N',
                          'rate': lambda_arr_const},
        'ZooLoss':       {'population': 'Z', 'nutrient': 'N',
                          'rate': M_Z_VAL},
        'Grazing':       {'resource': 'P', 'consumer': 'Z',
                          'phiPZ': phiPZ_omni,
                          'Imax': Imax_arr, 'KsZ': KSZ_VAL},
        'GrazingRouter': {'grazed_phyto': 'P', 'grazed_zoo': 'Z',
                          'assimilated_consumer': 'Z',
                          'excreted_nutrient': 'N',
                          'gamma': GAMMA_VAL},
        'FN_Forcing':    {'forcing_label': 'FN_supply', 'value': F_N_DEFAULT},
        'FN_Input':      {'var': 'N', 'forcing': 'FN_supply',
                          'rate': 1.0 / D_E},
        'DilutionN':     {'var': 'N', 'rate': DILUTION_RATE},
        'DilutionP':     {'var': 'P', 'rate': DILUTION_RATE},
        'DilutionZ':     {'var': 'Z', 'rate': DILUTION_RATE},
    },
    solver_kwargs=IVP_SOLVER_KWARGS,
)
