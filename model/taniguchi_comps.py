"""
Taniguchi Model 1 — XSO components
==================================
Implements the literature-equivalent NPZ baseline of Taniguchi, Franks &
Poulin (2014, *Mar Ecol Prog Ser* **514**: 13–33), Model 1 (single-prey,
herbivorous, predator–prey size ratio r = 1).

Companion files
---------------
* `taniguchi_setup.py`  — size grid, allometric coefficients, parameter
  values (verbatim from Taniguchi 2014 Table 1), model assembly, the
  three exported xso.setup objects (transient, slim, stability).
* `model context/Taniguchi_Model1_Baseline.tex` — analytical baseline
  derivation: P*(s) (Eq. 6), Z*(s) (Eq. 9), s_max(N_T) (Eq. 12), the
  three-bin Cariaco centroid / two-point slope under the resolved bin
  convention.
* `model context/Model Equations.md` §8–§14 — equation-set summary.
* `code/XSO_HANDOFF.md`               — XSO framework reference.

Model equations (single-prey, r = 1, F(s) = P(s); Model Equations §11)
----------------------------------------------------------------------
For each size class s_i, i = 1..n:

    dP(s_i)/dt = P(s_i) · [ μ(s_i) · N / (N + k_s(s_i))
                          − Λ
                          − g(s_i) · Z(s_i) / (P(s_i) + k_z(s_i)) ]

    dZ(s_i)/dt = Z(s_i) · [ Γ · g(s_i) · P(s_i) / (P(s_i) + k_z(s_i))
                          − Δ ]

    dN/dt      = − Σ_i U(s_i)
                 + Λ · Σ_i P(s_i)
                 + Δ · Σ_i Z(s_i)
                 + (1 − Γ) · Σ_i G(s_i)

where the per-class fluxes are

    U(s_i) = μ(s_i) · N · P(s_i) / (N + k_s(s_i))      Monod uptake
    G(s_i) = g(s_i) · Z(s_i) · P(s_i) / (P(s_i) + k_z(s_i))   Type-II grazing

and the closure constraint N_T = N + ΣP + ΣZ is held by construction
(no detritus pool, no external supply, no fish; all biomass-associated
losses Λ·P, Δ·Z and the unassimilated grazing fraction (1−Γ)·G return
directly to N — Model Equations §13).

Predator–prey size ratio r = 1 collapses each Z(s_i) onto the matched
P(s_i): the grazing in this file is *diagonal*, not kernel-weighted.
Later MS3 deviations (r = 10, multi-prey kernel, omnivory, distributed
quadratic closure, fish kernel) layer onto this baseline one at a time
— see Taniguchi_Model1_Baseline.tex §7 "Next steps".

XSO dim layout
--------------
Phytoplankton biomass lives on dim 'phyto'; microzooplankton biomass on
dim 'zoo'. For Model 1 (r = 1) the two grids carry *identical* values
— see `taniguchi_setup.py` — but keeping them as separate dims from
day one makes the r → 10 deviation (the first layered structural
change per the latex roadmap) a parameter swap rather than a
restructure. The MonodGrowth and per-class loss components live on
'phyto'; the matched grazing returns one flux on 'phyto' (prey loss)
and one on 'zoo' (predator gain), both numerically identical 1D arrays
of length n.
"""

import numpy as np
import xso


# =============================================================================
# STATE VARIABLES
# =============================================================================

@xso.component
class Nutrient:
    """Dissolved inorganic nitrogen — scalar state variable.

    Taniguchi 2014 closure (Eq. 2): N_T = N + Σ_i P(s_i) + Σ_i Z(s_i)
    is held constant. The integrator follows dN/dt explicitly; the
    closure residual `N_T − (N + ΣP + ΣZ)` becomes a free
    mass-conservation diagnostic at run time.
    """
    value = xso.variable(description='dissolved inorganic nitrogen',
                         attrs={'units': 'µmol N L-1'})


@xso.component
class PhytoSizeSpectrum:
    """Phytoplankton biomass across n logarithmically spaced size classes.

    Carries the canonical Taniguchi phyto size index (`phyto_esd`),
    broadcast as a foreign-referenceable parameter via
    `as_parameter=True`. Other components that need the size grid as
    numeric data (allometric setup functions in the setup file) read
    it via `xso.parameter(foreign=True, dims='phyto')`.

    Units of biomass are µmol N L⁻¹ (Taniguchi 2014, Table state-vars).
    """
    biomass = xso.variable(dims='phyto', description='phytoplankton biomass',
                           attrs={'units': 'µmol N L-1'})
    phyto_esd = xso.index(dims='phyto', as_parameter=True,
                          description='phytoplankton size classes (ESD)',
                          attrs={'units': 'µm ESD'})


@xso.component
class ZooSizeSpectrum:
    """Microzooplankton biomass across n logarithmically spaced size classes.

    Carries its own size index (`zoo_esd`). For Model 1 (r = 1) the
    setup file populates `zoo_esd_index` with the same numeric array
    as `phyto_esd_index`; the two dims remain syntactically distinct
    so later r ≠ 1 deviations only need to repopulate the grid.

    Taniguchi's Z represents *non-metazoan grazers* (protists) only
    (Taniguchi 2014 p. 17, Model Equations §8). The published
    parameter values for g(s) and k_z(s) were synthesised from
    protistan data (Hansen 1997 et al.) — extending Z into the
    metazoan range is a separate structural deviation that is not
    part of the Model 1 baseline.
    """
    biomass = xso.variable(dims='zoo', description='microzooplankton biomass',
                           attrs={'units': 'µmol N L-1'})
    zoo_esd = xso.index(dims='zoo', as_parameter=True,
                        description='microzooplankton size classes (ESD)',
                        attrs={'units': 'µm ESD'})


# =============================================================================
# PHYTOPLANKTON GROWTH — MONOD UPTAKE
# =============================================================================

@xso.component
class MonodGrowth_Tani:
    """Per-class Monod (Michaelis–Menten) nutrient uptake.

    Implements the first term of Taniguchi Eq. 3 (Model Equations §10.1):

        U(s_i) = μ(s_i) · N · P(s_i) / (N + k_s(s_i))

    Wired so that the *same* flux value contributes:
      * −U  to the scalar nutrient pool (resource)   — summed by XSO
            when an array flux is subtracted from a scalar destination
      * +U  per-class to the phytoplankton spectrum (consumer)

    Both `mu_max` and `halfsat` are per-phyto-class arrays
    (dims='phyto'); they are constructed in the setup file from
    Taniguchi's monotonic allometries μ(s) = 1.36·s⁻⁰·¹⁶ and
    k_s(s) = 0.33·s⁺⁰·⁴⁸ (Table 1).

    Distinction from MS3-as-built: MS3 uses Marañón's piecewise
    unimodal μ(s) (Model Equations §2.1) and the steeper
    k_s(s) = 0.144·s⁺⁰·⁸¹. Swapping these allometric arrays from
    Taniguchi to MS3 form is one of the layered deviations on the
    baseline.
    """
    resource = xso.variable(foreign=True, flux='uptake', negative=True,
                            description='dissolved nitrogen (scalar sink)')
    consumer = xso.variable(foreign=True, dims='phyto',
                            flux='uptake', negative=False,
                            description='phytoplankton biomass (per-class source)')

    mu_max = xso.parameter(dims='phyto',
                           description='maximum phytoplankton growth rate per class'
                                       ' [d-1] — Taniguchi μ(s) = 1.36·s^(-0.16)')
    halfsat = xso.parameter(dims='phyto',
                            description='nutrient half-saturation per class'
                                        ' [µmol N L-1] — Taniguchi k_s(s) = 0.33·s^(+0.48)')

    @xso.flux(dims='phyto')
    def uptake(self, resource, consumer, mu_max, halfsat):
        return mu_max * resource * consumer / (resource + halfsat)


# =============================================================================
# PHYTOPLANKTON BIOMASS-ASSOCIATED LOSS Λ — RETURNS DIRECTLY TO N
# =============================================================================

@xso.component
class PhytoLinearLoss_Tani:
    """Size-independent linear phyto loss with full recycling to N.

    Implements the second term of Taniguchi Eq. 3 (Model Equations
    §10.2), with the closure rule that biomass-associated losses
    return *directly* to the nutrient pool — there is no detritus
    pool in Taniguchi Model 1 (Model Equations §13):

        per-class P loss:   Λ · P(s_i)         (dims='phyto')
        N recycling source: Λ · Σ_i P(s_i)     (scalar)

    Λ represents non-grazing phytoplankton losses (viral lysis,
    autolysis, senescence) and is held constant across size classes.
    Taniguchi explicitly rejects an MTE-predicted -0.25 size
    dependence (p. 21) on the grounds that (i) MTE assumes the very
    equilibrium the model is designed to test and (ii) empirical Λ
    values for phytoplankton are sparse. Default Λ = 0.0015 d⁻¹ —
    chosen by Taniguchi to produce a realistic Z:P biomass ratio, not
    fitted (Table 1).

    Two flux methods are required because the per-class P sink and
    the scalar N source carry different values (Λ·P_i vs Λ·ΣP). They
    share the same `rate` parameter Λ.
    """
    population = xso.variable(foreign=True, dims='phyto',
                              flux='mortality', negative=True,
                              description='phytoplankton biomass (per-class sink)')
    nutrient = xso.variable(foreign=True,
                            flux='recycle_to_N', negative=False,
                            description='dissolved nitrogen (scalar source)')

    rate = xso.parameter(description='Λ — size-independent phyto loss rate'
                                     ' [d-1] (Taniguchi 0.0015)')

    @xso.flux(dims='phyto')
    def mortality(self, population, rate):
        return rate * population

    @xso.flux
    def recycle_to_N(self, population, rate):
        return rate * self.m.sum(population)


# =============================================================================
# MATCHED (DIAGONAL) GRAZING — TANIGUCHI MODEL 1, r = 1
# =============================================================================

@xso.component
class MatchedGrazing_Tani:
    """Diagonal Holling Type-II grazing: Z(s_i) eats only P(s_i).

    Implements the third term of Eq. 3 / first term of Eq. 4
    (Model Equations §10.3), specialised to Model 1 (η = 1,
    single-prey) at r = 1 (predator and prey share size class):

        G(s_i) = g(s_i) · Z(s_i) · P(s_i) / (P(s_i) + k_z(s_i))

    Three flux methods route G(s_i) onto its destinations:
      * `loss_phyto` : −G(s_i) per phyto class (dims='phyto')
      * `gain_zoo`   : +Γ · G(s_i) per zoo class (dims='zoo')
      * `recycle_to_N`: +(1 − Γ) · Σ_i G(s_i) onto scalar N

    The per-class G is recomputed in each of the three methods. This
    is intentionally simple — for the matched-grazing case G is a
    length-n vector, not a full matrix, so the redundancy is
    negligible. A group / group_to_arg refactor (cf.
    `cariaco_ssm_comps.SizebasedGrazingMatrix_Full_TypeIII` and
    `GGE_Full_withD`) is the standard XSO pattern for later
    deviations that compute G as a kernel-weighted matrix.

    Parameter conventions follow Taniguchi p. 22 / Model Equations
    §9.3–§9.5: `g_max` and `k_z` are *grazer* properties indexed by
    predator size, hence `dims='zoo'`. With r = 1 the predator-size
    grid is numerically identical to the prey-size grid, so the
    flux methods perform mixed-dim numpy multiplication — both
    arrays are length n at run time, and the flux `dims` label is
    metadata that routes the result to the correct destination.

    `gamma` (Γ) is scalar — Taniguchi reports no significant size
    trend (r² = 0.003, n = 14; Table 1). The Γ value includes
    metabolism — fraction of ingested mass becoming new Z biomass
    *after* respiration; this differs from Poulin & Franks (2010)'s
    γ which excluded metabolism (Taniguchi p. 16).
    """
    prey = xso.variable(foreign=True, dims='phyto',
                        flux='loss_phyto', negative=True,
                        description='phytoplankton (per-class sink)')
    predator = xso.variable(foreign=True, dims='zoo',
                            flux='gain_zoo', negative=False,
                            description='microzooplankton (per-class source)')
    nutrient = xso.variable(foreign=True,
                            flux='recycle_to_N', negative=False,
                            description='dissolved nitrogen (scalar source — '
                                        'sloppy-feeding (1−Γ) fraction)')

    g_max = xso.parameter(dims='zoo',
                          description='maximum grazing rate per predator class'
                                      ' [d-1] — Taniguchi g(s) = 33.96·s^(-0.66)')
    k_z = xso.parameter(dims='zoo',
                        description='grazing half-saturation per predator class'
                                    ' [µmol N L-1] — Taniguchi k_z(s) = 17.92·s^(-0.64)')
    gamma = xso.parameter(description='Γ — gross growth efficiency'
                                      ' (size-independent; Taniguchi 0.31)')

    @xso.flux(dims='phyto')
    def loss_phyto(self, prey, predator, g_max, k_z, gamma):
        # G_i = g_i · Z_i · P_i / (P_i + k_z_i)
        # With r = 1, predator and prey arrays both have length n;
        # numpy broadcasting handles the multiplication.
        return g_max * predator * prey / (prey + k_z)

    @xso.flux(dims='zoo')
    def gain_zoo(self, prey, predator, g_max, k_z, gamma):
        # +Γ · G_i routed to Z (per-class source)
        return gamma * g_max * predator * prey / (prey + k_z)

    @xso.flux
    def recycle_to_N(self, prey, predator, g_max, k_z, gamma):
        # +(1 − Γ) · Σ G_i routed to scalar N (sloppy feeding /
        # respiratory return). This is the (1 − Γ) closure term in
        # Model Equations §11 dN/dt expression.
        G = g_max * predator * prey / (prey + k_z)
        return (1.0 - gamma) * self.m.sum(G)


# =============================================================================
# MICROZOOPLANKTON BIOMASS-ASSOCIATED LOSS Δ — RETURNS DIRECTLY TO N
# =============================================================================

@xso.component
class ZooLinearLoss_Tani:
    """Size-independent linear zoo loss with full recycling to N.

    Implements the second term of Taniguchi Eq. 4 (Model Equations
    §10.4), again with full recycling to N (no detritus pool):

        per-class Z loss:   Δ · Z(s_i)         (dims='zoo')
        N recycling source: Δ · Σ_i Z(s_i)     (scalar)

    Δ represents non-grazing microzooplankton losses (viral lysis,
    senescence). Like Λ, it is held size-independent — Taniguchi p. 19:
    *"Δ is chosen by Taniguchi to produce a realistic Z:P biomass
    ratio and size range, not a fitted value."* Default Δ = 0.025 d⁻¹.

    Structural contrast with MS3-as-built: MS3 uses Banas-style
    *distributed quadratic* closure m_Z · Z_j · Σ_k Z_k with a split
    between detritus routing and export. Replacing the linear Δ·Z
    here with the quadratic form is one of the layered deviations
    on the baseline (Taniguchi_Model1_Baseline.tex §7, deviation 3).
    """
    population = xso.variable(foreign=True, dims='zoo',
                              flux='mortality', negative=True,
                              description='microzooplankton biomass (per-class sink)')
    nutrient = xso.variable(foreign=True,
                            flux='recycle_to_N', negative=False,
                            description='dissolved nitrogen (scalar source)')

    rate = xso.parameter(description='Δ — size-independent zoo loss rate'
                                     ' [d-1] (Taniguchi 0.025)')

    @xso.flux(dims='zoo')
    def mortality(self, population, rate):
        return rate * population

    @xso.flux
    def recycle_to_N(self, population, rate):
        return rate * self.m.sum(population)
