"""
N–P-only spectral model — XSO components
========================================
Components for the diagnostic N–P-only model (Step 1 of the MS3 layered
construction plan, 2026-05-18). No grazing, no zooplankton; purpose is
to demonstrate that a single-nutrient phytoplankton spectrum collapses
to Tilman-style competitive exclusion regardless of size-dependent or
size-independent loss form.

Two configurations supported via the setup file `np_setups.py`:

  * **Closed** — `N_T = N + ΣP` conserved, scan axis is `Nutrient__value_init`
    which sets `N_T`. Phyto biomass-associated loss `Λ` fully recycles to N.
  * **Open chemostat** — F_N supply via `LinearForcingInput` +
    `ConstantExternalNutrient`, plus `ChemostatDilution` applying a single
    rate `λ` uniformly to N and per-class P (i.e. dilution on everything).
    `N_T = F_N / (λ·d_e)` at steady state.

Each configuration is tested in two loss variants:
  * **const** — size-independent Λ = 0.0015 d⁻¹ (Taniguchi value)
  * **allom** — size-dependent Λ(s) = 0.0015·s^(-0.25) (MTE prediction,
    anchored so Λ(1 µm) = the const value; included to demonstrate that
    even MTE-style size-dependence does not break competitive exclusion)

This file is intentionally self-contained — it re-defines the
ConstantExternalNutrient and LinearForcingInput patterns rather than
importing from `cariaco_ssm_comps`, so the file can be moved or shared
without external dependencies.
"""

import numpy as np
import xso


# =============================================================================
# STATE VARIABLES
# =============================================================================

@xso.component
class Nutrient:
    """Dissolved inorganic nitrogen — scalar state variable.

    In the closed configuration the initial value sets `N_T` (mass is
    conserved over the run). In the open configuration the initial value
    is irrelevant to the steady state — only F_N and λ matter.
    """
    value = xso.variable(description='dissolved inorganic nitrogen',
                         attrs={'units': 'mmol N m-3'})


@xso.component
class PhytoSizeSpectrum:
    """Phytoplankton biomass across n logarithmically spaced size classes.

    Carries the phyto size index broadcast as a foreign-referenceable
    parameter (`as_parameter=True`).
    """
    biomass = xso.variable(dims='phyto', description='phytoplankton biomass',
                           attrs={'units': 'mmol N m-3'})
    phyto_esd = xso.index(dims='phyto', as_parameter=True,
                          description='phytoplankton size classes (ESD)',
                          attrs={'units': 'µm ESD'})


# =============================================================================
# FORCINGS
# =============================================================================

@xso.component
class ConstantExternalNutrient:
    """Constant-in-time external nutrient forcing — used by the open variant.

    Pattern from `cariaco_ssm_comps.ConstantExternalNutrient` redefined
    here for self-containment.
    """
    forcing = xso.forcing(setup_func='forcing_setup',
                          description='external nutrient supply')
    value = xso.parameter(description='constant supply value')

    def forcing_setup(self, value):
        @np.vectorize
        def f(t):
            return value
        return f


# =============================================================================
# F_N SUPPLY (OPEN VARIANT)
# =============================================================================

@xso.component
class LinearForcingInput:
    """Non-dimensional linear input flux — adds `forcing * rate` to a sink variable.

    Used in the open variant for the F_N/d_e nutrient supply: with the
    forcing value set to F_N and `rate = 1/d_e`, the flux `F_N/d_e` is
    added to N.
    """
    var = xso.variable(foreign=True, flux='input', negative=False,
                       description='variable receiving input')
    forcing = xso.forcing(foreign=True, description='forcing value')
    rate = xso.parameter(description='input rate scalar (e.g. 1/d_e)')

    @xso.flux
    def input(self, var, forcing, rate):
        return forcing * rate


# =============================================================================
# PHYTOPLANKTON GROWTH — MONOD UPTAKE
# =============================================================================

@xso.component
class MonodGrowth_NP:
    """Per-class Monod (Michaelis–Menten) nutrient uptake.

        U(s_i) = μ(s_i) · N · P(s_i) / (N + k_s(s_i))

    Same functional form as `taniguchi_comps.MonodGrowth_Tani`. The per-class
    `mu_max` and `halfsat` are passed in from the setup file as arrays.
    """
    resource = xso.variable(foreign=True, flux='uptake', negative=True,
                            description='dissolved nitrogen (scalar sink)')
    consumer = xso.variable(foreign=True, dims='phyto',
                            flux='uptake', negative=False,
                            description='phytoplankton biomass (per-class source)')

    mu_max = xso.parameter(dims='phyto',
                           description='maximum growth rate per class [d-1]')
    halfsat = xso.parameter(dims='phyto',
                            description='nutrient half-saturation per class'
                                        ' [mmol N m-3]')

    @xso.flux(dims='phyto')
    def uptake(self, resource, consumer, mu_max, halfsat):
        return mu_max * resource * consumer / (resource + halfsat)


# =============================================================================
# PHYTO BIOMASS-ASSOCIATED LOSS Λ — FULL RECYCLING TO N
# =============================================================================

@xso.component
class PhytoLinearLoss_recycled:
    """Linear per-class phyto loss with full sum-recycling to N.

        per-class P sink:  Λ(s_i) · P(s_i)         (dims='phyto')
        N recycling source: Σ_i Λ(s_i) · P(s_i)    (scalar)

    `rate` is declared `dims='phyto'` so the setup can pass either:
      * a per-class array (size-dependent Λ — the 'allom' variants), or
      * `np.full(n_classes, scalar_Λ)` (size-independent — the 'const' variants).

    No biomass leaves the system — this is the Taniguchi-style internal
    recycling. In the open variant the chemostat dilution (separate
    component) is what removes biomass.
    """
    population = xso.variable(foreign=True, dims='phyto',
                              flux='mortality', negative=True,
                              description='phytoplankton (per-class sink)')
    nutrient = xso.variable(foreign=True,
                            flux='recycle_to_N', negative=False,
                            description='dissolved nitrogen (scalar source)')

    rate = xso.parameter(dims='phyto',
                         description='Λ(s_i) per-class loss rate [d-1]'
                                     ' (pass scalar broadcast for size-independent)')

    @xso.flux(dims='phyto')
    def mortality(self, population, rate):
        return rate * population

    @xso.flux
    def recycle_to_N(self, population, rate):
        return self.m.sum(rate * population)


# =============================================================================
# CHEMOSTAT DILUTION (OPEN VARIANT)
# =============================================================================
# `list_input=True` only works when the foreign variables share the same
# shape (XSO concatenates along a shared `dims` and tracks per-label
# indexing; mixed-shape state vars cause an xarray-simlab dim-mismatch
# error at run time — observed 2026-05-18 attempting [N, P] with N scalar
# and P dim'd, see XSO_HANDOFF §17). Two single-target components below
# is the modular workaround for the N-P case; they share the dilution
# rate at the setup level.

@xso.component
class ChemostatDilution_Scalar:
    """Chemostat dilution applied to a single scalar state variable.

        flux:  rate · var    (sink, leaves system)

    Used here on the scalar nutrient pool N. Pair with
    `ChemostatDilution_PhytoDim` (below) to enforce the same dilution
    rate λ on N and the size-structured P spectrum.
    """
    var = xso.variable(foreign=True, flux='decay', negative=True,
                       description='scalar state variable to dilute')
    rate = xso.parameter(description='λ — chemostat dilution rate [d-1]')

    @xso.flux
    def decay(self, var, rate):
        return rate * var


@xso.component
class ChemostatDilution_PhytoDim:
    """Chemostat dilution applied to a phyto-dim'd state variable, per-class.

        flux(s_i):  rate · var(s_i)    (per-class sink, leaves system)

    Used here on the size-structured phytoplankton biomass. Pair with
    `ChemostatDilution_Scalar` (above) to enforce the same dilution rate
    λ on N and P; in the setup pass the same `rate` value to both.
    """
    var = xso.variable(foreign=True, dims='phyto',
                       flux='decay', negative=True,
                       description='per-class phyto state variable to dilute')
    rate = xso.parameter(description='λ — chemostat dilution rate [d-1]')

    @xso.flux(dims='phyto')
    def decay(self, var, rate):
        return rate * var
