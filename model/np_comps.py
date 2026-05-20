"""
N–P(–Z) spectral model — XSO components
=======================================
Components for the diagnostic N–P-only model (Step 1 of the MS3 layered
construction plan, 2026-05-18) and its NPZ extension (Ago-meeting bridge
figure, 2026-05-19). Same self-contained design principle throughout:
this file re-defines patterns rather than importing from
`cariaco_ssm_comps` or `taniguchi_comps`, so it can be moved or shared
without external dependencies.

N–P core (Step 1, existing)
---------------------------
Two configurations supported via the setup file `np_setups.py`:

  * **Closed** — `N_T = N + ΣP` conserved, scan axis is `Nutrient__value_init`
    which sets `N_T`. Phyto biomass-associated loss `Λ` fully recycles to N.
  * **Open chemostat** — F_N supply via `LinearForcingInput` +
    `ConstantExternalNutrient`, plus `ChemostatDilution` applying a single
    rate `λ` uniformly to N and per-class P (i.e. dilution on everything).
    `N_T = F_N / (λ·d_e)` at steady state.

Each configuration is tested in two loss variants:
  * **const** — size-independent Λ (Stock-style 0.1 d⁻¹ per `np_setups.py`)
  * **allom** — size-dependent Λ(s) = Λ₀·s^(-0.25) (MTE prediction)

NPZ extension (Ago-meeting bridge, 2026-05-19)
----------------------------------------------
Adds the smallest set of components needed to put a zooplankton size
spectrum on top of the existing open NP chemostat and demonstrate the
"grazing → coexistence" beat that NP-only cannot produce:

  * **`ZooSizeSpectrum`** — Z state variable on dim 'zoo'.
  * **`ZooLinearLoss_recycled`** — linear Δ on Z with sum-recycle to N
    (mirror of `PhytoLinearLoss_recycled`; Stock-style background
    mortality). Defined but not used by the current NPZ setups —
    superseded by `ZooQuadraticLoss_recycled` below after the linear
    form proved dynamically too soft under Type II grazing (predator-
    prey transients overshoot through the XSO instability-event floor).
  * **`ZooQuadraticLoss_recycled`** — distributed quadratic closure
    `m_Z · Z_j · Σ_k Z_k` (Banas 2011 / MS3 style) with sum-recycle
    to N. THIS is what the NPZ setups use. Self-buffers against Type
    II predator-prey overshoot: per-capita Z mortality scales with
    total Z, so when Z transiently explodes the closure also explodes
    quadratically.
  * **`ChemostatDilution_ZooDim`** — chemostat dilution on Z (mirror of
    `ChemostatDilution_PhytoDim`; same λ as N and P).
  * **`GrazingMatrix_TypeII`** and **`GrazingMatrix_TypeIII`** — single-
    component grazing with externally-supplied feeding-preference matrix
    `phiPZ` of shape `(n_P+n_Z, n_Z)`. The three structural variants
    (matched single-prey / multi-prey herbivorous / omnivorous) are
    selected entirely at setup time via the phiPZ matrix passed in;
    the components themselves are structure-agnostic.

Routing is Taniguchi-style (no detritus pool): per-prey loss on P/Z,
per-predator gain on Z (with GGE Γ), and the (1−Γ) sloppy-feeding
fraction recycled directly to N. Hansen-1997-derived defaults are
applied in the setup file.
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


# =============================================================================
# NPZ EXTENSION — STATE VARIABLES (Z)
# =============================================================================

@xso.component
class ZooSizeSpectrum:
    """Zooplankton biomass across n logarithmically spaced size classes.

    Mirror of `PhytoSizeSpectrum` on dim 'zoo'. Carries the zoo size
    index as a foreign-referenceable parameter (`as_parameter=True`).
    The actual numeric zoo grid is constructed in `np_setups.py`
    (default: 10× the phyto grid to match `theta_opt = 10`).
    """
    biomass = xso.variable(dims='zoo', description='zooplankton biomass',
                           attrs={'units': 'mmol N m-3'})
    zoo_esd = xso.index(dims='zoo', as_parameter=True,
                        description='zooplankton size classes (ESD)',
                        attrs={'units': 'µm ESD'})


# =============================================================================
# NPZ EXTENSION — Z LOSS Δ — FULL RECYCLING TO N
# =============================================================================

@xso.component
class ZooLinearLoss_recycled:
    """Linear per-class zoo loss Δ with full sum-recycling to N.

        per-class Z sink:   Δ(s_j) · Z(s_j)         (dims='zoo')
        N recycling source: Σ_j Δ(s_j) · Z(s_j)     (scalar)

    Mirror of `PhytoLinearLoss_recycled` on dim 'zoo'. Stock-style
    background mortality on Z, analogous to Λ on P. No biomass leaves
    the system here — chemostat dilution (separate component) removes
    biomass.

    `rate` is declared dims='zoo' so the setup can pass either a per-
    class array (size-dependent Δ) or `np.full(n_zoo, scalar_Δ)` for
    size-independent. The NPZ bridge figure uses the latter; Δ is
    NOT a slope-tuning lever here (cf. Taniguchi 2014 where it is).
    """
    population = xso.variable(foreign=True, dims='zoo',
                              flux='mortality', negative=True,
                              description='zooplankton (per-class sink)')
    nutrient = xso.variable(foreign=True,
                            flux='recycle_to_N', negative=False,
                            description='dissolved nitrogen (scalar source)')
    rate = xso.parameter(dims='zoo',
                         description='Δ(s_j) per-class zoo loss rate [d-1]'
                                     ' (pass scalar broadcast for size-independent)')

    @xso.flux(dims='zoo')
    def mortality(self, population, rate):
        return rate * population

    @xso.flux
    def recycle_to_N(self, population, rate):
        return self.m.sum(rate * population)


# =============================================================================
# NPZ EXTENSION — Z DISTRIBUTED QUADRATIC CLOSURE (Banas 2011 / MS3 style)
# =============================================================================

@xso.component
class ZooQuadraticLoss_recycled:
    """Distributed quadratic Z closure with full sum-recycling to N.

        per-class Z sink:   m_Z · Z(s_j) · Σ_k Z(s_k)    (dims='zoo')
        N recycling source: m_Z · (Σ_k Z(s_k))²          (scalar)

    Banas 2011 closure form for implicit higher-trophic-level predation
    (Edwards & Yool 2000 review on the role of closure terms): each
    predator's per-capita mortality scales with total Z biomass, so
    when the Z community transiently explodes the closure also
    explodes quadratically. This self-buffers against the predator-
    prey overshoot that Holling Type II grazing produces with linear-
    only closure — the same mechanism documented in MS3-as-built
    (`ZooQuadraticMortality_toD` in cariaco_ssm_comps.py:373-409).

    Compared to MS3's `ZooQuadraticMortality_toD`: same per-class
    formula `m_Z · Z_j · ΣZ` and same total flux `m_Z · (ΣZ)²`, but
    routes 100 % to N rather than splitting into D + export. The
    MS3-style D/export split assumes a detritus pool and an explicit
    higher predator (fish kernel) that are deliberately omitted from
    this minimal NPZ demo — keeping the recycle 100 % to N preserves
    mass conservation on N alone.

    Default coefficient `m_Z = 0.1 (mmol N m⁻³)⁻¹ d⁻¹` is the MS3
    setup value (Model Equations.md:213); Banas's analytical estimate
    for similar allometries is 0.26; Banas paper value is 1.0. The
    0.1 choice sits at the low end of the defensible range — enough
    to buffer Type II overshoot, low enough that the closure doesn't
    dominate equilibrium dynamics.
    """
    population = xso.variable(foreign=True, dims='zoo',
                              flux='mortality', negative=True,
                              description='zooplankton (per-class sink)')
    nutrient = xso.variable(foreign=True,
                            flux='recycle_to_N', negative=False,
                            description='dissolved nitrogen (scalar source)')
    rate = xso.parameter(description='m_Z — quadratic closure coefficient '
                                     '[(mmol N m-3)^-1 d^-1]')

    @xso.flux(dims='zoo')
    def mortality(self, population, rate):
        return rate * population * self.m.sum(population)

    @xso.flux
    def recycle_to_N(self, population, rate):
        total_Z = self.m.sum(population)
        return rate * total_Z * total_Z


# =============================================================================
# NPZ EXTENSION — CHEMOSTAT DILUTION ON Z
# =============================================================================

@xso.component
class ChemostatDilution_ZooDim:
    """Chemostat dilution applied to a zoo-dim'd state variable, per-class.

        flux(s_j):  rate · var(s_j)    (per-class sink, leaves system)

    Mirror of `ChemostatDilution_PhytoDim` on dim 'zoo'. Pair with the
    scalar and phyto dilution components in the setup so all three
    (N, P, Z) share the same λ. The mixed-dim same-component pattern
    is the workaround for XSO's `list_input=True` shape constraint
    documented in the `ChemostatDilution_PhytoDim` block above.
    """
    var = xso.variable(foreign=True, dims='zoo',
                       flux='decay', negative=True,
                       description='per-class zoo state variable to dilute')
    rate = xso.parameter(description='λ — chemostat dilution rate [d-1]')

    @xso.flux(dims='zoo')
    def decay(self, var, rate):
        return rate * var


# =============================================================================
# NPZ EXTENSION — GRAZING (HOLLING TYPE II AND TYPE III) — GROUP-FLUX PATTERN
# =============================================================================
# Canonical XSO group / group_to_arg routing (XSO_HANDOFF.md §8.1, §14.3):
# one component computes the (n_P+n_Z, n_Z) grazing matrix once per timestep
# and publishes it to the 'graze_matrix' XSO group; a separate router
# component reads the group and distributes per-prey loss / per-predator
# gain / scalar N recycle. Mirrors MS3-as-built `SizebasedGrazingMatrix_
# Full_TypeIII` + `GGE_Full_withD` (cariaco_ssm_comps.py:204-322), simplified
# (no detritus pool, no f_egest_D split — Taniguchi-style direct N recycle).
#
# Structural variant (matched single-prey / multi-prey herbivorous / omni-
# vorous) is set entirely by the `phiPZ` matrix supplied externally at
# setup time. The two grazing-matrix components below differ only in their
# saturation kernel (Type II vs Type III); both publish to the SAME group
# label, so the same `GrazingRouter` works for either model.


@xso.component
class GrazingMatrix_TypeII:
    """Compute the (full, zoo) Holling Type II grazing matrix once per step.

        G_kj = I_max,j · Z_j · φ_kj · B_k / (S_j + K_sZ)

    where `B = [P; Z]` is the concatenated prey biomass vector,
    `S_j = Σ_k φ_kj · B_k` is the kernel-weighted prey availability for
    predator j, and φ = phiPZ is the (n_P+n_Z, n_Z) external preference
    matrix. The matrix is published to the XSO group 'graze_matrix';
    downstream routing is handled by `GrazingRouter`.

    Pattern: producer side of the canonical group / group_to_arg idiom
    (XSO_HANDOFF.md §14.3). 'full' is just a project-level dim label
    naming the concatenated prey index, exactly as MS3 uses it
    (XSO_HANDOFF.md §17).
    """
    resource = xso.variable(foreign=True, dims='phyto',
                            description='phytoplankton biomass (prey)')
    consumer = xso.variable(foreign=True, dims='zoo',
                            description='zooplankton biomass (predator)')

    phiPZ = xso.parameter(dims=('full', 'zoo'),
                          description='feeding preference matrix (prey × predator),'
                                      ' shape (n_P+n_Z, n_Z), supplied externally')
    Imax = xso.parameter(dims='zoo',
                         description='max ingestion rate per predator class [d-1]')
    KsZ = xso.parameter(description='Type II half-saturation [mmol N m-3]')

    @xso.flux(group='graze_matrix', dims=('full', 'zoo'))
    def grazing(self, resource, consumer, phiPZ, Imax, KsZ):
        biomass = self.m.concatenate((resource, consumer))
        S = self.m.sum(phiPZ * biomass[:, None], axis=0)
        return Imax * consumer * phiPZ * biomass[:, None] / (S + KsZ)


@xso.component
class GrazingMatrix_TypeIII:
    """Compute the (full, zoo) Holling Type III grazing matrix once per step.

        G_kj = I_max,j · Z_j · φ_kj · B_k · S_j / (S_j² + K_sZ²)

    Sigmoidal saturation produces a low-prey refuge (Mattern et al. 2026 /
    Dutkiewicz et al. 2015, 2020; cf. MS3-as-built
    `SizebasedGrazingMatrix_Full_TypeIII` in `cariaco_ssm_comps.py:204-251`,
    which uses the same formula). Routing identical to `GrazingMatrix_TypeII`
    via the shared 'graze_matrix' group.
    """
    resource = xso.variable(foreign=True, dims='phyto',
                            description='phytoplankton biomass (prey)')
    consumer = xso.variable(foreign=True, dims='zoo',
                            description='zooplankton biomass (predator)')

    phiPZ = xso.parameter(dims=('full', 'zoo'),
                          description='feeding preference matrix (prey × predator),'
                                      ' shape (n_P+n_Z, n_Z), supplied externally')
    Imax = xso.parameter(dims='zoo',
                         description='max ingestion rate per predator class [d-1]')
    KsZ = xso.parameter(description='Type III half-saturation [mmol N m-3]')

    @xso.flux(group='graze_matrix', dims=('full', 'zoo'))
    def grazing(self, resource, consumer, phiPZ, Imax, KsZ):
        biomass = self.m.concatenate((resource, consumer))
        S = self.m.sum(phiPZ * biomass[:, None], axis=0)
        return Imax * consumer * phiPZ * biomass[:, None] * S / (S ** 2 + KsZ ** 2)


@xso.component
class GrazingRouter:
    """Route the 'graze_matrix' group flux into per-class prey loss,
    per-predator gain, and the scalar (1−Γ) recycle to N.

    Consumer side of the group / group_to_arg idiom (XSO_HANDOFF.md §8.1,
    §14.3). Reads the (n_P+n_Z, n_Z) matrix published by either
    `GrazingMatrix_TypeII` or `GrazingMatrix_TypeIII`, sums it along
    the appropriate axes, and distributes:

        loss on P_i  : −Σ_j G_ij           [first n_P rows of axis-1 sum]
        loss on Z_i  : −Σ_j G_(n_P+i)j     [remaining n_Z rows]
        gain on Z_j  : +Γ · Σ_k G_kj       [axis-0 sum × Γ]
        recycle to N : +(1−Γ) · Σ_kj G_kj  [scalar; sloppy feeding]

    Identical routing for Type II and Type III — the saturation kernel
    is already baked into the grazing matrix by the producer component.
    Mirrors MS3 `GGE_Full_withD` (cariaco_ssm_comps.py:256-322) with the
    detritus / egestion split removed (Taniguchi-style direct N recycle).

    `grazed_zoo` and `assimilated_consumer` are two foreign refs to the
    same Z label — XSO allows multi-role same-label wiring exactly as in
    MS3's GGE_Full_withD. `loss_Z_as_prey` is nonzero only for omnivorous
    phiPZ (Z-block of the matrix is zero in matched/herbivorous variants).
    """
    grazed_phyto = xso.variable(foreign=True, dims='phyto',
                                flux='loss_P', negative=True,
                                description='phytoplankton (per-class sink)')
    grazed_zoo = xso.variable(foreign=True, dims='zoo',
                              flux='loss_Z_as_prey', negative=True,
                              description='zooplankton-as-prey (per-class sink; '
                                          'nonzero only for omnivorous phiPZ)')
    assimilated_consumer = xso.variable(foreign=True, dims='zoo',
                                        flux='gain_Z',
                                        description='zooplankton (per-class source)')
    excreted_nutrient = xso.variable(foreign=True,
                                     flux='recycle_to_N',
                                     description='dissolved nitrogen (scalar source — '
                                                 'sloppy-feeding (1−Γ) fraction)')

    gamma = xso.parameter(description='Γ — gross growth efficiency (size-independent)')

    @xso.flux(dims='phyto', group_to_arg='graze_matrix')
    def loss_P(self, grazed_phyto, graze_matrix, gamma):
        # Σ over predators (axis=1) → length n_P+n_Z; take the first n_P entries
        per_prey_loss = self.m.sum(graze_matrix, axis=1)
        n_P = len(grazed_phyto)
        return per_prey_loss[0:n_P]

    @xso.flux(dims='zoo', group_to_arg='graze_matrix')
    def loss_Z_as_prey(self, grazed_phyto, grazed_zoo, graze_matrix, gamma):
        # Σ over predators (axis=1) → length n_P+n_Z; take the last n_Z entries
        per_prey_loss = self.m.sum(graze_matrix, axis=1)
        n_P = len(grazed_phyto)
        n_Z = len(grazed_zoo)
        return per_prey_loss[n_P:n_P + n_Z]

    @xso.flux(dims='zoo', group_to_arg='graze_matrix')
    def gain_Z(self, graze_matrix, gamma):
        # Σ over prey (axis=0) → per-predator total ingestion I_j, × Γ
        return self.m.sum(graze_matrix, axis=0) * gamma

    @xso.flux(group_to_arg='graze_matrix')
    def recycle_to_N(self, graze_matrix, gamma):
        # (1 − Γ) × Σ_kj G_kj — scalar sloppy-feeding return
        return (1.0 - gamma) * self.m.sum(graze_matrix)
