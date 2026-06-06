"""
Cariaco baseline NPZ model components (Option A)
================================================
Iteration-1 baseline for MS3: Taniguchi 2014 Model 1 biology (matched
single-prey grazing, Holling Type II, linear closure) layered onto
Stock-style physical setup (F_N/d_e supply, phyto sinking, no detritus).

Designed as the foundation for layered structural deviations per
`MS3 Project Background.md` and `Size Spectral Setup Survey.md` §20.4:
- Iteration-1 baseline (this file): matched single-prey + Type II + linear
  closure + Stock supply + phyto sinking, no detritus
- Planned later layers: Type III grazing (Survey §7), distributed quadratic
  Z closure (§11), Marañón unimodal μ (§3), allometric exponent swap (§6),
  detritus pool (§13), fish kernel (§14)

References:
- Taniguchi, Franks & Poulin 2014 (MEPS 514:13-33) — Model 1 NPZ structure
  (matched single-prey r=1, Type II, linear Δ·Z); Table 1 allometries
- Cloern 2018 (L&O 63:S392-S409) — direct precedent for Taniguchi-M1
  adaptation to a specific ecosystem; Stock-style physical extensions
- Stock, Powell & Levin 2008 (J. Mar. Syst. 74:134-152) — F_N/d_e supply
  Eq. 7; box model over euphotic depth d_e
- Hansen, Bjørnsen & Hansen 1997 (L&O 42:687-704) — allometric data
- Banas 2011 (Ecol. Model. 222:2663) — m_P = 0.1·μ_max (form, not adopted)
- `model context/Size Spectral Setup Survey.md` §§5, 6, 7, 8, 11, 12
- `model context/Taniguchi_Model1_Baseline.tex` §7.3 — open-system extension
"""

import numpy as np
import xso


# =============================================================================
# STATE VARIABLES
# =============================================================================

@xso.component
class Nutrient:
    """Dissolved inorganic nitrogen — scalar state variable."""
    value = xso.variable(description='dissolved inorganic nitrogen',
                         attrs={'units': 'mmol N m-3'})


@xso.component
class PhytoSizeSpectrum:
    """Phytoplankton biomass across n_phyto log-spaced size classes."""
    biomass = xso.variable(dims='phyto', description='phytoplankton biomass',
                           attrs={'units': 'mmol N m-3'})
    phyto_esd = xso.index(dims='phyto', as_parameter=True,
                          description='phyto size classes (ESD)',
                          attrs={'units': 'µm'})


@xso.component
class ZooSizeSpectrum:
    """Zooplankton biomass across n_zoo log-spaced size classes.

    For MS3 (and Stock/Banas) convention, zoo_esd = r · phyto_esd with r=10
    (Survey §9 — between Hansen 1994 ciliate 8:1 and copepod 18:1 ratios).
    Matched grazing requires n_zoo == n_phyto so each Z_i has its prey P_i.
    """
    biomass = xso.variable(dims='zoo', description='zooplankton biomass',
                           attrs={'units': 'mmol N m-3'})
    zoo_esd = xso.index(dims='zoo', as_parameter=True,
                        description='zoo size classes (ESD)',
                        attrs={'units': 'µm'})


# =============================================================================
# FORCING — STOCK 2008 NUTRIENT SUPPLY (Eq. 7)
# =============================================================================

@xso.component
class StockNutrientSupply:
    """Constant new-nutrient flux into the euphotic-zone box (Stock 2008 Eq. 7).

        J_supply = F_N / d_e   [mmol N m-3 d-1]

    F_N is the volumetric flux (mmol N m-2 d-1) of new nutrient into the
    surface layer; d_e is the box depth (m). For MS3 Cariaco, F_N is
    derived per month from Laws-2011 f-ratio × PP (see depth_profile_data.r).
    Mirrors cariaco_ssm_comps.StockNutrientSupply.

    d_e is a *broadcast* parameter so PhytoSinking foreign-references the same
    value: one regime d_e then fully specifies the box geometry — supply
    F_N/d_e and sinking w_sink/d_e both follow from it, with no duplicated
    depth bookkeeping. (Relies on the XSO broadcast-parameter fix.)
    """
    var = xso.variable(foreign=True, flux='input', negative=False,
                       description='nutrient receiving the supply')
    FN = xso.parameter(description='new-nutrient flux F_N [mmol N m-2 d-1]')
    de = xso.parameter(broadcast=True,
                       description='euphotic box depth d_e [m] (broadcast; '
                                   'shared with PhytoSinking)')

    @xso.flux
    def input(self, var, FN, de):
        return FN / de


# =============================================================================
# PHYTO GROWTH — MONOD UPTAKE (Taniguchi Eq. 3 / Cloern Eq. 1)
# =============================================================================

@xso.component
class MonodGrowth_NP:
    """Per-class Monod phytoplankton growth on N.

        U_i = μ_max,i · N · P_i / (N + k_s,i)

    Universal form across the surveyed literature (Taniguchi 2014 Eq. 3,
    Cloern 2018 Eq. 1, Banas 2011, Stock 2008). Per-class μ_max and k_s
    arrays carry the size dependence (set in the setup file).

    Taniguchi/Cloern allometric defaults (Survey §3, §4):
        μ_max(s) = 1.36 · s^(-0.16)   [d-1]
        k_s(s)   = 0.33 · s^(+0.48)   [mmol N m-3]
    """
    resource = xso.variable(foreign=True, flux='uptake', negative=True,
                            description='dissolved nitrogen (scalar sink)')
    consumer = xso.variable(foreign=True, dims='phyto',
                            flux='uptake', negative=False,
                            description='phyto biomass (per-class source)')
    mu_max = xso.parameter(dims='phyto',
                           description='per-class max growth rate [d-1]')
    halfsat = xso.parameter(dims='phyto',
                            description='per-class half-sat [mmol N m-3]')

    @xso.flux(dims='phyto')
    def uptake(self, resource, consumer, mu_max, halfsat):
        return mu_max * resource * consumer / (resource + halfsat)


# =============================================================================
# PHYTO BACKGROUND LOSS — TANIGUCHI Λ·P, RECYCLED TO N
# =============================================================================

@xso.component
class PhytoLinearLoss_recycled:
    """Linear per-class phyto loss with full recycling to N.

        Per-class P sink:  Λ · P_i
        N source (sum):    Σ_i Λ · P_i

    Taniguchi 2014 Eq. 3 / Cloern 2018 (set to 0 there) / Banas 2011 form
    (Banas uses m_P = 0.1·μ_max). Represents viral lysis + autolysis +
    senescence. Recycled to N in iteration-1 (Taniguchi convention: closed
    mass balance except for explicit physical sinks).

    Taniguchi default Λ = 0.0015 d-1 (Table 1) is calibrated for realistic
    Z:P ratio (Taniguchi p. 21), NOT slope-tuned — the slope theorem in
    Taniguchi_Model1_Baseline.tex Eq. 6-8 shows Λ does not enter P*(s).
    """
    population = xso.variable(foreign=True, dims='phyto',
                              flux='mortality', negative=True,
                              description='phyto (per-class sink)')
    nutrient = xso.variable(foreign=True,
                            flux='recycle_to_N', negative=False,
                            description='nutrient (scalar source)')
    rate = xso.parameter(dims='phyto', description='Λ per-class [d-1]')

    @xso.flux(dims='phyto')
    def mortality(self, population, rate):
        return rate * population

    @xso.flux
    def recycle_to_N(self, population, rate):
        return self.m.sum(rate * population)


# =============================================================================
# ZOO BACKGROUND LOSS — TANIGUCHI Δ·Z, RECYCLED TO N
# =============================================================================

@xso.component
class ZooLinearLoss_recycled:
    """Linear per-class zoo loss with full recycling to N.

        Per-class Z sink:  Δ · Z_i
        N source (sum):    Σ_i Δ · Z_i

    Taniguchi 2014 Eq. 4 / Cloern 2018 Eq. 2. Background higher-trophic
    mortality, recycled to N.

    Taniguchi default Δ = 0.025 d-1 (Table 1, calibrated for realistic Z:P).
    Cloern adjusted to 0.06 d-1 for SF Bay (calibrated to match the observed
    biomass fraction in their largest size class — Z:P / height tuning, not
    slope tuning; slope is set by allometric exponents e_g, e_kz per the
    Taniguchi/Poulin&Franks slope theorem).
    """
    population = xso.variable(foreign=True, dims='zoo',
                              flux='mortality', negative=True,
                              description='zoo (per-class sink)')
    nutrient = xso.variable(foreign=True,
                            flux='recycle_to_N', negative=False,
                            description='nutrient (scalar source)')
    rate = xso.parameter(dims='zoo', description='Δ per-class [d-1]')

    @xso.flux(dims='zoo')
    def mortality(self, population, rate):
        return rate * population

    @xso.flux
    def recycle_to_N(self, population, rate):
        return self.m.sum(rate * population)


# =============================================================================
# MATCHED GRAZING — TANIGUCHI M1 STRUCTURE WITH r=10
# =============================================================================

@xso.component
class MatchedGrazing_TypeII:
    """Matched single-prey grazing, Holling Type II, scalar-GGE routing.

    For each matched pair i (predator Z_i grazes prey P_i):

        G_i = I_max,i · Z_i · P_i / (P_i + K_sZ,i)

    Routing (Taniguchi M1, scalar GGE):
        Per-class P sink:  G_i
        Per-class Z gain:  Γ · G_i
        Scalar N source:   (1 − Γ) · Σ_i G_i   (sloppy feeding / excretion)

    Pred:prey size ratio r enters through the zoo grid construction:
    zoo_esd = r · phyto_esd. With r=10 (Stock/Banas/MS3 convention,
    Survey §9), Z_i sits at 10× the ESD of its matched prey P_i.
    Matched-by-index — requires n_zoo == n_phyto.

    Taniguchi 2014 Table 1 default allometric exponents (microzoo-only
    Hansen subset, Survey §6 / §8):
        I_max(s) = 33.96 · s^(-0.66)  [d-1]
        K_sZ(s)  = 17.92 · s^(-0.64)  [mmol N m-3]

    MS3-defensible alternative for mesozoo-inclusive grid (Survey §6/§8,
    Correction.md): I_max = 26·s^(-0.48) (Stock/Hansen/Ward/Mattern
    cluster), K_sZ = 3.0 uniform (Hansen 1997 all-groups, settled in
    Correction.md). Swap arrays in the setup; component unchanged.

    To swap functional response: replace this component with the
    MatchedGrazing_TypeIII sibling below in the setup's xso.create() call.
    """
    phyto = xso.variable(foreign=True, dims='phyto',
                         flux='loss_P', negative=True,
                         description='phyto prey (per-class sink)')
    zoo = xso.variable(foreign=True, dims='zoo',
                       flux='gain_Z', negative=False,
                       description='zoo predator (per-class source)')
    nutrient = xso.variable(foreign=True,
                            flux='recycle_to_N', negative=False,
                            description='nutrient (scalar source — sloppy feeding)')
    Imax = xso.parameter(dims='zoo',
                         description='per-class max ingestion rate [d-1]')
    KsZ = xso.parameter(dims='zoo',
                        description='per-class half-saturation [mmol N m-3]')
    gamma = xso.parameter(description='Γ scalar GGE (gross growth efficiency)')

    @xso.flux(dims='phyto')
    def loss_P(self, phyto, zoo, Imax, KsZ, gamma):
        return Imax * zoo * phyto / (phyto + KsZ)

    @xso.flux(dims='zoo')
    def gain_Z(self, phyto, zoo, Imax, KsZ, gamma):
        return gamma * Imax * zoo * phyto / (phyto + KsZ)

    @xso.flux
    def recycle_to_N(self, phyto, zoo, Imax, KsZ, gamma):
        G = Imax * zoo * phyto / (phyto + KsZ)
        return (1.0 - gamma) * self.m.sum(G)


@xso.component
class MatchedGrazing_TypeIII:
    """Matched single-prey grazing, Holling Type III, scalar-GGE routing.

        G_i = I_max,i · Z_i · P_i² / (P_i² + K_sZ,i²)

    Same routing as Type II. Type III adds a low-prey refuge (G ∝ P² at
    P << K_sZ). Per Rohr 2022 (Survey §7): substantially more stable than
    Type II (37.5% of tested cases unstable for II vs 1.7% for III).
    Used in MS3-as-built (cariaco_ssm_comps.SizebasedGrazingMatrix_Full_TypeIII).
    Provided here as a one-line swap-in for the planned numbered deviation.
    """
    phyto = xso.variable(foreign=True, dims='phyto',
                         flux='loss_P', negative=True)
    zoo = xso.variable(foreign=True, dims='zoo',
                       flux='gain_Z', negative=False)
    nutrient = xso.variable(foreign=True,
                            flux='recycle_to_N', negative=False)
    Imax = xso.parameter(dims='zoo', description='per-class max ingestion [d-1]')
    KsZ = xso.parameter(dims='zoo', description='per-class half-sat [mmol N m-3]')
    gamma = xso.parameter(description='Γ scalar GGE')

    @xso.flux(dims='phyto')
    def loss_P(self, phyto, zoo, Imax, KsZ, gamma):
        return Imax * zoo * phyto**2 / (phyto**2 + KsZ**2)

    @xso.flux(dims='zoo')
    def gain_Z(self, phyto, zoo, Imax, KsZ, gamma):
        return gamma * Imax * zoo * phyto**2 / (phyto**2 + KsZ**2)

    @xso.flux
    def recycle_to_N(self, phyto, zoo, Imax, KsZ, gamma):
        G = Imax * zoo * phyto**2 / (phyto**2 + KsZ**2)
        return (1.0 - gamma) * self.m.sum(G)


# =============================================================================
# PHYTO SINKING — ONE-WAY EXPORT (no recycling)
# =============================================================================

@xso.component
class PhytoSinking_export:
    """Per-class phyto sinking out of the euphotic box.

        Per-class P sink: (w_sink / d_e) · P_i    [no recycling]

    Stock 2008 / Banas 2011 form (Survey §12). Mass leaves the box one-way,
    providing the principal export sink in Stock-style models without a
    detritus pool. Combined with F_N supply, this creates a true open-system
    chemostat with finite steady-state biomass.

    Iteration-1 default: scalar w_sink. Size-dependent option (Ward 2012,
    Laws 1975: w_p = 0.28 · V^0.39 m/d) is a candidate numbered deviation
    for later iterations.

    d_e is foreign-referenced from the Inflow supply component (broadcast), so
    setting one regime d_e drives both supply (F_N/d_e) and sinking (w_sink/d_e);
    w_sink is a true constant.
    """
    population = xso.variable(foreign=True, dims='phyto',
                              flux='sinking', negative=True,
                              description='phyto (per-class sink)')
    w_sink = xso.parameter(description='sinking velocity w_sink [m d-1]')
    de = xso.parameter(foreign=True,
                       description='euphotic box depth d_e [m] (shared from Inflow)')

    @xso.flux(dims='phyto')
    def sinking(self, population, w_sink, de):
        return (w_sink / de) * population
