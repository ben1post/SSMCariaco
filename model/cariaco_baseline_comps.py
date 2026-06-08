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


# =============================================================================
# DISTRIBUTED (KERNEL) GRAZING — group-flux family, per-class Imax AND KsZ
# =============================================================================
# Layered deviation beyond the matched single-prey baseline above. Unlike
# MatchedGrazing_TypeII/III (1:1 predator-prey, delta kernel), these spread
# each predator's feeding over a size window via the phiPZ preference matrix
# (Survey §9), so prey availability S_j is a resolution-invariant integral
# rather than a single ~1/N bin. The structural variant is set ENTIRELY by
# which phiPZ is passed (compute_grazing_kernel mode):
#   'matched' — delta on the P-block diagonal (reproduces MatchedGrazing as a
#               cross-check, given zoo_esd = theta_opt * phyto_esd)
#   'herb'    — log-normal kernel on P prey only (Taniguchi Model 2 analogue)
#   'omni'    — log-normal kernel on P+Z prey (Taniguchi Model 3 / MS3-as-built)
#
# CRITICAL: both Imax and KsZ are dims='zoo' per-class arrays here, so the
# grazing allometry (e_g on Imax, e_kz on KsZ — the Poulin-Franks/Taniguchi
# slope levers, Survey §18) is always explicit and a scalar regression is
# structurally impossible. Pass np.full(n_zoo, x) in the setup for a uniform
# value; pass an allometry for a size-dependent one. The choice is never hidden.
#
# Group-flux routing (XSO_HANDOFF §8.1, §16): the matrix component publishes
# the (n_P+n_Z, n_Z) grazing matrix to the 'graze_matrix' group; the router
# reads it and distributes per-prey loss / per-predator gain / N recycle.

def compute_grazing_kernel(phyto_esd, zoo_esd, mode='omni',
                           theta_opt=10.0, sigma_log=0.25):
    """Feeding-preference matrix phiPZ of shape (n_P + n_Z, n_Z).

    mode : {'matched', 'herb', 'omni'}
        'matched' — delta on the P-block diagonal (Taniguchi M1 structure;
                    with zoo_esd = theta_opt * phyto_esd the peak lands on
                    the matched class, so this is a true 1:1 delta).
        'herb'    — log-normal kernel on P prey only (Z-block zero).
        'omni'    — log-normal kernel on P+Z prey, Z-on-self diagonal zeroed
                    (no within-class cannibalism).
    sigma_log is the kernel width in log10(ESD) space, 2*sigma**2 convention
    (Survey §9; MS3 default sigma_log=0.25). theta_opt is the predator:prey
    ESD ratio (kernel peak; Survey §9, MS3 default 10).
    """
    phyto_esd = np.asarray(phyto_esd)
    zoo_esd = np.asarray(zoo_esd)
    n_P, n_Z = len(phyto_esd), len(zoo_esd)
    prey_esd = np.concatenate([phyto_esd, zoo_esd])
    log_ratio = np.log10(zoo_esd[None, :] / prey_esd[:, None])
    kernel = np.exp(-((log_ratio - np.log10(theta_opt)) ** 2) / (2 * sigma_log ** 2))
    phiPZ = np.zeros((n_P + n_Z, n_Z))
    if mode == 'matched':
        for j in range(n_Z):
            phiPZ[j, j] = 1.0
    elif mode == 'herb':
        phiPZ[:n_P, :] = kernel[:n_P, :]
    elif mode == 'omni':
        phiPZ[:] = kernel
        for j in range(n_Z):
            phiPZ[n_P + j, j] = 0.0
    else:
        raise ValueError(f"mode must be 'matched'/'herb'/'omni', got {mode!r}")
    return phiPZ


@xso.component
class DistributedGrazing_TypeII:
    """Distributed Holling Type II grazing (group-flux). Per-class Imax AND KsZ.

        S_j  = Σ_k φ_kj · B_k                          (B = [P; Z])
        G_kj = Imax_j · Z_j · φ_kj · B_k / (S_j + KsZ_j)

    Publishes the (n_P+n_Z, n_Z) matrix to the 'graze_matrix' group; routed by
    DistributedGrazingRouter. φ = phiPZ supplied from the setup (kernel mode).
    """
    resource = xso.variable(foreign=True, dims='phyto',
                            description='phytoplankton biomass (prey)')
    consumer = xso.variable(foreign=True, dims='zoo',
                            description='zooplankton biomass (predator)')
    phiPZ = xso.parameter(dims=('full', 'zoo'),
                          description='feeding preference matrix (prey × predator)')
    Imax = xso.parameter(dims='zoo',
                         description='per-class max ingestion rate [d-1]')
    KsZ = xso.parameter(dims='zoo',
                        description='per-class grazing half-saturation [mmol N m-3]')

    @xso.flux(group='graze_matrix', dims=('full', 'zoo'))
    def grazing(self, resource, consumer, phiPZ, Imax, KsZ):
        biomass = self.m.concatenate((resource, consumer))
        S = self.m.sum(phiPZ * biomass[:, None], axis=0)
        return Imax * consumer * phiPZ * biomass[:, None] / (S + KsZ)


@xso.component
class DistributedGrazing_TypeIII:
    """Distributed Holling Type III grazing (group-flux). Per-class Imax AND KsZ.

        G_kj = Imax_j · Z_j · φ_kj · B_k · S_j / (S_j² + KsZ_j²)

    Low-prey refuge (G ∝ S² at S << KsZ; Rohr 2022, Survey §7). Same routing as
    DistributedGrazing_TypeII via the shared 'graze_matrix' group.
    """
    resource = xso.variable(foreign=True, dims='phyto',
                            description='phytoplankton biomass (prey)')
    consumer = xso.variable(foreign=True, dims='zoo',
                            description='zooplankton biomass (predator)')
    phiPZ = xso.parameter(dims=('full', 'zoo'),
                          description='feeding preference matrix (prey × predator)')
    Imax = xso.parameter(dims='zoo',
                         description='per-class max ingestion rate [d-1]')
    KsZ = xso.parameter(dims='zoo',
                        description='per-class grazing half-saturation [mmol N m-3]')

    @xso.flux(group='graze_matrix', dims=('full', 'zoo'))
    def grazing(self, resource, consumer, phiPZ, Imax, KsZ):
        biomass = self.m.concatenate((resource, consumer))
        S = self.m.sum(phiPZ * biomass[:, None], axis=0)
        return Imax * consumer * phiPZ * biomass[:, None] * S / (S ** 2 + KsZ ** 2)


@xso.component
class DistributedGrazingRouter:
    """Route the 'graze_matrix' group into per-prey loss (P and Z), per-predator
    gain (× Γ), and the scalar (1−Γ) recycle to N (Taniguchi-style direct
    recycle; no detritus). loss_Z is nonzero only for omnivorous phiPZ.

    Mirrors the matched routing of MatchedGrazing_TypeII/III but for the
    multi-prey kernel case. Consumer side of the group / group_to_arg idiom
    (XSO_HANDOFF §8.1).
    """
    grazed_phyto = xso.variable(foreign=True, dims='phyto',
                                flux='loss_P', negative=True,
                                description='phytoplankton (per-class sink)')
    grazed_zoo = xso.variable(foreign=True, dims='zoo',
                              flux='loss_Z', negative=True,
                              description='zooplankton-as-prey (per-class sink; '
                                          'nonzero only for omnivorous phiPZ)')
    assimilated_consumer = xso.variable(foreign=True, dims='zoo',
                                        flux='gain_Z',
                                        description='zooplankton (per-class source)')
    excreted_nutrient = xso.variable(foreign=True,
                                     flux='recycle_to_N',
                                     description='nutrient (scalar source — '
                                                 'sloppy-feeding (1−Γ) fraction)')
    gamma = xso.parameter(description='Γ — gross growth efficiency (scalar)')

    @xso.flux(dims='phyto', group_to_arg='graze_matrix')
    def loss_P(self, grazed_phyto, graze_matrix, gamma):
        per_prey_loss = self.m.sum(graze_matrix, axis=1)
        return per_prey_loss[0:len(grazed_phyto)]

    @xso.flux(dims='zoo', group_to_arg='graze_matrix')
    def loss_Z(self, grazed_phyto, grazed_zoo, graze_matrix, gamma):
        n_P = len(grazed_phyto)
        per_prey_loss = self.m.sum(graze_matrix, axis=1)
        return per_prey_loss[n_P:n_P + len(grazed_zoo)]

    @xso.flux(dims='zoo', group_to_arg='graze_matrix')
    def gain_Z(self, graze_matrix, gamma):
        return self.m.sum(graze_matrix, axis=0) * gamma

    @xso.flux(group_to_arg='graze_matrix')
    def recycle_to_N(self, graze_matrix, gamma):
        return (1.0 - gamma) * self.m.sum(graze_matrix)


# =============================================================================
# ZOOPLANKTON QUADRATIC CLOSURE — full recycle to N (Banas 2011 form, no D)
# =============================================================================

@xso.component
class ZooQuadraticLoss_recycled:
    """Distributed quadratic Z closure with full sum-recycle to N.

        per-class Z sink:   m_Z · Z_j · Σ_k Z_k    (dims='zoo')
        N recycling source: m_Z · (Σ_k Z_k)²       (scalar)

    Banas 2011 closure form (Survey §11). m_Z is a quadratic coefficient
    [(mmol N m-3)^-1 d-1] — scalar by definition, NOT a size allometry, so a
    scalar here is correct (unlike the grazing Imax/KsZ). Routes 100 % to N
    (Taniguchi-style; no detritus pool in this baseline). MS3 default m_Z=0.1.
    """
    population = xso.variable(foreign=True, dims='zoo',
                              flux='mortality', negative=True,
                              description='zooplankton (per-class sink)')
    nutrient = xso.variable(foreign=True,
                            flux='recycle_to_N', negative=False,
                            description='nutrient (scalar source)')
    rate = xso.parameter(description='m_Z quadratic closure coeff '
                                     '[(mmol N m-3)^-1 d-1]')

    @xso.flux(dims='zoo')
    def mortality(self, population, rate):
        return rate * population * self.m.sum(population)

    @xso.flux
    def recycle_to_N(self, population, rate):
        total_Z = self.m.sum(population)
        return rate * total_Z * total_Z


# =============================================================================
# TEMPERATURE — Q10 scaling of growth and grazing (Cloern 2018)
# =============================================================================
# Box temperature as a forcing (constant per regime for steady-state R0; swap
# for a time-varying forcing for transient runs). Growth and grazing each scale
# by Q10^((T - T_ref)/10). Cloern (2018): growth Q10 = 1.62, grazing Q10 = 2.48
# (grazing rises with T faster than growth). T_ref = 20 °C (allometry reference).

@xso.component
class ConstantTemperatureForcing:
    """Constant box temperature [°C] as a forcing (per-regime for R0)."""
    forcing = xso.forcing(setup_func='forcing_setup',
                          description='box temperature [°C]')
    value = xso.parameter(description='temperature [°C]')

    def forcing_setup(self, value):
        @np.vectorize
        def f(t):
            return value
        return f


@xso.component
class MonodGrowth_T:
    """Monod nutrient uptake with Q10 temperature scaling on μ_max.

        U(s_i) = Q10^((T-T_ref)/10) · μ(s_i) · N · P_i / (N + k_s,i)

    Identical to MonodGrowth_NP plus the temperature factor (Cloern 2018).
    """
    resource = xso.variable(foreign=True, flux='uptake', negative=True,
                            description='dissolved nitrogen (scalar sink)')
    consumer = xso.variable(foreign=True, dims='phyto', flux='uptake',
                            negative=False, description='phyto (per-class source)')
    temperature = xso.forcing(foreign=True, description='box temperature [°C]')
    mu_max = xso.parameter(dims='phyto', description='max growth rate per class [d-1]')
    halfsat = xso.parameter(dims='phyto', description='nutrient half-sat per class')
    q10 = xso.parameter(description='growth Q10 (Cloern 2018: 1.62)')
    t_ref = xso.parameter(description='reference temperature [°C] (20)')

    @xso.flux(dims='phyto')
    def uptake(self, resource, consumer, temperature, mu_max, halfsat, q10, t_ref):
        f_T = q10 ** ((temperature - t_ref) / 10.0)
        return f_T * mu_max * resource * consumer / (resource + halfsat)


@xso.component
class DistributedGrazing_TypeIII_T:
    """DistributedGrazing_TypeIII with Q10 temperature scaling on I_max.

        G_kj = Q10^((T-T_ref)/10) · Imax_j · Z_j · φ_kj · B_k · S_j / (S_j² + KsZ_j²)

    Cloern (2018) grazing Q10 = 2.48 (> growth's 1.62). Same group-flux routing
    (DistributedGrazingRouter) as the non-temperature version.
    """
    resource = xso.variable(foreign=True, dims='phyto')
    consumer = xso.variable(foreign=True, dims='zoo')
    temperature = xso.forcing(foreign=True, description='box temperature [°C]')
    phiPZ = xso.parameter(dims=('full', 'zoo'),
                          description='feeding preference matrix (prey × predator)')
    Imax = xso.parameter(dims='zoo', description='per-class max ingestion [d-1]')
    KsZ = xso.parameter(dims='zoo', description='per-class grazing half-sat [mmol N m-3]')
    q10 = xso.parameter(description='grazing Q10 (Cloern 2018: 2.48)')
    t_ref = xso.parameter(description='reference temperature [°C] (20)')

    @xso.flux(group='graze_matrix', dims=('full', 'zoo'))
    def grazing(self, resource, consumer, temperature, phiPZ, Imax, KsZ, q10, t_ref):
        f_T = q10 ** ((temperature - t_ref) / 10.0)
        biomass = self.m.concatenate((resource, consumer))
        S = self.m.sum(phiPZ * biomass[:, None], axis=0)
        return f_T * Imax * consumer * phiPZ * biomass[:, None] * S / (S ** 2 + KsZ ** 2)
