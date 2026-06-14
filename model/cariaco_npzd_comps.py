"""
Cariaco N-P-Z-D size-spectrum model — COMPONENTS (final baseline, 2026-06-09).

Self-contained component set for the corrected previous-MS3 baseline:
N-P-Z-D + detritus, distributed Holling Type III grazing (omnivory), Stock
F_N/d_e supply, Cloern temperature on growth + grazing, Rykaczewski sardine
grazing (scalar rate), single-source allometries (Banas growth / Dutkiewicz
grazing — set in the setups file). Full spec: model context/MS3_Final_Baseline_Model.tex.

Loss-fate design (2026-06-09, Benny): every loss acting on P and Z — phyto
linear mortality, zoo linear mortality, zoo quadratic closure, and the
unassimilated (1-GGE) fraction of grazing — is FREELY ROUTABLE three ways:
to N (recycle), to D (detritus), and exported (removed from system). Each
routing component takes `frac_D` and `frac_export`; the remainder
`frac_N = 1 - frac_D - frac_export` goes to N. Set fractions to put a loss
anywhere on the N/D/export simplex. NB: NONE of the mortality/closure terms
are temperature-modulated — temperature (Cloern Q10) acts on growth + grazing
ONLY (Dutkiewicz's temperature-modulated closure is deliberately not adopted).

References: Stock et al. (2008) Prog. Oceanogr. 76:189 (F_N/d_e supply,
quadratic closure); Banas (2011) Ecol. Modelling 222:2663 (growth allometry,
quadratic closure); Dutkiewicz et al. (2020) Biogeosciences 17:609 (grazing,
Type III); Mattern et al. (2026) L&O (kernel σ=0.15); Cloern (2018) (Q10);
Fasham et al. (1990) JMR 48:591 (detritus routing); Rykaczewski (2019) MEPS
617:165 (sardine clearance curve).
"""

import numpy as np
import xso


# =============================================================================
# KERNEL HELPERS (setup-time array builders)
# =============================================================================

def compute_grazing_kernel(phyto_esd, zoo_esd, mode='omni',
                           theta_opt=10.0, sigma_log=0.15, convention='mattern'):
    """Feeding-preference matrix phiPZ of shape (n_P + n_Z, n_Z).

    mode : {'matched', 'herb', 'omni'}; the baseline uses 'omni' (zoo graze
        P AND Z; within-class cannibalism zeroed).
    convention : 'mattern' -> exp(-((Δ)/σ)²), Mattern (2026) / Dutkiewicz (2020),
        default σ=0.15 (≡ Gaussian std 0.106) — the baseline kernel.
        '2sigma2' -> exp(-(Δ)²/(2σ²)) (legacy MS3, σ=0.25).
    theta_opt : predator:prey ESD ratio at the kernel peak (1:10).
    """
    phyto_esd = np.asarray(phyto_esd)
    zoo_esd = np.asarray(zoo_esd)
    n_P, n_Z = len(phyto_esd), len(zoo_esd)
    prey_esd = np.concatenate([phyto_esd, zoo_esd])
    log_ratio = np.log10(zoo_esd[None, :] / prey_esd[:, None])
    delta = log_ratio - np.log10(theta_opt)
    if convention == 'mattern':
        kernel = np.exp(-(delta / sigma_log) ** 2)
    elif convention == '2sigma2':
        kernel = np.exp(-(delta ** 2) / (2.0 * sigma_log ** 2))
    else:
        raise ValueError(f"convention must be '2sigma2' or 'mattern', got {convention!r}")
    phiPZ = np.zeros((n_P + n_Z, n_Z))
    if mode == 'matched':
        for j in range(n_Z):
            phiPZ[j, j] = 1.0
    elif mode == 'herb':
        phiPZ[:n_P, :] = kernel[:n_P, :]
    elif mode == 'omni':
        phiPZ[:] = kernel
        for j in range(n_Z):
            phiPZ[n_P + j, j] = 0.0   # no within-class cannibalism
    else:
        raise ValueError(f"mode must be 'matched'/'herb'/'omni', got {mode!r}")
    return phiPZ


def clearance_rate_sardine_vdl(prey_length_um, filter_feeding=False):
    """Sardine size-specific clearance rate, Rykaczewski (2019) Eq. 3.

    Sum of two logistic terms (fine filter ~15 µm, coarse retention ~800 µm).
    filter_feeding=True clamps flat above 1230 µm (not used in the baseline).
    Prey length in µm (copepod prosome length ~2-3× ESD — apply a length
    correction to zoo ESDs if desired; baseline passes ESD directly).
    """
    x = np.asarray(prey_length_um, dtype=float)

    def _f(xv):
        e1 = np.exp(0.0198 * (xv - 15.0))
        term1 = (9.03 * e1) / (12.03 + 0.75 * e1)
        e2 = np.exp(0.00843 * (xv - 800.0))
        term2 = (9.96 * e2) / (30.8 + 0.323 * e2)
        return term1 + term2

    F_S = _f(x)
    if filter_feeding:
        F_S = np.where(x > 1230.0, _f(np.array(1230.0)), F_S)
    return F_S


def compute_fish_kernel_vdl_joint(phyto_esd, zoo_esd):
    """Rykaczewski sardine kernel on P and Z grids, jointly peak-normalised
    (single shared max, so the curve's relative P-vs-Z weighting is preserved;
    the curve peaks ~1230 µm, inside the zoo grid)."""
    F_P = clearance_rate_sardine_vdl(phyto_esd)
    F_Z = clearance_rate_sardine_vdl(zoo_esd)
    F_max = max(F_P.max(), F_Z.max())
    return F_P / F_max, F_Z / F_max


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
    """Zooplankton biomass across n_zoo log-spaced size classes (zoo_esd = 10·phyto_esd)."""
    biomass = xso.variable(dims='zoo', description='zooplankton biomass',
                           attrs={'units': 'mmol N m-3'})
    zoo_esd = xso.index(dims='zoo', as_parameter=True,
                        description='zoo size classes (ESD)',
                        attrs={'units': 'µm'})


@xso.component
class Detritus:
    """Single scalar detrital nitrogen pool (PON + suspended POM)."""
    value = xso.variable(description='detritus concentration',
                         attrs={'units': 'mmol N m-3'})


# =============================================================================
# FORCINGS
# =============================================================================

@xso.component
class StockNutrientSupply:
    """New-nutrient supply, Stock 2008 Eq. 7:  J = F_N / d_e  [mmol N m-3 d-1].
    F_N = new-N flux per area [mmol N m-2 d-1]; d_e = box depth [m]."""
    var = xso.variable(foreign=True, flux='input', negative=False,
                       description='nutrient pool receiving new-N supply')
    FN = xso.parameter(description='new-nutrient flux F_N [mmol N m-2 d-1]')
    de = xso.parameter(broadcast=True,
                       description='euphotic box depth d_e [m] — broadcast; SINGLE '
                                   'source of truth (DetritusSink foreign-references it, '
                                   'so one regime d_e drives both F_N/d_e and w_sink/d_e)')

    @xso.flux
    def input(self, var, FN, de):
        return FN / de


@xso.component
class ConstantTemperatureForcing:
    """Constant box temperature [°C] as a forcing (per-regime for steady state)."""
    forcing = xso.forcing(setup_func='forcing_setup',
                          description='box temperature [°C]')
    value = xso.parameter(description='temperature [°C]')

    def forcing_setup(self, value):
        @np.vectorize
        def f(t):
            return value
        return f


# =============================================================================
# PHYTO GROWTH — MONOD UPTAKE WITH Q10 (Banas allometry, Cloern temperature)
# =============================================================================

@xso.component
class MonodGrowth_T:
    """Monod nutrient uptake with Q10 temperature scaling on μ_max.

        U_i = Q10^((T-T_ref)/10) · μ_max,i · N · P_i / (N + K_s,i)
    """
    resource = xso.variable(foreign=True, flux='uptake', negative=True,
                            description='dissolved nitrogen (scalar sink)')
    consumer = xso.variable(foreign=True, dims='phyto', flux='uptake',
                            negative=False, description='phyto (per-class source)')
    temperature = xso.forcing(foreign=True, description='box temperature [°C]')
    mu_max = xso.parameter(broadcast=True, dims='phyto',
                           description='max growth rate per class [d-1] (broadcast so a '
                                       'mortality component can foreign-reference it)')
    halfsat = xso.parameter(dims='phyto', description='nutrient half-sat per class')
    q10 = xso.parameter(description='growth Q10 (Cloern 2018: 1.62)')
    t_ref = xso.parameter(description='reference temperature [°C] (20)')

    @xso.flux(dims='phyto')
    def uptake(self, resource, consumer, temperature, mu_max, halfsat, q10, t_ref):
        f_T = q10 ** ((temperature - t_ref) / 10.0)
        return f_T * mu_max * resource * consumer / (resource + halfsat)


# =============================================================================
# GRAZING — DISTRIBUTED HOLLING TYPE III WITH Q10 (Dutkiewicz Eq. S1.6)
# =============================================================================

@xso.component
class DistributedGrazing_TypeIII_T:
    """Distributed (kernel) Holling Type III grazing with Q10 on I_max.

        S_j  = Σ_k φ_kj · B_k                        (B = [P; Z], community prey)
        G_kj = Q10^((T-T_ref)/10) · I_max,j · Z_j · φ_kj · B_k · S_j / (S_j² + K_sZ²)

    Community-saturated Type III (matches Dutkiewicz 2020 Eq. S1.6 exactly).
    Publishes the (n_P+n_Z, n_Z) matrix to the 'graze_matrix' group;
    DistributedGrazingRouter_route consumes it. Cloern grazing Q10 = 2.48.
    """
    resource = xso.variable(foreign=True, dims='phyto')
    consumer = xso.variable(foreign=True, dims='zoo')
    temperature = xso.forcing(foreign=True, description='box temperature [°C]')
    phiPZ = xso.parameter(dims=('full', 'zoo'),
                          description='feeding preference matrix (prey × predator)')
    Imax = xso.parameter(dims='zoo', description='per-class max ingestion [d-1]')
    KsZ = xso.parameter(dims='zoo', description='grazing half-sat [mmol N m-3] (uniform)')
    q10 = xso.parameter(description='grazing Q10 (Cloern 2018: 2.48)')
    t_ref = xso.parameter(description='reference temperature [°C] (20)')

    @xso.flux(group='graze_matrix', dims=('full', 'zoo'))
    def grazing(self, resource, consumer, temperature, phiPZ, Imax, KsZ, q10, t_ref):
        f_T = q10 ** ((temperature - t_ref) / 10.0)
        biomass = self.m.concatenate((resource, consumer))
        S = self.m.sum(phiPZ * biomass[:, None], axis=0)
        return f_T * Imax * consumer * phiPZ * biomass[:, None] * S / (S ** 2 + KsZ ** 2)


@xso.component
class DistributedGrazingRouter_route:
    """Route the 'graze_matrix' group: assimilate GGE·I to Z, remove grazed
    biomass from P and Z, and route the unassimilated (1-GGE) fraction
    three ways (N / D / export).

        prey loss (P):  Σ_pred G_kj                      (per phyto class)
        prey loss (Z):  Σ_pred G_kj                      (per zoo class; omnivory)
        gain to Z:      GGE · Σ_prey G_kj                 (per predator)
        to D:           frac_D · (1-GGE) · ΣG             (egestion/fecal)
        to N:           (1-frac_D-frac_export) · (1-GGE) · ΣG  (excretion/DOM)
        exported:       frac_export · (1-GGE) · ΣG        (leaves system; no flux)
    """
    grazed_phyto = xso.variable(foreign=True, dims='phyto',
                                flux='loss_P', negative=True,
                                description='phytoplankton (per-class sink)')
    grazed_zoo = xso.variable(foreign=True, dims='zoo',
                              flux='loss_Z', negative=True,
                              description='zoo-as-prey (per-class sink; omnivory)')
    assimilated_consumer = xso.variable(foreign=True, dims='zoo',
                                        flux='gain_Z',
                                        description='zooplankton (per-class source)')
    egested_detritus = xso.variable(foreign=True, flux='egestion_to_D',
                                    description='detritus (scalar source)')
    excreted_nutrient = xso.variable(foreign=True, flux='excretion_to_N',
                                     description='nutrient (scalar source)')
    gge = xso.parameter(description='GGE — gross growth efficiency (scalar)')
    frac_D = xso.parameter(description='fraction of unassimilated grazing -> D')
    frac_export = xso.parameter(description='fraction of unassimilated grazing exported '
                                            '(remainder -> N)')

    @xso.flux(dims='phyto', group_to_arg='graze_matrix')
    def loss_P(self, grazed_phyto, grazed_zoo, assimilated_consumer,
               egested_detritus, excreted_nutrient, graze_matrix, gge, frac_D, frac_export):
        return self.m.sum(graze_matrix, axis=1)[0:len(grazed_phyto)]

    @xso.flux(dims='zoo', group_to_arg='graze_matrix')
    def loss_Z(self, grazed_phyto, grazed_zoo, assimilated_consumer,
               egested_detritus, excreted_nutrient, graze_matrix, gge, frac_D, frac_export):
        n_P = len(grazed_phyto)
        return self.m.sum(graze_matrix, axis=1)[n_P:n_P + len(grazed_zoo)]

    @xso.flux(dims='zoo', group_to_arg='graze_matrix')
    def gain_Z(self, grazed_phyto, grazed_zoo, assimilated_consumer,
               egested_detritus, excreted_nutrient, graze_matrix, gge, frac_D, frac_export):
        return self.m.sum(graze_matrix, axis=0) * gge

    @xso.flux(group_to_arg='graze_matrix')
    def egestion_to_D(self, grazed_phyto, grazed_zoo, assimilated_consumer,
                      egested_detritus, excreted_nutrient, graze_matrix, gge, frac_D, frac_export):
        return (1.0 - gge) * self.m.sum(graze_matrix) * frac_D

    @xso.flux(group_to_arg='graze_matrix')
    def excretion_to_N(self, grazed_phyto, grazed_zoo, assimilated_consumer,
                       egested_detritus, excreted_nutrient, graze_matrix, gge, frac_D, frac_export):
        return (1.0 - gge) * self.m.sum(graze_matrix) * (1.0 - frac_D - frac_export)


# =============================================================================
# MORTALITY / CLOSURE — each freely routable N / D / export
# =============================================================================

@xso.component
class PhytoMortality_route:
    """Linear phyto mortality m_P·P, routed N / D / export.
    Default (frac_D=0.9, frac_export=0) -> 90% D, 10% N (Fasham-style)."""
    population = xso.variable(foreign=True, dims='phyto', flux='mortality', negative=True)
    detritus = xso.variable(foreign=True, flux='mortality_to_D', negative=False)
    nutrient = xso.variable(foreign=True, flux='mortality_to_N', negative=False)
    rate = xso.parameter(dims='phyto', description='linear phyto mortality rate m_P [d-1]')
    frac_D = xso.parameter(description='fraction of phyto mortality -> D')
    frac_export = xso.parameter(description='fraction exported (remainder -> N)')

    @xso.flux(dims='phyto')
    def mortality(self, population, detritus, nutrient, rate, frac_D, frac_export):
        return rate * population

    @xso.flux
    def mortality_to_D(self, population, detritus, nutrient, rate, frac_D, frac_export):
        return self.m.sum(rate * population) * frac_D

    @xso.flux
    def mortality_to_N(self, population, detritus, nutrient, rate, frac_D, frac_export):
        return self.m.sum(rate * population) * (1.0 - frac_D - frac_export)


@xso.component
class BanasPhytoMortality_route:
    """Banas (2011) non-grazing mortality, m_P = coeff·μ_max·P, with μ_max
    FOREIGN-referenced (broadcast) from the growth component so it always tracks
    the actual growth allometry in use. Banas DEFINES mortality as a fraction of
    the max growth rate, so it must follow μ_max rather than be a frozen array;
    that is the difference from PhytoMortality_route (which takes a fixed `rate`).
    `coeff` is a scalar (0.1 = Banas's 10%) and so is a clean scan axis. Routed
    N / D / export identically to PhytoMortality_route."""
    population = xso.variable(foreign=True, dims='phyto', flux='mortality', negative=True)
    detritus = xso.variable(foreign=True, flux='mortality_to_D', negative=False)
    nutrient = xso.variable(foreign=True, flux='mortality_to_N', negative=False)
    mu_max = xso.parameter(foreign=True, dims='phyto',
                           description='max growth rate (foreign, from the growth component)')
    coeff = xso.parameter(description='Banas mortality coefficient (0.1 = 10% of μ_max)')
    frac_D = xso.parameter(description='fraction of phyto mortality -> D')
    frac_export = xso.parameter(description='fraction exported (remainder -> N)')

    @xso.flux(dims='phyto')
    def mortality(self, population, detritus, nutrient, mu_max, coeff, frac_D, frac_export):
        return coeff * mu_max * population

    @xso.flux
    def mortality_to_D(self, population, detritus, nutrient, mu_max, coeff, frac_D, frac_export):
        return self.m.sum(coeff * mu_max * population) * frac_D

    @xso.flux
    def mortality_to_N(self, population, detritus, nutrient, mu_max, coeff, frac_D, frac_export):
        return self.m.sum(coeff * mu_max * population) * (1.0 - frac_D - frac_export)


@xso.component
class ZooLinearMortality_route:
    """Linear zoo mortality/excretion m_Zlin·Z, routed N / D / export.
    Basic-setup default (frac_D=1.0, frac_export=0) -> 100% D (Benny, 2026-06-09)."""
    population = xso.variable(foreign=True, dims='zoo', flux='mortality', negative=True)
    detritus = xso.variable(foreign=True, flux='mortality_to_D', negative=False)
    nutrient = xso.variable(foreign=True, flux='mortality_to_N', negative=False)
    rate = xso.parameter(description='linear zoo mortality/excretion rate m_Zlin [d-1]')
    frac_D = xso.parameter(description='fraction of linear zoo mortality -> D')
    frac_export = xso.parameter(description='fraction exported (remainder -> N)')

    @xso.flux(dims='zoo')
    def mortality(self, population, detritus, nutrient, rate, frac_D, frac_export):
        return rate * population

    @xso.flux
    def mortality_to_D(self, population, detritus, nutrient, rate, frac_D, frac_export):
        return self.m.sum(rate * population) * frac_D

    @xso.flux
    def mortality_to_N(self, population, detritus, nutrient, rate, frac_D, frac_export):
        return self.m.sum(rate * population) * (1.0 - frac_D - frac_export)


@xso.component
class ZooQuadraticMortality_route:
    """Quadratic (Banas) zoo closure m_Z·Z_j·ΣZ, routed N / D / export.
    Default (frac_D=0.5, frac_export=0.5) -> 50% D, 50% export (Stock-style).
    The exported fraction = unresolved higher predation; D fraction = carcasses."""
    population = xso.variable(foreign=True, dims='zoo', flux='mortality', negative=True)
    detritus = xso.variable(foreign=True, flux='mortality_to_D', negative=False)
    nutrient = xso.variable(foreign=True, flux='mortality_to_N', negative=False)
    rate = xso.parameter(description='quadratic closure coeff m_Z [(mmol N m-3)^-1 d-1]')
    frac_D = xso.parameter(description='fraction of quadratic closure -> D')
    frac_export = xso.parameter(description='fraction exported (remainder -> N)')

    @xso.flux(dims='zoo')
    def mortality(self, population, detritus, nutrient, rate, frac_D, frac_export):
        return rate * population * self.m.sum(population)

    @xso.flux
    def mortality_to_D(self, population, detritus, nutrient, rate, frac_D, frac_export):
        total = rate * self.m.sum(population) * self.m.sum(population)
        return total * frac_D

    @xso.flux
    def mortality_to_N(self, population, detritus, nutrient, rate, frac_D, frac_export):
        total = rate * self.m.sum(population) * self.m.sum(population)
        return total * (1.0 - frac_D - frac_export)


@xso.component
class ZooQuadraticMortality_perclass_route:
    """Per-class quadratic zoo closure m_Z·Z_j², routed N / D / export.

    Variant of ZooQuadraticMortality_route — replaces bulk Z_j·ΣZ with
    per-class Z_j² so each zoo class self-regulates without inter-class
    coupling through the total zoo biomass. Default routing matches the
    bulk form (frac_D=0.5, frac_export=0.5 = Stock-style)."""
    population = xso.variable(foreign=True, dims='zoo', flux='mortality', negative=True)
    detritus = xso.variable(foreign=True, flux='mortality_to_D', negative=False)
    nutrient = xso.variable(foreign=True, flux='mortality_to_N', negative=False)
    rate = xso.parameter(description='per-class quadratic coeff m_Z [(mmol N m-3)^-1 d-1]')
    frac_D = xso.parameter(description='fraction of quadratic closure -> D')
    frac_export = xso.parameter(description='fraction exported (remainder -> N)')

    @xso.flux(dims='zoo')
    def mortality(self, population, detritus, nutrient, rate, frac_D, frac_export):
        return rate * population * population

    @xso.flux
    def mortality_to_D(self, population, detritus, nutrient, rate, frac_D, frac_export):
        total = rate * self.m.sum(population * population)
        return total * frac_D

    @xso.flux
    def mortality_to_N(self, population, detritus, nutrient, rate, frac_D, frac_export):
        total = rate * self.m.sum(population * population)
        return total * (1.0 - frac_D - frac_export)


# =============================================================================
# DETRITUS — REMINERALIZATION (-> N) AND SINKING (export)
# =============================================================================

@xso.component
class DetritusRemineralization:
    """Linear remineralization k_remin · D  (D -> N). Warm tropical ~0.1 d-1."""
    detritus = xso.variable(foreign=True, flux='remineralization', negative=True)
    nutrient = xso.variable(foreign=True, flux='remineralization', negative=False)
    k_remin = xso.parameter(description='remineralization rate [d-1]')

    @xso.flux
    def remineralization(self, detritus, nutrient, k_remin):
        return k_remin * detritus


@xso.component
class DetritusSinking:
    """Detritus sinking out of the box: (w_sink/d_e) · D  (D -> export).
    d_e is foreign-referenced from StockNutrientSupply (broadcast), so one regime
    d_e is the single source of truth driving both supply (F_N/d_e) and sinking
    (w_sink/d_e); w_sink is a true constant. The 152-m-trap-equivalent flux is
    (w_sink/d_e)·D·d_e = w_sink·D."""
    detritus = xso.variable(foreign=True, flux='sinking', negative=True)
    w_sink = xso.parameter(description='detritus sinking velocity w_sink [m d-1]')
    de = xso.parameter(foreign=True,
                       description='euphotic box depth d_e [m] (shared from Inflow)')

    @xso.flux
    def sinking(self, detritus, w_sink, de):
        return (w_sink / de) * detritus


# =============================================================================
# FISH GRAZING — RYKACZEWSKI KERNEL, SCALAR RATE (top-down lever)
# =============================================================================

@xso.component
class FishGrazing_Kernel_rate:
    """Sardine predation as a size-selective, one-way-export mortality on P and Z.

        L_P,i = r_F · kernel_P,i · P_i        L_Z,j = r_F · kernel_Z,j · Z_j

    kernel_P/kernel_Z = jointly peak-normalised Rykaczewski (2019) clearance
    curve (compute_fish_kernel_vdl_joint). r_F is the scalar top-down lever
    (no fish state variable; r_F folds in the constant fish biomass). Grazed
    material leaves the system (locked into fish stock / catch). r_F = 0
    recovers the no-fish baseline.
    """
    phyto = xso.variable(dims='phyto', foreign=True, flux='fish_graze_phyto', negative=True)
    zoo = xso.variable(dims='zoo', foreign=True, flux='fish_graze_zoo', negative=True)
    kernel_P = xso.parameter(dims='phyto', description='Rykaczewski selectivity on P (peak=1)')
    kernel_Z = xso.parameter(dims='zoo', description='Rykaczewski selectivity on Z (peak=1)')
    rate = xso.parameter(description='fish grazing rate r_F (top-down lever) [d-1]')

    @xso.flux(dims='phyto')
    def fish_graze_phyto(self, phyto, zoo, kernel_P, kernel_Z, rate):
        return rate * kernel_P * phyto

    @xso.flux(dims='zoo')
    def fish_graze_zoo(self, phyto, zoo, kernel_P, kernel_Z, rate):
        return rate * kernel_Z * zoo
