
import numpy as np
import xso


# =============================================================================
# STATE VARIABLES
# =============================================================================

@xso.component
class Nutrient:
    """Dissolved inorganic nutrient (scalar)."""
    value = xso.variable(description='nutrient concentration',
                         attrs={'units': 'mmol N m-3'})


@xso.component
class PhytoSizeSpectrum:
    """Phytoplankton biomass across n size classes."""
    biomass = xso.variable(dims='phyto', description='phytoplankton biomass',
                           attrs={'units': 'mmol N m-3'})
    phyto = xso.index(dims='phyto', description='phytoplankton size classes',
                      attrs={'units': 'µm ESD'})


@xso.component
class ZooSizeSpectrum:
    """Zooplankton biomass across n size classes."""
    biomass = xso.variable(dims='zoo', description='zooplankton biomass',
                           attrs={'units': 'mmol N m-3'})
    zoo = xso.index(dims='zoo', description='zooplankton size classes',
                    attrs={'units': 'µm ESD'})


@xso.component
class Detritus:
    """Single scalar detrital nitrogen pool (mmol N m-3).

    Not size-structured. Represents sinking + suspended particulate
    organic nitrogen together.
    """
    value = xso.variable(description='detritus concentration',
                         attrs={'units': 'mmol N m-3'})



# =============================================================================
# FORCINGS
# =============================================================================

@xso.component
class ConstantExternalNutrient:
    """Provides a constant external nutrient forcing value."""
    forcing = xso.forcing(foreign=False, setup_func='forcing_setup',
                          description='external nutrient concentration')
    value = xso.parameter(description='constant nutrient value')

    def forcing_setup(self, value):
        @np.vectorize
        def forcing(time):
            return value
        return forcing


@xso.component
class ConstantFishForcing:
    """Provides a constant fish biomass as forcing."""
    forcing = xso.forcing(foreign=False, setup_func='forcing_setup',
                          description='prescribed fish biomass')
    value = xso.parameter(description='constant fish biomass value')

    def forcing_setup(self, value):
        @np.vectorize
        def forcing(time):
            return value
        return forcing


# =============================================================================
# NUTRIENT INFLOW
# =============================================================================

@xso.component
class LinearForcingInput:
    """Non-dimensional linear forcing input flux (chemostat nutrient supply)."""
    var = xso.variable(foreign=True, flux='input', negative=False,
                       description='variable receiving input')
    forcing = xso.forcing(foreign=True, description='forcing value')
    rate = xso.parameter(description='dilution/exchange rate')

    @xso.flux
    def input(self, var, forcing, rate):
        return forcing * rate


# =============================================================================
# PHYTOPLANKTON GROWTH
# =============================================================================

@xso.component
class MonodGrowth_SizeBased:
    """Monod (Michaelis-Menten) nutrient uptake, size-dependent."""
    resource = xso.variable(foreign=True, flux='uptake', negative=True)
    consumer = xso.variable(foreign=True, dims='phyto', flux='uptake', negative=False)

    halfsat = xso.parameter(dims='phyto', description='half-saturation constants')
    mu_max = xso.parameter(dims='phyto', description='maximum growth rates')

    @xso.flux(dims='phyto')
    def uptake(self, resource, consumer, halfsat, mu_max):
        return mu_max * resource / (resource + halfsat) * consumer



# =============================================================================
# GRAZING — HOLLING TYPE III
# =============================================================================
 
@xso.component
class SizebasedGrazingMatrix_Full_TypeIII:
    """Holling Type III grazing with full (P+Z) prey dimension.
 
    Following Mattern et al. (2026) / Dutkiewicz et al. (2015, 2020):
 
        S_j  = Σ_k  φ_kj · B_k
        G_ij = g_max_j · Z_j · φ_ij · B_i · S_j / (S_j² + K_graz²)
 
    At low prey (S_j << K_graz):  grazing ∝ S_j² ≈ 0  (refuge for rare prey)
    At high prey (S_j >> K_graz): grazing saturates like Type II
    """
    resource = xso.variable(foreign=True, dims='phyto')
    consumer = xso.variable(foreign=True, dims='zoo')
    phiPZ = xso.parameter(dims=('full', 'zoo'),
                          description='feeding preference matrix (prey x predator)')
    Imax = xso.parameter(dims='zoo', description='maximum ingestion rates')
    KsZ = xso.parameter(description='half-saturation of Type III grazing response')
 
    @xso.flux(group='graze_matrix', dims=('full', 'zoo'))
    def grazing(self, resource, consumer, phiPZ, Imax, KsZ):
        biomass = self.m.concatenate((resource, consumer))
        S_prey = self.m.sum(phiPZ * biomass[:, None], axis=0)
        FgrazMatrix = (Imax * consumer
                       * phiPZ * biomass[:, None]
                       * S_prey
                       / (S_prey ** 2 + KsZ ** 2))
        return FgrazMatrix
 

# =============================================================================
# GGE WITH DETRITUS ROUTING
# =============================================================================

@xso.component
class GGE_Full_SizeDep_withD:
    """Routes grazing fluxes with size-dependent GGE, splitting losses
    between detritus and nutrient recycling.

    Mass balance for each predator Z_j with total ingestion I_j:
        To Z_j biomass:  gge_j * I_j              (assimilation)
        To Detritus:     (1 - gge_j) * f_egest_D * I_j   (fecal pellets)
        To Nutrient:     (1 - gge_j) * (1 - f_egest_D) * I_j  (sloppy feeding / DOM)

    Prey removal (grazing_phyto, grazing_zoo) is unchanged relative to
    GGE_Full_SizeDep - we only change where the non-assimilated fraction
    of ingested material goes.

    Default f_egest_D = 0.75 follows Fasham et al. (1990): ~75% of
    non-assimilated ingestion is egested as fecal pellets (-> D), ~25%
    is lost as DOM / sloppy feeding (-> N immediately).
    """
    grazed_phyto = xso.variable(dims='phyto', foreign=True,
                                flux='grazing_phyto', negative=True)
    grazed_zoo = xso.variable(dims='zoo', foreign=True,
                              flux='grazing_zoo', negative=True)
    assimilated_consumer = xso.variable(dims='zoo', foreign=True,
                                        flux='assimilation')
    egested_detritus = xso.variable(foreign=True, flux='egestion_to_D')
    excreted_nutrient = xso.variable(foreign=True, flux='excretion_to_N')

    gge = xso.parameter(dims='zoo',
                        description='size-dependent gross growth efficiency')
    f_egest_D = xso.parameter(
        description='fraction of non-assimilated ingestion routed to D '
                    '(remainder goes to N as sloppy feeding / DOM)')

    @xso.flux(dims='phyto', group_to_arg='graze_matrix')
    def grazing_phyto(self, grazed_phyto, grazed_zoo, assimilated_consumer,
                      egested_detritus, excreted_nutrient, graze_matrix,
                      gge, f_egest_D):
        total_grazed_per_prey = self.m.sum(graze_matrix, axis=1)
        return total_grazed_per_prey[0:len(grazed_phyto)]

    @xso.flux(dims='zoo', group_to_arg='graze_matrix')
    def grazing_zoo(self, grazed_phyto, grazed_zoo, assimilated_consumer,
                    egested_detritus, excreted_nutrient, graze_matrix,
                    gge, f_egest_D):
        n_P = len(grazed_phyto)
        total_grazed_per_prey = self.m.sum(graze_matrix, axis=1)
        return total_grazed_per_prey[n_P:n_P + len(grazed_zoo)]

    @xso.flux(dims='zoo', group_to_arg='graze_matrix')
    def assimilation(self, grazed_phyto, grazed_zoo, assimilated_consumer,
                     egested_detritus, excreted_nutrient, graze_matrix,
                     gge, f_egest_D):
        total_ingested = self.m.sum(graze_matrix, axis=0)
        return total_ingested * gge

    @xso.flux(group_to_arg='graze_matrix')
    def egestion_to_D(self, grazed_phyto, grazed_zoo, assimilated_consumer,
                      egested_detritus, excreted_nutrient, graze_matrix,
                      gge, f_egest_D):
        total_ingested = self.m.sum(graze_matrix, axis=0)
        return self.m.sum(total_ingested * (1 - gge)) * f_egest_D

    @xso.flux(group_to_arg='graze_matrix')
    def excretion_to_N(self, grazed_phyto, grazed_zoo, assimilated_consumer,
                       egested_detritus, excreted_nutrient, graze_matrix,
                       gge, f_egest_D):
        total_ingested = self.m.sum(graze_matrix, axis=0)
        return self.m.sum(total_ingested * (1 - gge)) * (1 - f_egest_D)


# =============================================================================
# PHYTOPLANKTON MORTALITY WITH D ROUTING
# =============================================================================

@xso.component
class PhytoMortality_toD_toN:
    """Linear phytoplankton mortality, partitioned between D and N.

    Mortality flux out of P_i:  rate_i * P_i       (unchanged)
    Routed:
        To D:  f_mort_D * sum(rate * P)
        To N:  (1 - f_mort_D) * sum(rate * P)

    Default f_mort_D = 0.9 follows the NPZD convention (Fasham et al. 1990
    and descendants) that the bulk of phyto mortality (aggregation,
    senescence, viral lysis producing particulate debris) becomes
    particulate detritus, with a small fraction released as DOM / direct
    ammonification.
    """
    population = xso.variable(dims='phyto', foreign=True,
                              flux='mortality', negative=True)
    detritus = xso.variable(foreign=True, flux='mortality_to_D',
                            negative=False)
    nutrient = xso.variable(foreign=True, flux='mortality_to_N',
                            negative=False)

    rate = xso.parameter(dims='phyto', description='linear mortality rate')
    f_mort_D = xso.parameter(
        description='fraction of phyto mortality routed to D '
                    '(remainder goes directly to N)')

    @xso.flux(dims='phyto')
    def mortality(self, population, detritus, nutrient, rate, f_mort_D):
        return population * rate

    @xso.flux
    def mortality_to_D(self, population, detritus, nutrient, rate, f_mort_D):
        return self.m.sum(population * rate) * f_mort_D

    @xso.flux
    def mortality_to_N(self, population, detritus, nutrient, rate, f_mort_D):
        return self.m.sum(population * rate) * (1 - f_mort_D)


# =============================================================================
# ZOOPLANKTON QUADRATIC MORTALITY WITH D ROUTING
# =============================================================================

@xso.component
class ZooQuadraticMortality_toD:
    """Quadratic (Banas-style) zooplankton mortality, partitioned between
    D and export.

    Mortality flux out of Z_i:  rate * Z_i * sum(Z)  (unchanged)
    Routed:
        To D:      f_mort_D * total
        Exported:  (1 - f_mort_D) * total  (leaves the system)

    The exported fraction represents predation by higher trophic levels
    not explicitly resolved (fish, gelatinous predators) whose biomass
    is removed from the surface system. The D fraction represents
    unconsumed carcasses and aggregation of zooplankton debris.

    Default f_mort_D = 0.5: half of the closure term becomes detritus,
    half is exported. Stock et al. (2008) use f_mz4 = 0.5 for the same
    decomposition.
    """
    population = xso.variable(dims='zoo', foreign=True,
                              flux='mortality', negative=True)
    detritus = xso.variable(foreign=True, flux='mortality_to_D',
                            negative=False)

    rate = xso.parameter(description='quadratic mortality rate')
    f_mort_D = xso.parameter(
        description='fraction of zoo quadratic mortality routed to D '
                    '(remainder is exported out of the system)')

    @xso.flux(dims='zoo')
    def mortality(self, population, detritus, rate, f_mort_D):
        return rate * population * self.m.sum(population)

    @xso.flux
    def mortality_to_D(self, population, detritus, rate, f_mort_D):
        total_mortality = rate * self.m.sum(population) * self.m.sum(population)
        return total_mortality * f_mort_D


# =============================================================================
# DETRITUS REMINERALIZATION
# =============================================================================

@xso.component
class DetritusRemineralization:
    """Linear remineralization of detritus to dissolved nutrient.

        Remin flux:  k_remin * D    (D -> N)

    Fasham et al. (1990) use k_remin ~ 0.05 d-1; warm tropical systems
    (Cariaco surface ~25 C) can justify ~0.1 d-1.
    """
    detritus = xso.variable(foreign=True, flux='remineralization',
                            negative=True)
    nutrient = xso.variable(foreign=True, flux='remineralization',
                            negative=False)

    k_remin = xso.parameter(description='remineralization rate [d-1]')

    @xso.flux
    def remineralization(self, detritus, nutrient, k_remin):
        return k_remin * detritus


# =============================================================================
# DETRITUS SINKING (EXPORT OUT OF BOX)
# =============================================================================

@xso.component
class DetritusSinking:
    """Linear sinking loss of detritus out of the surface box.

        Sinking flux:  (w_sink / d_e) * D    (D -> out of system)

    The sinking rate parameter is the effective first-order loss rate
    for the box: w_sink / d_e. For d_e = 50 m and w_sink = 5 m/d,
    sinking_rate = 0.1 d-1.

    This flux is the primary validation diagnostic against the CARIACO
    152m sediment trap PON flux. To compare the flux from this component
    (units mmol N m-3 d-1) against trap flux (units mmol N m-2 d-1),
    multiply by d_e:

        trap_flux_model = sinking_rate * D * d_e  =  w_sink * D
    """
    detritus = xso.variable(foreign=True, flux='sinking', negative=True)

    sinking_rate = xso.parameter(
        description='effective D sinking loss rate = w_sink / d_e [d-1]')

    @xso.flux
    def sinking(self, detritus, sinking_rate):
        return sinking_rate * detritus


# =============================================================================
# FISH GRAZING — LOG-NORMAL KERNEL IN ESD SPACE (Option A)
# =============================================================================

@xso.component
class FishGrazing_Kernel:
    """Sardine grazing as external forcing with a log-normal size kernel.

    Imposes a size-selective mortality on P and Z of the form

        μ_S(D', t) = rate * F(t) * K(D') * B(D')

    where F(t) is the prescribed fish biomass forcing, B(D') is the prey
    biomass in the size class with ESD D', and K(D') is a log-normal
    selectivity kernel in log10(ESD) space (precomputed via
    compute_fish_kernel_lognormal and passed as kernel_P / kernel_Z).

    The kernel is normalized to peak = 1 at D_pref, so `rate` retains its
    meaning as the maximum mass-specific grazing rate per unit fish biomass
    (units: [fish biomass]^-1 d^-1) imposed on prey at the preferred size.

    Grazed material leaves the system entirely (locked into fish stock,
    eventually exported as catch). To recycle a fraction to N, model this
    on FishGrazingForced.

    # References:
    # - Log-normal size-selection kernel: Ursin (1973), as used in
    #   Andersen & Pedersen (2010, Eq. M2) and Heneghan et al. (2016, Eq. E4).
    # - Kernel formulated in log10(ESD) space following the plankton
    #   size-spectrum convention of Banas (2011).
    # - Use of a broad kernel (rather than narrow β~100, σ~1 typical of
    #   generic fish) reflects sardine-specific filter-feeding biology:
    #   sardines retain the capacity to feed on prey across ~2 decades of
    #   ESD, from <20 µm up to ~2 mm (van der Lingen 1994; van der Lingen
    #   et al. 2006; Rykaczewski 2019). The planktivore-specific broadened
    #   kernel approach follows Andrades (2012, Ch. 3, Eqs. 3.12–3.13).
    # - External-forcing (F(t) · K(D')) closure structure follows the
    #   standard size-selective predation closure used in plankton
    #   size-spectrum models (e.g. Stock et al. 2008; Banas 2011).
    """
    phyto = xso.variable(dims='phyto', foreign=True,
                         flux='fish_graze_phyto', negative=True)
    zoo = xso.variable(dims='zoo', foreign=True,
                       flux='fish_graze_zoo', negative=True)

    fish_forcing = xso.forcing(foreign=True,
                               description='prescribed fish biomass')

    kernel_P = xso.parameter(dims='phyto',
                             description='log-normal selectivity weights on P (peak=1)')
    kernel_Z = xso.parameter(dims='zoo',
                             description='log-normal selectivity weights on Z (peak=1)')
    rate = xso.parameter(description='peak fish grazing rate per unit fish biomass')

    @xso.flux(dims='phyto')
    def fish_graze_phyto(self, phyto, zoo, fish_forcing,
                         kernel_P, kernel_Z, rate):
        return rate * fish_forcing * kernel_P * phyto

    @xso.flux(dims='zoo')
    def fish_graze_zoo(self, phyto, zoo, fish_forcing,
                       kernel_P, kernel_Z, rate):
        return rate * fish_forcing * kernel_Z * zoo