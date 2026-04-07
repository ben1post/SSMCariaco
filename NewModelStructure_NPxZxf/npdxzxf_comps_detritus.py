"""
Detritus Extension Components for NPxZxF → NPDxZxF
===================================================

Adds a single scalar detritus pool D to the size-structured plankton
model, with sources from phytoplankton mortality, zooplankton egestion
(the non-assimilated fraction of ingestion), and zooplankton mortality.
Sinks are linear remineralization to N and linear sinking out of the
box.

Design follows:
- Fasham et al. (1990), J. Mar. Res. 48, 591-639: routing of phyto
  mortality and zoo egestion to a detrital pool, with a fraction of
  zoo egestion (sloppy feeding) going directly to dissolved nutrients.
- Stock et al. (2008), J. Mar. Sys. 74, 134-152: size-dependent partition
  of zoo egestion between recycling and export; partial export of zoo
  mortality.
- Standard NPZD closure for D dynamics: linear remin (rate k_remin) and
  linear sinking loss (w_sink / d_e).

These components are drop-in replacements for
    GGE_Full_SizeDep       -> GGE_Full_SizeDep_withD
    PhytoMortality_toN     -> PhytoMortality_toD_toN
    ZooQuadraticMortality  -> ZooQuadraticMortality_toD
plus three new components:
    Detritus                  (state variable)
    DetritusRemineralization  (D -> N)
    DetritusSinking           (D -> out of system; trap validation target)

Fish grazing (FishGrazing_Lognormal) is unchanged - fish biomass leaves
the system entirely, not via D.
"""

import numpy as np
import xso


# =============================================================================
# STATE VARIABLE
# =============================================================================

@xso.component
class Detritus:
    """Single scalar detrital nitrogen pool (mmol N m-3).

    Not size-structured. Represents sinking + suspended particulate
    organic nitrogen together; a single effective pool is sufficient
    when the model is calibrated against bulk PON flux rather than
    size-fractionated flux.
    """
    value = xso.variable(description='detritus concentration',
                         attrs={'units': 'mmol N m-3'})


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