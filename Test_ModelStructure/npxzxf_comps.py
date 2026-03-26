"""
NPxZxf Model Components (no Detritus)
======================================
XSO components for a symmetric size-structured plankton model
with fish as prescribed forcing.

Model structure: N - P_1...P_n - Z_1...Z_n, with F as forcing

Based on:
- Banas et al. (2011) size-spectral grazing kernel
- Stock et al. (2008) intraguild predation via 'full' dimension
"""

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


@xso.component
class ChemostatInput:
    """Chemostat nutrient supply: flux = d * (N0 - N)."""
    var = xso.variable(foreign=True, flux='input')
    forcing = xso.forcing(foreign=True, description='external concentration N0')
    rate = xso.parameter(description='exchange rate d')

    @xso.flux
    def input(self, var, forcing, rate):
        return rate * (forcing - var)
        

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
# SIZE-BASED GRAZING WITH INTRAGUILD PREDATION
# =============================================================================

@xso.component
class SizebasedGrazingMatrix_Full_TypeII:
    """Banas-style Holling Type II grazing with full (P+Z) prey dimension.

    Computes a grazing matrix G of dims (n_P+n_Z, n_Z), where each entry
    G_ij is the rate at which predator Z_j consumes prey item i (from the
    concatenated P and Z biomass vector).
    """
    resource = xso.variable(foreign=True, dims='phyto')
    consumer = xso.variable(foreign=True, dims='zoo')
    phiPZ = xso.parameter(dims=('full', 'zoo'),
                          description='feeding preference matrix (prey x predator)')
    Imax = xso.parameter(dims='zoo', description='maximum ingestion rates')
    KsZ = xso.parameter(description='half-saturation constant of grazing')

    @xso.flux(group='graze_matrix', dims=('full', 'zoo'))
    def grazing(self, resource, consumer, phiPZ, Imax, KsZ):
        biomass = self.m.concatenate((resource, consumer))
        BscaledAsFood = phiPZ * biomass[:, None] / KsZ
        FgrazMatrix = (Imax * consumer * BscaledAsFood
                       / (1 + self.m.sum(BscaledAsFood, axis=0)))
        return FgrazMatrix


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
# GROSS GROWTH EFFICIENCY — SIZE-DEPENDENT, RECYCLING TO N
# =============================================================================

@xso.component
class GGE_Full_SizeDep:
    """Routes grazing fluxes with size-dependent GGE.

    Mass balance for each predator Z_j:
        Total ingested:  I_j = sum_i(G_ij)
        To Z_j biomass:  gge_j * I_j
        To Nutrient:     (1 - gge_j) * I_j   (immediate recycling)
    """
    grazed_phyto = xso.variable(dims='phyto', foreign=True,
                                flux='grazing_phyto', negative=True)
    grazed_zoo = xso.variable(dims='zoo', foreign=True,
                              flux='grazing_zoo', negative=True)
    assimilated_consumer = xso.variable(dims='zoo', foreign=True,
                                        flux='assimilation')
    recycled_nutrient = xso.variable(foreign=True, flux='excretion')

    gge = xso.parameter(dims='zoo',
                        description='size-dependent gross growth efficiency')

    @xso.flux(dims='phyto', group_to_arg='graze_matrix')
    def grazing_phyto(self, grazed_phyto, grazed_zoo, assimilated_consumer,
                      recycled_nutrient, graze_matrix, gge):
        total_grazed_per_prey = self.m.sum(graze_matrix, axis=1)
        return total_grazed_per_prey[0:len(grazed_phyto)]

    @xso.flux(dims='zoo', group_to_arg='graze_matrix')
    def grazing_zoo(self, grazed_phyto, grazed_zoo, assimilated_consumer,
                    recycled_nutrient, graze_matrix, gge):
        n_P = len(grazed_phyto)
        total_grazed_per_prey = self.m.sum(graze_matrix, axis=1)
        return total_grazed_per_prey[n_P:n_P + len(grazed_zoo)]

    @xso.flux(dims='zoo', group_to_arg='graze_matrix')
    def assimilation(self, grazed_phyto, grazed_zoo, assimilated_consumer,
                     recycled_nutrient, graze_matrix, gge):
        total_ingested = self.m.sum(graze_matrix, axis=0)
        return total_ingested * gge

    @xso.flux(group_to_arg='graze_matrix')
    def excretion(self, grazed_phyto, grazed_zoo, assimilated_consumer,
                  recycled_nutrient, graze_matrix, gge):
        total_ingested = self.m.sum(graze_matrix, axis=0)
        return self.m.sum(total_ingested * (1 - gge))


# =============================================================================
# MORTALITY
# =============================================================================

@xso.component
class PhytoMortality_toN:
    """Linear phytoplankton mortality, recycled to nutrient."""
    population = xso.variable(dims='phyto', foreign=True,
                              flux='mortality', negative=True)
    nutrient = xso.variable(foreign=True, flux='recycled_mort', negative=False)

    rate = xso.parameter(dims='phyto', description='linear mortality rate')

    @xso.flux(dims='phyto')
    def mortality(self, population, nutrient, rate):
        return population * rate

    @xso.flux
    def recycled_mort(self, population, nutrient, rate):
        return self.m.sum(population * rate)


@xso.component
class ZooQuadraticMortality:
    """Quadratic zooplankton mortality (Banas-style).

    Mortality = rate * Z_i * sum(Z). Leaves the system (closure term).
    """
    population = xso.variable(dims='zoo', foreign=True,
                              flux='mortality', negative=True)

    rate = xso.parameter(description='quadratic mortality rate')

    @xso.flux(dims='zoo')
    def mortality(self, population, rate):
        return rate * population * self.m.sum(population)


# =============================================================================
# FISH GRAZING — FORCED
# =============================================================================

@xso.component
class FishGrazingForced:
    """Linear fish predation on large P and Z, with F as a prescribed forcing.

    The assimilated fraction (eps_F) leaves the system (locked in fish);
    the remainder is recycled to nutrient.
    """
    phyto = xso.variable(dims='phyto', foreign=True,
                         flux='fish_graze_phyto', negative=True)
    zoo = xso.variable(dims='zoo', foreign=True,
                       flux='fish_graze_zoo', negative=True)
    nutrient = xso.variable(foreign=True, flux='fish_to_nutrient', negative=False)

    fish_forcing = xso.forcing(foreign=True,
                               description='prescribed fish biomass')

    f_P = xso.parameter(dims='phyto',
                        description='fish predation rates on phytoplankton')
    f_Z = xso.parameter(dims='zoo',
                        description='fish predation rates on zooplankton')
    eps_F = xso.parameter(
        description='fish assimilation efficiency (this fraction leaves system)')

    @xso.flux(dims='phyto')
    def fish_graze_phyto(self, phyto, zoo, nutrient, fish_forcing,
                         f_P, f_Z, eps_F):
        return f_P * fish_forcing * phyto

    @xso.flux(dims='zoo')
    def fish_graze_zoo(self, phyto, zoo, nutrient, fish_forcing,
                       f_P, f_Z, eps_F):
        return f_Z * fish_forcing * zoo

    @xso.flux
    def fish_to_nutrient(self, phyto, zoo, nutrient, fish_forcing,
                         f_P, f_Z, eps_F):
        total_consumed = self.m.sum(f_P * phyto) + self.m.sum(f_Z * zoo)
        return (1 - eps_F) * fish_forcing * total_consumed


@xso.component
class FishGrazing_SizeBased:
    """Size-based fish predation. Grazed material leaves the system entirely."""
    phyto = xso.variable(dims='phyto', foreign=True,
                         flux='fish_graze_phyto', negative=True)
    zoo = xso.variable(dims='zoo', foreign=True,
                       flux='fish_graze_zoo', negative=True)

    fish_forcing = xso.forcing(foreign=True,
                               description='prescribed fish biomass')

    w_P = xso.parameter(dims='phyto', description='fish feeding weights on P')
    w_Z = xso.parameter(dims='zoo', description='fish feeding weights on Z')
    rate = xso.parameter(description='fish predation rate')

    @xso.flux(dims='phyto')
    def fish_graze_phyto(self, phyto, zoo, fish_forcing, w_P, w_Z, rate):
        return w_P * rate * fish_forcing * phyto

    @xso.flux(dims='zoo')
    def fish_graze_zoo(self, phyto, zoo, fish_forcing, w_P, w_Z, rate):
        return w_Z * rate * fish_forcing * zoo