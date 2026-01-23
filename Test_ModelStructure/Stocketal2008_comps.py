# import necessary packages
import numpy as np
import matplotlib.pyplot as plt
import xso

@xso.component
class Nutrient:
    """XSO component to define a state variable in the model."""
    value = xso.variable(description='nutrient concentration',
                         attrs={'units': 'mmol N m-3'})


@xso.component
class PhytoSizeSpectrum:
    """XSO component to define an array of state variables in the model.
    Additionally, there is a parameter defined, that stores an array of cell sizes."""

    biomass = xso.variable(dims='phyto', description='phytoplankton biomass',
                           attrs={'units': 'mmol N m-3', 'long_name': 'Phytoplankton biomass concentration',
                                  'standard_name': 'Phytoplankton'})
    phyto = xso.index(dims='phyto', description='size spectrum of phytoplankton',
                     attrs={'units': 'µm ESD', 'long_name': 'Phytoplankton size classes',
                                   'standard_name': 'P size classes'})


@xso.component
class ZooSizeSpectrum:
    """XSO component to define an array of state variables in the model.
    Additionally, there is a parameter defined, that stores an array of cell sizes."""

    biomass = xso.variable(dims='zoo', description='zooplankton biomass',
                           attrs={'units': 'mmol N m-3', 'long_name': 'Zooplankton biomass concentration',
                                  'standard_name': 'Zooplankton'})
    zoo = xso.index(dims='zoo', description='size spectrum of zooplankton',
                          attrs={'units': 'µm ESD', 'long_name': 'Zooplankton size classes',
                                 'standard_name': 'Z size classes'})


@xso.component
class ConstantExternalNutrient:
    """Component that provides a constant external nutrient
     as a forcing value.
    """

    forcing = xso.forcing(foreign=False, setup_func='forcing_setup', description='external nutrient')
    value = xso.parameter(description='constant value')

    def forcing_setup(self, value):
        """Method that returns forcing function providing the
        forcing value as a function of time."""
        @np.vectorize
        def forcing(time):
            return value

        return forcing


@xso.component
class LinearForcingInput:
    """Non-dimensional linear forcing input flux."""
    var = xso.variable(foreign=True, flux='input', negative=False, description='variable affected by flux')
    forcing = xso.forcing(foreign=True, description='forcing affecting flux')
    rate = xso.parameter(description='linear rate of change')

    @xso.flux
    def input(self, var, forcing, rate):
        """ """
        return forcing * rate



@xso.component
class MonodGrowth_SizeBased:
    """Component that calculates the growth flux of phytoplankton on a singular nutrient,
    based on Michealis-menten kinetics."""

    resource = xso.variable(foreign=True, flux='uptake', negative=True)
    consumer = xso.variable(foreign=True, dims='phyto', flux='uptake', negative=False)

    halfsat = xso.parameter(dims='phyto', description='half-saturation constants')
    mu_max = xso.parameter(dims='phyto', description='maximum growth rates')

    @xso.flux(dims='phyto')
    def uptake(self, resource, consumer, halfsat, mu_max):
        return mu_max * resource / (resource + halfsat) * consumer



### CUSTOMIZED COMPONENTS FOR STOCK ET AL 2008:

@xso.component
class StockPhytoMortality:
    """Linear Phytplankton Mortality Flux."""
    population = xso.variable(dims='phyto', foreign=True, flux='mortality', negative=True, description='variable affected by flux')
    nutrient = xso.variable(foreign=True, flux='recycled_mortality', negative=False, description='recycled mortality to nutrient')
    
    rate = xso.parameter(dims='phyto', description='linear rate of mortality')
    exponent = xso.parameter(dims='phyto', description='allows for conditional quadratic mortality')
    recycling = xso.parameter(dims='phyto', description='amount of mortality going to nutrient')
    
    @xso.flux(dims='phyto')
    def mortality(self, population, nutrient, rate, exponent, recycling):
        """Linear or quadratic decay function."""
        return (population ** exponent) * rate

    @xso.flux
    def recycled_mortality(self, population, nutrient, rate, exponent, recycling):
        """Linear or quadratic decay function."""
        return self.m.sum(((population ** exponent) * rate) * recycling)


@xso.component
class StockZooMortality:
    """Linear Zoplankton Mortality Flux."""
    population = xso.variable(dims='zoo', foreign=True, flux='mortality', negative=True, description='variable affected by flux')
    nutrient = xso.variable(foreign=True, flux='recycled_mortality', negative=False, description='recycled mortality to nutrient')
    
    rate = xso.parameter(dims='zoo', description='linear rate of mortality')
    exponent = xso.parameter(dims='zoo', description='allows for conditional quadratic mortality')
    recycling = xso.parameter(dims='zoo', description='amount of mortality going to nutrient')
    
    @xso.flux(dims='zoo')
    def mortality(self, population, nutrient, rate, exponent, recycling):
        """Linear or quadratic decay function."""
        return (population ** exponent) * rate

    @xso.flux
    def recycled_mortality(self, population, nutrient, rate, exponent, recycling):
        """Linear or quadratic decay function."""
        return self.m.sum(((population ** exponent) * rate) * recycling)



@xso.component
class StockZooMortality_simpleinput:
    """Linear Zoplankton Mortality Flux."""
    population = xso.variable(dims='zoo', foreign=True, flux='mortality', negative=True, description='variable affected by flux')
    nutrient = xso.variable(foreign=True, flux='recycled_mortality', negative=False, description='recycled mortality to nutrient')
    
    rate = xso.parameter(description='linear rate of mortality')
    exponent = xso.parameter(dims='zoo', description='allows for conditional quadratic mortality')
    recycling = xso.parameter(dims='zoo', description='amount of mortality going to nutrient')
    
    @xso.flux(dims='zoo')
    def mortality(self, population, nutrient, rate, exponent, recycling):
        """Linear or quadratic decay function."""
        rate_ext = np.array([0,0,0,rate[0]])
        return (population ** exponent) * rate_ext

    @xso.flux
    def recycled_mortality(self, population, nutrient, rate, exponent, recycling):
        """Linear or quadratic decay function."""
        return self.m.sum(((population ** exponent) * rate) * recycling)


        

@xso.component
class StockGrazingMatrix:
    """Size-based grazing function, adapted from Banas et al. (2011).

    The grazing function defines a complex pair-wise interaction between
    the size-spectra of phytoplankton and zooplankton. The grazing function
    scales with the size of the consumer and the feeding preference of the
    consumer for a given resource size. The grazing function is further
    scaled by the maximum ingestion rate of the consumer and the half-saturation
    constant of grazing.

    It is implemented in two parts. This component calculates the grazing matrix of each
    size class interaction. The second calculates receives the grazing matrix via the
    'group' argument to the flux function, and sums over the matrix to route the fluxes.
    """
    resource = xso.variable(foreign=True, dims='phyto')
    consumer = xso.variable(foreign=True, dims='zoo')
    phiPZ = xso.parameter(dims=('zoo', 'full'), description='feeding preferences')
    Imax = xso.parameter(dims='zoo', description='maximum ingestion rate')
    KsZ = xso.parameter(description='half saturation constant of grazing')

    @xso.flux(group='graze_matrix', dims=('full', 'zoo'))
    def grazing(self, resource, consumer, phiPZ, Imax, KsZ):
        """Here we are using a matrix calculation, to define the pair-wise interaction."""
        biomass = self.m.concatenate((resource, consumer))
        
        graz_pref = phiPZ.T * (np.transpose((biomass * phiPZ)**2)/self.m.sum((biomass * phiPZ)**2, axis=1))**0.5
 
        BMscaledAsFood = np.transpose(biomass * graz_pref.T)

        FgrazP = Imax * BMscaledAsFood / (KsZ + self.m.sum(BMscaledAsFood, axis=0))

        return FgrazP * consumer


@xso.component
class Stock_GGE_MatrixGrazing:
    """ Coponent to calculate the grazing fluxes for each of the model variables, adapted from Banas et al. (2011).

    The grazing fluxes are calculated by multiplying the grazing matrix with the
    biomass of the resource. The grazing matrix is calculated by the SizebasedGrazingKernel_Dims

    to N: beta*(1-epsilon)
    to D: 1-beta
    to Z: beta*epsilon
    """
    grazed_phyto = xso.variable(dims='phyto', foreign=True, flux='grazing_phyto', negative=True)
    grazed_zoo = xso.variable(dims='zoo', foreign=True, flux='grazing_zoo', negative=True)
    assimilated_consumer = xso.variable(dims='zoo', foreign=True, flux='assimilation')
    egested_detritus = xso.variable(foreign=True, flux='recycled')

    gge = xso.parameter(description='gross growth efficiency')
    R = xso.parameter(description='fraction excreted/respired')
    f_I = xso.parameter(dims='zoo', description='fraction ingested, that is recycled to N')
    alpha = xso.parameter(description='net production efficiency')

    @xso.flux(dims='phyto', group_to_arg='graze_matrix')
    def grazing_phyto(self, assimilated_consumer, egested_detritus, grazed_phyto, grazed_zoo, graze_matrix, gge, R, f_I, alpha):
        """ """
        out = self.m.sum(graze_matrix, axis=1)[0:len(grazed_phyto)]
        return out

    @xso.flux(dims='zoo', group_to_arg='graze_matrix')
    def grazing_zoo(self, assimilated_consumer, egested_detritus, grazed_phyto, grazed_zoo, graze_matrix, gge, R, f_I, alpha):
        """ """
        out = self.m.sum(graze_matrix, axis=1)[len(grazed_phyto):len(grazed_phyto)+len(grazed_zoo)]
        return out

    @xso.flux(dims='zoo', group_to_arg='graze_matrix')
    def assimilation(self, assimilated_consumer, egested_detritus,  grazed_phyto, grazed_zoo, graze_matrix, gge, R, f_I, alpha):
        """ """
        out = self.m.sum(graze_matrix, axis=0) * gge
        return out

    @xso.flux(group_to_arg='graze_matrix')
    def recycled(self, assimilated_consumer, egested_detritus,  grazed_phyto, grazed_zoo, graze_matrix, gge, R, f_I, alpha):
        """ """
        out = self.m.sum(f_I*(1-alpha) * self.m.sum(graze_matrix, axis=0)) + self.m.sum(graze_matrix) * R
        return out