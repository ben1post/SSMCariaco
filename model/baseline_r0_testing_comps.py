"""
baseline_r0_testing_comps.py — clean-room R0 ladder components
==============================================================
A fresh, self-contained component set for the step-by-step R0 stability
diagnosis (2026-06-08). Built deliberately, with literature citations on each
equation, and NO inherited "_recycled" baggage. The defining design choice is
that every biomass loss has an explicit, tunable LOSS FATE via a
`recycle_fraction` scalar: the fraction returned to N vs exported out of the
system. recycle_fraction=0 = full export (Banas 2011 open-system convention,
Survey §12); recycle_fraction=1 = full recycle (closed Taniguchi/Cloern style).

Rung 1 (minimal omnivory baseline): N-P-Z size-structured, distributed
Holling Type III grazing (omnivory), quadratic zoo closure, temperature (Cloern
Q10), Stock F_N/d_e supply + phyto sinking, allometric rates. recycle_fraction=0.

Equation sources (Survey = `model context/Size Spectral Setup Survey.md`):
  - Monod uptake:        Taniguchi 2014 Eq.3 / Cloern 2018 Eq.1 / Banas 2011
  - Type III grazing:    Dutkiewicz 2020 Eq.S1.6 / Mattern 2026 Eq.3 (Survey §7)
  - palatability kernel: Banas 2011 / Mattern 2026 Eq.5 / Dutkiewicz 2020 (§9)
  - quadratic closure:   Banas 2011 ζ·Z_j·Z_tot (Survey §11)
  - F_N/d_e supply:      Stock 2008 Eq.7 (Survey §12)
  - phyto sinking:       Stock 2008 / Banas 2011 (Survey §12)
  - temperature Q10:     Cloern 2018 (growth 1.62, grazing 2.48)
Loss-fate (recycle vs export): open schemes export (Banas exports 100%, folds
regeneration into the supply S); closed schemes recycle (Survey §12, line ~291).
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
    """Zooplankton biomass across n_zoo log-spaced size classes (zoo=r·phyto)."""
    biomass = xso.variable(dims='zoo', description='zooplankton biomass',
                           attrs={'units': 'mmol N m-3'})
    zoo_esd = xso.index(dims='zoo', as_parameter=True,
                        description='zoo size classes (ESD)',
                        attrs={'units': 'µm'})


# =============================================================================
# SUPPLY + FORCING
# =============================================================================

@xso.component
class StockNutrientSupply:
    """New-nutrient supply over the euphotic box (Stock 2008 Eq.7, Survey §12).

        dN/dt += F_N / d_e

    d_e is a broadcast parameter shared with PhytoSinking so one regime depth
    drives both supply (F_N/d_e) and sinking (w_sink/d_e).
    """
    var = xso.variable(foreign=True, flux='input', negative=False,
                       description='nutrient receiving the supply')
    FN = xso.parameter(description='new-nutrient flux F_N [mmol N m-2 d-1]')
    de = xso.parameter(broadcast=True,
                       description='euphotic box depth d_e [m] (broadcast)')

    @xso.flux
    def input(self, var, FN, de):
        return FN / de


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


# =============================================================================
# PHYTO GROWTH — Monod uptake with Q10 (Taniguchi Eq.3 / Cloern Eq.1)
# =============================================================================

@xso.component
class MonodGrowth_T:
    """Per-class Monod uptake with Cloern (2018) Q10 temperature scaling.

        U_i = q10^((T-T_ref)/10) · μ_max,i · N · P_i / (N + k_s,i)

    Per-class μ_max and k_s arrays carry the size dependence (set in setup;
    Banas 2011 Table 2: μ_max=2.6·s^-0.45, k_s=0.1·s^1.0). Metabolic losses
    (respiration, exudation) are already folded into μ_max (Taniguchi p.16), so
    U is NET growth before grazing and background loss.
    """
    resource = xso.variable(foreign=True, flux='uptake', negative=True,
                            description='dissolved nitrogen (scalar sink)')
    consumer = xso.variable(foreign=True, dims='phyto', flux='uptake',
                            negative=False, description='phyto (per-class source)')
    temperature = xso.forcing(foreign=True, description='box temperature [°C]')
    mu_max = xso.parameter(dims='phyto', description='max growth rate per class [d-1]')
    halfsat = xso.parameter(dims='phyto', description='nutrient half-sat per class [mmol N m-3]')
    q10 = xso.parameter(description='growth Q10 (Cloern 2018: 1.62)')
    t_ref = xso.parameter(description='reference temperature [°C] (20)')

    @xso.flux(dims='phyto')
    def uptake(self, resource, consumer, temperature, mu_max, halfsat, q10, t_ref):
        f_T = q10 ** ((temperature - t_ref) / 10.0)
        return f_T * mu_max * resource * consumer / (resource + halfsat)


# =============================================================================
# GRAZING — distributed Holling Type III (Dutkiewicz 2020 Eq.S1.6 / Mattern Eq.3)
# =============================================================================

def compute_grazing_kernel(phyto_esd, zoo_esd, mode='omni',
                           theta_opt=10.0, sigma_log=0.25, convention='2sigma2'):
    """Feeding-preference (palatability) matrix phiPZ of shape (n_P+n_Z, n_Z).

    Gaussian on the log10 predator:prey size ratio, peak at theta_opt (1:10;
    Banas 2011 / Mattern 2026 Eq.5 / Dutkiewicz 2020, Survey §9).

    mode : 'herb'  — kernel on phyto prey only (zoo block zero).
           'omni'  — kernel on phyto + zoo prey, zoo-on-self diagonal zeroed
                     (no within-class cannibalism). Literature standard
                     (Mattern/Dutkiewicz/Ward all omnivorous; Survey §7).
    convention : '2sigma2'  exp(-(Δ)²/(2σ²)); σ = Gaussian std (MS3 historical).
                 'mattern'  exp(-((Δ)/σ)²);  Mattern Eq.5, σ=0.15 ≡ std 0.106.
    """
    phyto_esd = np.asarray(phyto_esd); zoo_esd = np.asarray(zoo_esd)
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
    if mode == 'herb':
        phiPZ[:n_P, :] = kernel[:n_P, :]
    elif mode == 'omni':
        phiPZ[:] = kernel
        for j in range(n_Z):
            phiPZ[n_P + j, j] = 0.0       # no self-cannibalism within a class
    else:
        raise ValueError(f"mode must be 'herb'/'omni', got {mode!r}")
    return phiPZ


@xso.component
class DistributedGrazing_TypeIII_T:
    """Distributed Holling Type III grazing with Q10 (Dutkiewicz 2020 Eq.S1.6).

        S_j   = Σ_k φ_kj · B_k                              (B = [P; Z])
        G_kj  = q10^((T-T_ref)/10) · I_max,j · Z_j · φ_kj · B_k · S_j
                / (S_j² + K_sZ,j²)

    Type III (squared-prey saturation → low-prey refuge, the stabiliser per
    Rohr 2022). Publishes the (n_P+n_Z, n_Z) matrix to the 'graze_matrix' group;
    DistributedGrazingRouter_split routes it. φ = phiPZ from the setup (kernel
    mode). K_sZ uniform (Hansen 1997: size-independent half-sat; Type III
    requires uniform K_sZ). Grazing Q10=2.48 (Cloern 2018).
    """
    resource = xso.variable(foreign=True, dims='phyto', description='phyto prey')
    consumer = xso.variable(foreign=True, dims='zoo', description='zoo predator')
    temperature = xso.forcing(foreign=True, description='box temperature [°C]')
    phiPZ = xso.parameter(dims=('full', 'zoo'),
                          description='palatability matrix (prey × predator)')
    Imax = xso.parameter(dims='zoo', description='per-class max ingestion [d-1]')
    KsZ = xso.parameter(dims='zoo', description='per-class grazing half-sat [mmol N m-3]')
    q10 = xso.parameter(description='grazing Q10 (Cloern 2018: 2.48)')
    t_ref = xso.parameter(description='reference temperature [°C] (20)')

    @xso.flux(group='graze_matrix', dims=('full', 'zoo'))
    def grazing(self, resource, consumer, temperature, phiPZ, Imax, KsZ, q10, t_ref):
        f_T = q10 ** ((temperature - t_ref) / 10.0)
        biomass = self.m.concatenate((resource, consumer))      # (full,)
        S_prey = self.m.sum(phiPZ * biomass[:, None], axis=0)   # (zoo,)
        return (f_T * Imax * consumer * S_prey / (S_prey ** 2 + KsZ ** 2)
                * phiPZ * biomass[:, None])


@xso.component
class DistributedGrazingRouter_split:
    """Route the 'graze_matrix' group into per-prey loss, per-predator gain
    (×Γ assimilation), and the (1−Γ) unassimilated fraction split between
    return-to-N and EXPORT via `recycle_fraction`.

        prey loss (P):  Σ_pred G_kj                       (per phyto class)
        prey loss (Z):  Σ_pred G_kj                       (per zoo class; omnivory)
        predator gain:  Γ · Σ_prey G_kj                   (per zoo predator)
        to N:           recycle_fraction · (1−Γ) · ΣG     (scalar)
        exported:       (1−recycle_fraction) · (1−Γ) · ΣG (leaves system; no flux)

    Γ = gross growth efficiency (Stock 2008 = 0.25). The unassimilated (1−Γ)
    fraction = sloppy feeding + egestion; in an open system its literature fate
    is partly/fully export (Banas 2011), here tunable via recycle_fraction.
    """
    grazed_phyto = xso.variable(foreign=True, dims='phyto',
                                flux='loss_P', negative=True,
                                description='phyto (per-class grazing sink)')
    grazed_zoo = xso.variable(foreign=True, dims='zoo',
                              flux='loss_Z', negative=True,
                              description='zoo-as-prey (per-class sink; omnivory)')
    assimilated_consumer = xso.variable(foreign=True, dims='zoo',
                                        flux='gain_Z',
                                        description='zoo (per-class source, ×Γ)')
    recycled_nutrient = xso.variable(foreign=True, flux='recycle_to_N',
                                     description='N (scalar source; recycled (1−Γ) fraction)')
    gamma = xso.parameter(description='Γ — gross growth efficiency')
    recycle_fraction = xso.parameter(
        description='fraction of (1−Γ) unassimilated grazing returned to N '
                    '(0 = full export, Banas open; 1 = full recycle, closed)')

    @xso.flux(dims='phyto', group_to_arg='graze_matrix')
    def loss_P(self, grazed_phyto, graze_matrix, gamma, recycle_fraction):
        return self.m.sum(graze_matrix, axis=1)[0:len(grazed_phyto)]

    @xso.flux(dims='zoo', group_to_arg='graze_matrix')
    def loss_Z(self, grazed_phyto, grazed_zoo, graze_matrix, gamma, recycle_fraction):
        n_P = len(grazed_phyto)
        return self.m.sum(graze_matrix, axis=1)[n_P:n_P + len(grazed_zoo)]

    @xso.flux(dims='zoo', group_to_arg='graze_matrix')
    def gain_Z(self, graze_matrix, gamma, recycle_fraction):
        return self.m.sum(graze_matrix, axis=0) * gamma

    @xso.flux(group_to_arg='graze_matrix')
    def recycle_to_N(self, graze_matrix, gamma, recycle_fraction):
        return recycle_fraction * (1.0 - gamma) * self.m.sum(graze_matrix)


# =============================================================================
# LOSS TERMS — explicit recycle/export split
# =============================================================================

@xso.component
class PhytoLinearLoss_split:
    """Per-class phyto background loss (viral, lysis, senescence), split
    recycle/export.

        per-class P sink:  m_P,i · P_i               (m_P = 0.1·μ_max, Banas 2011)
        to N (scalar):     recycle_fraction · Σ_i m_P,i · P_i
        exported:          (1−recycle_fraction) · Σ_i m_P,i · P_i  (leaves system)
    """
    population = xso.variable(foreign=True, dims='phyto',
                              flux='mortality', negative=True,
                              description='phyto (per-class sink)')
    recycled_nutrient = xso.variable(foreign=True, flux='recycle_to_N',
                                     description='N (scalar source)')
    rate = xso.parameter(dims='phyto', description='m_P per-class loss rate [d-1]')
    recycle_fraction = xso.parameter(
        description='fraction returned to N (0 = export, 1 = recycle)')

    @xso.flux(dims='phyto')
    def mortality(self, population, rate, recycle_fraction):
        return rate * population

    @xso.flux
    def recycle_to_N(self, population, rate, recycle_fraction):
        return recycle_fraction * self.m.sum(rate * population)


@xso.component
class ZooQuadraticLoss_split:
    """Distributed quadratic zoo closure (Banas 2011 ζ·Z_j·Z_tot, Survey §11),
    split recycle/export. Represents unresolved higher-trophic mortality.

        per-class Z sink:  m_Z · Z_j · Σ_k Z_k
        to N (scalar):     recycle_fraction · m_Z · (Σ_k Z_k)²
        exported:          (1−recycle_fraction) · m_Z · (Σ_k Z_k)²  (leaves system)

    Banas runs this as a 100%-EXPORT term in his open system (Survey §12); the
    Taniguchi/Cloern closed systems recycle it 100%. recycle_fraction spans both.
    m_Z is a quadratic coefficient [(mmol N m-3)^-1 d-1] (scalar, not allometric).
    """
    population = xso.variable(foreign=True, dims='zoo',
                              flux='mortality', negative=True,
                              description='zoo (per-class sink)')
    recycled_nutrient = xso.variable(foreign=True, flux='recycle_to_N',
                                     description='N (scalar source)')
    rate = xso.parameter(description='m_Z quadratic closure coeff [(mmol N m-3)^-1 d-1]')
    recycle_fraction = xso.parameter(
        description='fraction returned to N (0 = export, 1 = recycle)')

    @xso.flux(dims='zoo')
    def mortality(self, population, rate, recycle_fraction):
        return rate * population * self.m.sum(population)

    @xso.flux
    def recycle_to_N(self, population, rate, recycle_fraction):
        total_Z = self.m.sum(population)
        return recycle_fraction * rate * total_Z * total_Z


# =============================================================================
# PHYTO SINKING — one-way export (Stock 2008 / Banas 2011, Survey §12)
# =============================================================================

@xso.component
class PhytoSinking_export:
    """Per-class phyto sinking out of the euphotic box (always exports).

        per-class P sink:  (w_sink / d_e) · P_i

    The principal true sink in Stock-style open models. d_e foreign-referenced
    from the supply component (broadcast).
    """
    population = xso.variable(foreign=True, dims='phyto',
                              flux='sinking', negative=True,
                              description='phyto (per-class sink)')
    w_sink = xso.parameter(description='sinking velocity w_sink [m d-1]')
    de = xso.parameter(foreign=True, description='euphotic box depth d_e [m]')

    @xso.flux(dims='phyto')
    def sinking(self, population, w_sink, de):
        return (w_sink / de) * population
