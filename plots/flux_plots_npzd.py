"""flux_plots_npzd.py — per-class P and Z flux decomposition for the Cariaco NPZD model.

Adapted from flux_plots.py (which targets the older cariaco_ssm_comps model) to the
current N-P-Z-D baseline (cariaco_npzd_comps / cariaco_npzd_setups). Realized
steady-state decomposition only; the intrinsic-reference-biomass view from the
original is NOT ported here.

NPZD wiring differences handled vs the original:
  - GGE is `GrazingRouter__gge`            (was `GGE__gge`)
  - zoo closure is split into two terms:   `ZooQuadMortality__mortality_value`
    and `ZooLinMortality__mortality_value` (was a single `ZooMortality`)
  - grazing matrix `Grazing__grazing_value` is (prey = n_P+n_Z, predator = n_Z)
  - size grids are passed in (phyto_esd, zoo_esd), not read from the output

Requires a FULL-output run so that `Grazing__grazing_value` and the routing fluxes
are stored. The per-(prey,predator) matrix is the heart of the decomposition:
phyto rows 0..n_P-1, zoo rows n_P.. ; columns are predators (zoo).

Predator/prey size groups (small / optimal / large) are taken from the kernel's
own argmax, so they follow theta_opt / sigma automatically.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

# ---- colours -----------------------------------------------------------------
COLOR_UPTAKE = '#2c7fb8'
COLOR_MORT   = '#9e9e9e'        # phyto mortality / zoo quadratic closure
COLOR_MORT2  = '#cfcfcf'        # zoo linear closure
COLOR_FISH   = '#d73027'
COLOR_GRAZE_SMALL = '#fdd49e'; COLOR_GRAZE_OPT = '#f1a340'; COLOR_GRAZE_LARGE = '#7f3b08'
COLOR_SRC_P_SMALL = '#c7e9c0'; COLOR_SRC_P_OPT = '#41ab5d'; COLOR_SRC_P_LARGE = '#00441b'
COLOR_SRC_Z_SMALL = '#c6dbef'; COLOR_SRC_Z_OPT = '#4292c6'; COLOR_SRC_Z_LARGE = '#08306b'
COLOR_PICO = '#eef3fb'; COLOR_NANO = '#fbf6e8'; COLOR_MICRO = '#fbece8'

_EPS = 1e-30
def _dn(a):
    a = np.abs(np.asarray(a, float))
    return np.where(a < _EPS, 0.0, a)


# ---- size-group classification (kernel-argmax relative) ----------------------
def _classify_predator_groups(phiPZ_phyto):
    """Per prey row, classify predators 0/1/2 = smaller/optimal/larger than the
    kernel argmax predator. phiPZ_phyto: (n_prey, n_Z)."""
    n_prey, n_Z = phiPZ_phyto.shape
    j_opt = np.argmax(phiPZ_phyto, axis=1)
    g = np.full((n_prey, n_Z), -1, int)
    for i in range(n_prey):
        g[i, :j_opt[i]] = 0; g[i, j_opt[i]] = 1; g[i, j_opt[i] + 1:] = 2
    return g, j_opt


def _classify_prey_groups(phiPZ_full, prey_esd):
    """Per predator column, classify prey 0/1/2 = smaller/optimal/larger (in ESD)
    than that predator's kernel-argmax prey."""
    k_opt = np.argmax(phiPZ_full, axis=0)
    opt = prey_esd[k_opt]
    return np.sign(prey_esd[:, None] - opt[None, :]).astype(int) + 1, k_opt


# =============================================================================
# PHYTOPLANKTON
# =============================================================================
def extract_phyto_fluxes(out, phyto_esd, zoo_esd, time_index=-1):
    sel = dict(time=time_index)
    n_P = np.asarray(phyto_esd).size
    P  = out['Phytoplankton__biomass'].isel(**sel).values
    Z  = out['Zooplankton__biomass'].isel(**sel).values
    U  = out['Growth__uptake_value'].isel(**sel).values
    M  = out['PhytoMortality__mortality_value'].isel(**sel).values
    Ff = (out['FishGrazing__fish_graze_phyto_value'].isel(**sel).values
          if 'FishGrazing__fish_graze_phyto_value' in out.data_vars else np.zeros(n_P))
    G_full = out['Grazing__grazing_value'].isel(**sel).values          # (n_P+n_Z, n_Z)
    phiPZ  = out['Grazing__phiPZ'].values
    return dict(P_ss=_dn(P), Z_ss=_dn(Z), uptake=_dn(U), mortality=_dn(M), fish=_dn(Ff),
                grazing_per_predator=_dn(G_full[:n_P, :]),
                phyto_esd=np.asarray(phyto_esd), zoo_esd=np.asarray(zoo_esd),
                phiPZ_phyto=phiPZ[:n_P, :])


def _phyto_graze_groups(f):
    G = f['grazing_per_predator']
    groups, _ = _classify_predator_groups(f['phiPZ_phyto'])
    return (np.where(groups == 0, G, 0).sum(1),
            np.where(groups == 1, G, 0).sum(1),
            np.where(groups == 2, G, 0).sum(1))


def summarize_phyto_fluxes(out, phyto_esd, zoo_esd, time_index=-1, extinct=1e-7):
    f = extract_phyto_fluxes(out, phyto_esd, zoo_esd, time_index)
    P, e = f['P_ss'], f['phyto_esd']
    U, M, Ff = f['uptake'], f['mortality'], f['fish']
    Gs, Go, Gl = _phyto_graze_groups(f)
    tot = Gs + Go + Gl + M + Ff
    alive = np.isfinite(P) & (P > extinct)
    print('=' * 96); print('PHYTO per-class flux decomposition (realized SS)'); print('=' * 96)
    print(f"{'ESD':>8}{'P_ss':>11}{'uptake':>10} || {'Gr<opt':>9}{'Gr=opt':>9}{'Gr>opt':>9}"
          f"{'mort':>9}{'fish':>9} | {'Σloss':>10}{'net':>10}")
    for i in np.where(alive)[0]:
        print(f"{e[i]:8.2f}{P[i]:11.2e}{U[i]:10.2e} || {Gs[i]:9.2e}{Go[i]:9.2e}{Gl[i]:9.2e}"
              f"{M[i]:9.2e}{Ff[i]:9.2e} | {tot[i]:10.2e}{U[i]-tot[i]:+10.1e}")
    sl = (Gs + Go + Gl).sum() + M.sum() + Ff.sum()
    print(f"\nΣloss split — grazing {(Gs+Go+Gl).sum()/sl:.2f} / mortality {M.sum()/sl:.2f} / fish {Ff.sum()/sl:.2f}")
    return f


def plot_phyto_flux_decomposition(out, phyto_esd, zoo_esd, *, mode='biomass', dt=1.0,
                                  time_index=-1, extinct=1e-7, ax=None, title=None):
    f = extract_phyto_fluxes(out, phyto_esd, zoo_esd, time_index)
    P, e = f['P_ss'], f['phyto_esd']; n_P = P.size
    U, M, Ff = f['uptake'], f['mortality'], f['fish']
    Gs, Go, Gl = _phyto_graze_groups(f)
    alive = np.isfinite(P) & (P > extinct)
    if mode == 'per_capita':
        s = np.where(alive, 1.0 / np.where(P > 0, P, np.nan), 0.0)
        centre = np.zeros(n_P); log_y = False
        pos = [('uptake', U * s, COLOR_UPTAKE)]
        neg = [('Gr<opt', Gs*s, COLOR_GRAZE_SMALL), ('Gr=opt', Go*s, COLOR_GRAZE_OPT),
               ('Gr>opt', Gl*s, COLOR_GRAZE_LARGE), ('mort', M*s, COLOR_MORT), ('fish', Ff*s, COLOR_FISH)]
        ylab = 'per-capita rate [d⁻¹]'
    else:
        centre = P.copy(); log_y = True
        pos = [('uptake', U*dt, COLOR_UPTAKE)]
        neg = [('Gr<opt', Gs*dt, COLOR_GRAZE_SMALL), ('Gr=opt', Go*dt, COLOR_GRAZE_OPT),
               ('Gr>opt', Gl*dt, COLOR_GRAZE_LARGE), ('mort', M*dt, COLOR_MORT), ('fish', Ff*dt, COLOR_FISH)]
        ylab = f'biomass [mmol N m⁻³] (±flux×{dt:g}d)'
    ax = _draw(ax, e, P, centre, pos, neg, alive, log_y, ylab, (2.0, 20.0),
               title or f'PHYTO flux decomposition ({mode})', 'phyto ESD [µm]')
    return ax


# =============================================================================
# ZOOPLANKTON
# =============================================================================
def extract_zoo_fluxes(out, phyto_esd, zoo_esd, time_index=-1):
    sel = dict(time=time_index)
    n_P = np.asarray(phyto_esd).size
    P = out['Phytoplankton__biomass'].isel(**sel).values
    Z = out['Zooplankton__biomass'].isel(**sel).values
    gge = float(out['GrazingRouter__gge'].values)
    G_full = out['Grazing__grazing_value'].isel(**sel).values
    Mq = out['ZooQuadMortality__mortality_value'].isel(**sel).values
    Ml = out['ZooLinMortality__mortality_value'].isel(**sel).values
    Ff = (out['FishGrazing__fish_graze_zoo_value'].isel(**sel).values
          if 'FishGrazing__fish_graze_zoo_value' in out.data_vars else np.zeros_like(Z))
    return dict(P_ss=_dn(P), Z_ss=_dn(Z), gge=gge, G_full=_dn(G_full),
                closure_quad=_dn(Mq), closure_lin=_dn(Ml), fish=_dn(Ff),
                phiPZ_full=out['Grazing__phiPZ'].values, n_P=n_P,
                prey_esd=np.concatenate([phyto_esd, zoo_esd]),
                phyto_esd=np.asarray(phyto_esd), zoo_esd=np.asarray(zoo_esd))


def _decompose_zoo_fluxes(f):
    n_P = f['n_P']; n_Z = f['zoo_esd'].size; gge = f['gge']; G = f['G_full']
    is_phyto = np.arange(n_P + n_Z) < n_P
    prey_groups, _ = _classify_prey_groups(f['phiPZ_full'], f['prey_esd'])
    def _src(filt, gid):
        mask = (prey_groups == gid) & filt[:, None]
        return (gge * G * mask).sum(0)
    src = {'P_small': _src(is_phyto, 0), 'P_opt': _src(is_phyto, 1), 'P_large': _src(is_phyto, 2),
           'Z_small': _src(~is_phyto, 0), 'Z_opt': _src(~is_phyto, 1), 'Z_large': _src(~is_phyto, 2)}
    src['from_P'] = src['P_small'] + src['P_opt'] + src['P_large']
    src['from_Z'] = src['Z_small'] + src['Z_opt'] + src['Z_large']
    src['total'] = src['from_P'] + src['from_Z']
    G_on_Z = G[n_P:, :]
    pg, _ = _classify_predator_groups(f['phiPZ_full'][n_P:, :])
    grz = {'pred_small': np.where(pg == 0, G_on_Z, 0).sum(1),
           'pred_opt':   np.where(pg == 1, G_on_Z, 0).sum(1),
           'pred_large': np.where(pg == 2, G_on_Z, 0).sum(1)}
    grz['total_graze'] = grz['pred_small'] + grz['pred_opt'] + grz['pred_large']
    return src, grz


def summarize_zoo_fluxes(out, phyto_esd, zoo_esd, time_index=-1, extinct=1e-7):
    f = extract_zoo_fluxes(out, phyto_esd, zoo_esd, time_index)
    Z, e = f['Z_ss'], f['zoo_esd']
    src, grz = _decompose_zoo_fluxes(f)
    Mq, Ml, Ff = f['closure_quad'], f['closure_lin'], f['fish']
    loss = grz['total_graze'] + Mq + Ml + Ff
    alive = np.isfinite(Z) & (Z > extinct)
    print('=' * 110); print('ZOO per-class flux decomposition (realized SS)'); print('=' * 110)
    print(f"{'ESD':>8}{'Z_ss':>11}{'assim':>10}{'%fromZ':>8} || {'Pr<opt':>9}{'Pr=opt':>9}{'Pr>opt':>9}"
          f"{'quad':>9}{'lin':>9}{'fish':>8} | {'Σloss':>10}{'net':>10}")
    for j in np.where(alive)[0]:
        tot = src['total'][j]; fz = 100*src['from_Z'][j]/tot if tot > 0 else 0
        print(f"{e[j]:8.1f}{Z[j]:11.2e}{tot:10.2e}{fz:8.0f} || {grz['pred_small'][j]:9.2e}"
              f"{grz['pred_opt'][j]:9.2e}{grz['pred_large'][j]:9.2e}{Mq[j]:9.2e}{Ml[j]:9.2e}{Ff[j]:8.2e}"
              f" | {loss[j]:10.2e}{src['total'][j]-loss[j]:+10.1e}")
    return f


def plot_zoo_flux_decomposition(out, phyto_esd, zoo_esd, *, mode='biomass', dt=1.0,
                                time_index=-1, extinct=1e-7, ax=None, title=None):
    f = extract_zoo_fluxes(out, phyto_esd, zoo_esd, time_index)
    Z, e = f['Z_ss'], f['zoo_esd']; n_Z = Z.size
    src, grz = _decompose_zoo_fluxes(f)
    Mq, Ml, Ff = f['closure_quad'], f['closure_lin'], f['fish']
    alive = np.isfinite(Z) & (Z > extinct)
    sc = (np.where(alive, 1.0/np.where(Z > 0, Z, np.nan), 0.0) if mode == 'per_capita' else dt)
    centre = np.zeros(n_Z) if mode == 'per_capita' else Z.copy()
    log_y = (mode != 'per_capita')
    pos = [('P<opt', src['P_small']*sc, COLOR_SRC_P_SMALL), ('P=opt', src['P_opt']*sc, COLOR_SRC_P_OPT),
           ('P>opt', src['P_large']*sc, COLOR_SRC_P_LARGE), ('Z<opt', src['Z_small']*sc, COLOR_SRC_Z_SMALL),
           ('Z=opt', src['Z_opt']*sc, COLOR_SRC_Z_OPT), ('Z>opt', src['Z_large']*sc, COLOR_SRC_Z_LARGE)]
    neg = [('Pred<opt', grz['pred_small']*sc, COLOR_GRAZE_SMALL), ('Pred=opt', grz['pred_opt']*sc, COLOR_GRAZE_OPT),
           ('Pred>opt', grz['pred_large']*sc, COLOR_GRAZE_LARGE), ('quad clos.', Mq*sc, COLOR_MORT),
           ('lin clos.', Ml*sc, COLOR_MORT2), ('fish', Ff*sc, COLOR_FISH)]
    ylab = 'per-capita rate [d⁻¹]' if mode == 'per_capita' else f'biomass [mmol N m⁻³] (±flux×{dt:g}d)'
    ax = _draw(ax, e, Z, centre, pos, neg, alive, log_y, ylab, (200.0,),
               title or f'ZOO flux decomposition ({mode})  (greens/blues up = assimilation from P / from Z)',
               'zoo ESD [µm]', ncol=3)
    return ax


# ---- shared drawing ----------------------------------------------------------
def _draw(ax, esd, bio, centre, pos, neg, alive, log_y, ylab, vlines, title, xlab, ncol=2):
    if ax is None:
        _, ax = plt.subplots(figsize=(12.5, 6.0))
    n = len(esd); x = np.arange(n); ai = np.where(alive)[0]
    if ai.size:
        xa, ca = ai.astype(float), centre[ai]
        bot = ca.copy()
        for lab, v, c in pos:
            ax.bar(xa, v[ai], bottom=bot, width=0.7, color=c, edgecolor='k', linewidth=0.3, label=lab, zorder=3)
            bot = bot + v[ai]
        top = ca.copy(); floor = 1e-3*bio[ai] if log_y else None
        for lab, v, c in neg:
            va = v[ai]
            vc = np.minimum(va, np.maximum(top - floor, 0.0)) if log_y else va
            ax.bar(xa, -vc, bottom=top, width=0.7, color=c, edgecolor='k', linewidth=0.3, label=lab, zorder=3)
            top = top - vc
        for xi, c in zip(xa, ca):
            ax.hlines(c, xi-0.35, xi+0.35, colors='k', linewidth=1.2, zorder=5)
    if not log_y:
        net = centre.copy()  # zero baseline; net marker = sum(pos)-sum(neg)
        net = sum(v for _, v, _ in pos) - sum(v for _, v, _ in neg)
        ax.plot(x, np.where(alive, net, np.nan), 'k_', ms=11, mew=2, zorder=6, label='net')
        ax.axhline(0, color='k', lw=0.8)
    for b in vlines:
        ax.axvline(np.interp(np.log10(b), np.log10(esd), x), c='grey', ls=':')
    if log_y and ai.size:
        sp = sum(v[ai] for _, v, _ in pos); sn = sum(v[ai] for _, v, _ in neg)
        ax.set_ylim(max(np.maximum(bio[ai]-sn, 1e-3*bio[ai]).min()*0.5, 1e-30), (bio[ai]+sp).max()*2)
        ax.set_yscale('log')
    ax.set_xticks(x[::3]); ax.set_xticklabels([f'{e:.2g}' for e in esd[::3]], rotation=45, ha='right')
    ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_xlim(-0.5, n-0.5); ax.set_title(title, fontsize=10)
    ax.legend(loc='upper right', fontsize=7, ncol=ncol, framealpha=0.95)
    ax.grid(axis='y', ls=':', alpha=0.4, zorder=1)
    return ax


# =============================================================================
# INTRINSIC FLUX VIEW — per-class rates at a uniform REFERENCE biomass B0
# =============================================================================
# Unlike the realized-SS decomposition (where net ≈ 0 for every coexisting class),
# this evaluates each class's per-capita rates on a uniform biomass field B0 at a
# chosen nutrient N_ref, so the NET (invasion) rate is informative: it shows which
# classes would grow / shrink from that reference. Pass the construct's allometry
# arrays explicitly so it works for any growth / grazing / mortality choice.

def _intrinsic_Gpc(phyto_esd, zoo_esd, Imax, KsZ, phiPZ, B0, fT_graze):
    """Per-capita grazing matrix at uniform biomass B0: Gpc[k,j] = per-capita
    grazing of predator j on prey k (also the prey-k contribution to predator j's
    per-capita ingestion, since B_k = Z_j = B0). Returns (Gpc (nP+nZ, nZ), S)."""
    nP, nZ = phyto_esd.size, zoo_esd.size
    B = np.full(nP + nZ, B0)
    S = (phiPZ * B[:, None]).sum(0)
    Gpc = fT_graze * Imax[None, :] * B0 * phiPZ * (S / (S**2 + KsZ**2))[None, :]
    return Gpc, S


def plot_intrinsic_phyto(phyto_esd, zoo_esd, mu_max, ks, mP, Imax, KsZ, phiPZ, *,
                         N_ref, B0=1.0, fT_grow=1.0, fT_graze=1.0, ax=None, title=None):
    nP = phyto_esd.size
    growth = fT_grow * mu_max * N_ref / (N_ref + ks)
    mort = np.full(nP, mP) if np.ndim(mP) == 0 else np.asarray(mP, float)
    Gpc, _ = _intrinsic_Gpc(phyto_esd, zoo_esd, Imax, KsZ, phiPZ, B0, fT_graze)
    Gphy = Gpc[:nP, :]
    grp, _ = _classify_predator_groups(phiPZ[:nP, :])
    gs = np.where(grp == 0, Gphy, 0).sum(1); go = np.where(grp == 1, Gphy, 0).sum(1); gl = np.where(grp == 2, Gphy, 0).sum(1)
    net = growth - mort - (gs + go + gl)
    if ax is None: _, ax = plt.subplots(figsize=(12.5, 5.2))
    x = np.arange(nP)
    ax.bar(x, growth, color=COLOR_UPTAKE, label='growth')
    ax.bar(x, -mort, color=COLOR_MORT, label='mortality')
    ax.bar(x, -gs, bottom=-mort, color=COLOR_GRAZE_SMALL, label='graze by <opt pred')
    ax.bar(x, -go, bottom=-(mort + gs), color=COLOR_GRAZE_OPT, label='graze by =opt pred')
    ax.bar(x, -gl, bottom=-(mort + gs + go), color=COLOR_GRAZE_LARGE, label='graze by >opt pred')
    ax.plot(x, net, 'k_', ms=11, mew=2, label='net (invasion rate)')
    ax.axhline(0, c='k', lw=.8)
    for b in (2, 20): ax.axvline(np.interp(np.log10(b), np.log10(phyto_esd), x), c='grey', ls=':')
    ax.set_xticks(x[::3]); ax.set_xticklabels([f'{e:.2g}' for e in phyto_esd[::3]], rotation=45, ha='right')
    ax.set_xlabel('phyto ESD [µm]'); ax.set_ylabel('per-capita rate [d⁻¹]')
    ax.set_title(title or f'PHYTO intrinsic rates  (N_ref={N_ref:.4g}, B0={B0:g})'); ax.legend(fontsize=8, ncol=2)
    return ax


def plot_intrinsic_zoo(phyto_esd, zoo_esd, Imax, KsZ, phiPZ, gge, mZ, mZlin, *,
                       B0=1.0, fT_graze=1.0, ax=None, title=None):
    nP, nZ = phyto_esd.size, zoo_esd.size
    Gpc, _ = _intrinsic_Gpc(phyto_esd, zoo_esd, Imax, KsZ, phiPZ, B0, fT_graze)
    assim_P = gge * Gpc[:nP, :].sum(0)
    assim_Z = gge * Gpc[nP:, :].sum(0)
    Gon = Gpc[nP:, :]
    grp, _ = _classify_predator_groups(phiPZ[nP:, :])
    gs = np.where(grp == 0, Gon, 0).sum(1); go = np.where(grp == 1, Gon, 0).sum(1); gl = np.where(grp == 2, Gon, 0).sum(1)
    quad = np.full(nZ, mZ * nZ * B0); lin = np.full(nZ, mZlin)
    net = (assim_P + assim_Z) - (gs + go + gl) - quad - lin
    if ax is None: _, ax = plt.subplots(figsize=(12.5, 5.2))
    x = np.arange(nZ)
    ax.bar(x, assim_P, color=COLOR_SRC_P_OPT, label='assimilation from P')
    ax.bar(x, assim_Z, bottom=assim_P, color=COLOR_SRC_Z_OPT, label='assimilation from Z (intraguild)')
    ax.bar(x, -gs, color=COLOR_GRAZE_SMALL, label='grazed by <opt pred')
    ax.bar(x, -go, bottom=-gs, color=COLOR_GRAZE_OPT, label='grazed by =opt pred')
    ax.bar(x, -gl, bottom=-(gs + go), color=COLOR_GRAZE_LARGE, label='grazed by >opt pred')
    ax.bar(x, -quad, bottom=-(gs + go + gl), color=COLOR_MORT, label='quad closure')
    ax.bar(x, -lin, bottom=-(gs + go + gl + quad), color=COLOR_MORT2, label='lin closure')
    ax.plot(x, net, 'k_', ms=11, mew=2, label='net (invasion rate)')
    ax.axhline(0, c='k', lw=.8)
    for b in (20, 200): ax.axvline(np.interp(np.log10(b), np.log10(zoo_esd), x), c='grey', ls=':')
    ax.set_xticks(x[::3]); ax.set_xticklabels([f'{e:.0f}' for e in zoo_esd[::3]], rotation=45, ha='right')
    ax.set_xlabel('zoo ESD [µm]'); ax.set_ylabel('per-capita rate [d⁻¹]')
    ax.set_title(title or f'ZOO intrinsic rates  (B0={B0:g})'); ax.legend(fontsize=7, ncol=2)
    return ax
