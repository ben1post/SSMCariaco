"""
fig4_panels.py -- standalone candidate sub-panels for MS3 Figure 4.

Each `panel_*` draws into a supplied `ax` (so they compose into the final layout
later); styling follows Figures 1/3 (era colours, white-halo trio, in-ticks, log-µm
mcs axis, square panels). Model = the settled construct D:
    maranon_ward | GGE 0.31, mP 0.0015, m_Z 0.10, KsZ 0.23, sigma_log 0.20
    graded fish: pre+recovery r_F 0.4 / post r_F 0.0
Run THREAD-PINNED (chaotic model -- keep the 1-thread header active in the notebook).

In the notebook, set retina FIRST (it's an IPython magic, can't live here):
    %config InlineBackend.figure_format = 'retina'
    import fig4_panels as f4; f4.set_style()        # DPI / fonts (Fig 1/3 convention)

Two signature styles:
  - per-ERA panels (one era per ax, for side-by-side shared-y pairs): panel_cycle, panel_npzd
  - both-era panels (loop internally): panel_mcs_fn, panel_2x2, panel_duration, panel_dumbbell,
    panel_spectrum, panel_phase, panel_box, panel_violin
panel_box / panel_violin / panel_spectrum / panel_dumbbell are appendix candidates.
"""
import os
import pickle
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, NullLocator

import seasonal_scan_harness as ssh

# ---------------------------------------------------------------- settled construct
CONSTRUCT = 'maranon_ward'
PARAMS    = dict(GGE=0.31, mP=0.0015, m_Z=0.10, KsZ=0.23, sigma_log=0.20)
ERAS      = {'pre+recovery': 0.4, 'post': 0.0}          # era config -> graded fish rate
SK        = ssh.SEASONAL_SOLVER_KWARGS # {**ssh.SEASONAL_SOLVER_KWARGS, 'instability_neg_threshold': -1e-2}
DEFAULT_PKL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig4_modeloutput.pkl')

# ---------------------------------------------------------------- styling (Fig 1/3)
ERA_COL = {'pre+recovery': '#2E86AB', 'post': '#E07A5F', 'recovery': '#81B29A'}
ERA_LAB = {'pre+recovery': 'pre+recovery (fish on)', 'post': 'post (fish off)'}
NPZD_C  = {'N': '#6A8EAE', 'sumP': '#52A25E', 'Ztot': '#E0892F', 'D': '#9B6A4A'}
HALO     = [pe.withStroke(linewidth=2.5, foreground='white')]
HALO3    = [pe.withStroke(linewidth=3.0, foreground='white')]    # main model lines
LBL_HALO = [pe.withStroke(linewidth=2.0, foreground='white')]    # text
TINY_HALO = [pe.withStroke(linewidth=0.6, foreground='white')]   # markers (avoid hiding neighbours)
GEOM     = np.array([0.63, 6.3, 63.0])
UM_TICKS = [1, 2, 5, 10, 20, 40]
_MO      = np.arange(1, 13)
_MONTH_INI = list('JFMAMJJASOND')
_RNG     = np.random.default_rng(0)                     # reproducible jitter
MLAB = {'mcs': 'mean cell size (µm)', 'micro': 'Micro fraction', 'Z200': 'Zoo>200 (mmol N m$^{-3}$)',
        'Z500': 'Zoo>500 (mmol N m$^{-3}$)', 'sumP': '$\\Sigma$P (mmol N m$^{-3}$)',
        'N': 'N (mmol N m$^{-3}$)', 'PP': 'PP (mmol N m$^{-3}$ d$^{-1}$)',
        'Export': 'Export (mmol N m$^{-2}$ d$^{-1}$)'}


def set_style():
    """DPI / font defaults matching the other figures (call once in the notebook,
    after the `%config InlineBackend.figure_format = 'retina'` magic)."""
    plt.rcParams.update({'figure.dpi': 140, 'savefig.dpi': 200, 'font.size': 9,
                         'axes.titlesize': 9, 'axes.labelsize': 8, 'legend.frameon': False})


def _style(ax):
    ax.tick_params(top=False, right=True, direction='in', labelsize=7.5)


def _mcs_axis(ax):
    """Mean-cell-size on a log axis with µm ticks [1,2,5,10,20,40] -- matches Figs 1/3."""
    ax.set_yscale('log')
    ax.set_yticks(UM_TICKS); ax.set_yticklabels([str(u) for u in UM_TICKS])
    ax.yaxis.set_minor_locator(NullLocator())


# ---------------------------------------------------------------- data prep
def _run(forc_era, fish):
    return ssh.run_one(ssh.allometry(CONSTRUCT), forc_era, fish_rate=fish, years=60, spinup=15,
                       mP=PARAMS['mP'], m_Z=PARAMS['m_Z'],
                       grazing={'KsZ': PARAMS['KsZ'], 'sigma_log': PARAMS['sigma_log']},
                       iv_overrides={'GrazingRouter': {'gge': PARAMS['GGE']}},
                       solver_kwargs=SK, return_traj=True)


def prep(regimes=None, forcing_complete=True):
    """Run the 4 configs (era diagonal + 2x2 off-diagonals) + load obs. Returns a bundle
    dict: runs[(era, fish)] = (clim, r); obs_m (raw monthly, carries a 'regime' column); obs_t.
    `regimes` drops transition/NaN obs at build time (default None keeps all; plots mark them).
    `forcing_complete` (default True) restricts obs to F_N/d_e/T-complete months -- the fair set."""
    forc = ssh.build_forcings(['pre+recovery', 'post'])
    runs = {(era, fish): _run(forc[era], fish)
            for era in ('pre+recovery', 'post') for fish in (0.0, 0.4)}
    return dict(runs=runs,
                obs_m=ssh.build_obs_monthly(['pre+recovery', 'post'], regimes=regimes, forcing_complete=forcing_complete),
                obs_t=ssh.build_obs_targets(['pre+recovery', 'post'], regimes=regimes, forcing_complete=forcing_complete))


def build_output(path=DEFAULT_PKL, regimes=None, forcing_complete=True):
    """Run the 4 configs + obs (run this cell THREAD-PINNED), tag with metadata, and pickle
    to `path` (beside this script). Re-run when the construct / params / obs filters change.
    `regimes` drops transition/NaN obs; `forcing_complete` (default True) keeps only
    F_N/d_e/T-complete months (the fair comparison set). Both recorded in the metadata."""
    D = prep(regimes=regimes, forcing_complete=forcing_complete)
    D['meta'] = dict(construct=CONSTRUCT, params=dict(PARAMS), fish=dict(ERAS),
                     solver_floor=SK.get('instability_neg_threshold'),
                     regimes=regimes, forcing_complete=forcing_complete,
                     created=time.strftime('%Y-%m-%d %H:%M'))
    with open(path, 'wb') as fh:
        pickle.dump(D, fh)
    print(f"[fig4] model output saved -> {path}\n       {D['meta']}")
    return D


def load_output(path=DEFAULT_PKL):
    """Read the pickled bundle for plotting; print its metadata; warn if the stored
    construct/params differ from the module's current values (staleness guard)."""
    with open(path, 'rb') as fh:
        D = pickle.load(fh)
    meta = D.get('meta', {})
    print(f"[fig4] loaded {path}\n       {meta}")
    if meta.get('construct') != CONSTRUCT or meta.get('params') != PARAMS:
        print("  *** STALE WARNING: pickle construct/params != module defaults; "
              "re-run build_output() to refresh ***")
    return D


def _peryear_month(r, key, spinup=15):
    """Per-(year, calendar-month) means of `key` -> the interannual cloud, (n, 2) (month, value)."""
    keep = r['t'] >= spinup * 365.0
    t, v = r['t'][keep], np.asarray(r[key], float)[keep]
    mo = np.clip(np.searchsorted(ssh._MEDGES, np.mod(t, 365.0), 'right') + 1, 1, 12)
    yr = (t // 365.0).astype(int)
    out = []
    for y in np.unique(yr):
        for m in _MO:
            sel = v[(yr == y) & (mo == m)]; sel = sel[np.isfinite(sel)]
            if sel.size:
                out.append((m, sel.mean()))
    return np.array(out) if out else np.empty((0, 2))


def _monthly(r, key, spinup=15):
    keep = r['t'] >= spinup * 365.0
    mo = np.clip(np.searchsorted(ssh._MEDGES, np.mod(r['t'][keep], 365.0), 'right') + 1, 1, 12)
    v = np.asarray(r[key], float)[keep]
    return np.array([np.nanmean(v[mo == m]) if np.any(mo == m) else np.nan for m in _MO])


def _postspin(r, key, spinup=15):
    if key not in r:
        return np.array([])
    keep = r['t'] >= spinup * 365.0
    v = np.asarray(r[key], float)[keep]
    return v[np.isfinite(v)]


def _obs_classes(om):
    reg = om['regime'].astype('object')
    is_norm = reg.isin(['upwelling', 'relaxed'])
    is_tran = reg == 'transition'
    return om[is_norm], om[is_tran], om[~(is_norm | is_tran)]


def _plot_obs(ax, om, xcol, ycol, base, flag_mode='mark', s=18, z=3):
    """Obs scatter with regime marking. flag_mode: 'mark' (transition = open ring, no-regime =
    grey x), 'hide' (drop transition + no-regime), 'include' (all as plain obs points). Falls
    back to a plain scatter if the obs has no 'regime' column (old pickle -> rebuild to mark)."""
    if 'regime' not in om.columns or flag_mode == 'include':
        ax.scatter(om[xcol], om[ycol], s=s, color=base, alpha=0.5, lw=0, zorder=z, label='obs')
        return
    norm, tran, nanr = _obs_classes(om)
    ax.scatter(norm[xcol], norm[ycol], s=s, color=base, alpha=0.5, lw=0, zorder=z, label='obs')
    if flag_mode == 'mark':
        if len(tran):
            ax.scatter(tran[xcol], tran[ycol], s=s + 10, facecolors='none', edgecolors=base,
                       lw=1.1, zorder=z + 1, label='transition')
        if len(nanr):
            ax.scatter(nanr[xcol], nanr[ycol], s=s, marker='x', color='0.55', lw=0.9,
                       zorder=z + 1, label='no regime')


def _emptiest_corner(ax, P):
    """'loc' string of the axes corner with the fewest plotted points P (Nx2 data coords).
    Geometric midpoints on log axes. Use after the axis limits are final."""
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    xm = np.sqrt(x0 * x1) if ax.get_xscale() == 'log' else 0.5 * (x0 + x1)
    ym = np.sqrt(y0 * y1) if ax.get_yscale() == 'log' else 0.5 * (y0 + y1)
    P = np.asarray(P, float); P = P[np.isfinite(P).all(1)]
    lox, hix, loy, hiy = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
    P = P[(P[:, 0] >= lox) & (P[:, 0] <= hix) & (P[:, 1] >= loy) & (P[:, 1] <= hiy)]
    cnt = {'upper left': int(((P[:, 0] < xm) & (P[:, 1] > ym)).sum()),
           'upper right': int(((P[:, 0] > xm) & (P[:, 1] > ym)).sum()),
           'lower left': int(((P[:, 0] < xm) & (P[:, 1] < ym)).sum()),
           'lower right': int(((P[:, 0] > xm) & (P[:, 1] < ym)).sum())}
    return min(cnt, key=cnt.get)


def _label_months(ax, fn, mc, col):
    """Number every month; repel the labels (off each other and their dots) with adjustText,
    each carrying a thin leader line to its dot. Call AFTER the axis limits + box aspect are
    set -- adjustText works in display coords, so the geometry must be final."""
    texts = [ax.text(fn[mo - 1], mc[mo - 1], str(mo), fontsize=5.5, color=col, zorder=6,
                     ha='center', va='center', path_effects=TINY_HALO) for mo in _MO]
    try:
        from adjustText import adjust_text
        adjust_text(texts, force_points=0.2, force_text=0.2, expand_points=(1, 1),
                    expand_text=(1, 1), arrowprops=dict(arrowstyle='-', color=col, lw=0.5), ax=ax)
    except ImportError:
        print("fig4: `pip install adjustText` for the month-label repel")


# ============================================================ 1-3. seasonal cycle (ONE era / ax)
def panel_cycle(ax, D, era, metric='mcs', show_ylabel=True, show_legend=True, flag_mode='mark',
                legend_loc='auto'):
    """One era: obs monthly cloud (grey; transition/no-regime marked) + model interannual
    cloud (per-year monthly means, era colour) + model monthly-median line. Call once per era,
    shared y, for a pre|post pair. mcs on the log-µm axis (Fig 1/3); month axis = initials."""
    col = ERA_COL[era]
    clim, r = D['runs'][(era, ERAS[era])]
    om = D['obs_m'][era].dropna(subset=[metric])
    _plot_obs(ax, om, 'mo', metric, '0.30', flag_mode, s=24, z=3)
    ym = _peryear_month(r, metric)
    if len(ym):
        ax.scatter(ym[:, 0] + _RNG.uniform(-0.18, 0.18, len(ym)), ym[:, 1],
                   s=10, color=col, alpha=0.40, lw=0, zorder=2, label='model (interannual)')
        med = np.array([np.nanmedian(ym[ym[:, 0] == m, 1]) if np.any(ym[:, 0] == m) else np.nan for m in _MO])
        w = (med[0] + med[11]) / 2.0                                  # Dec<->Jan wrap value
        xw = np.r_[0.5, _MO, 12.5]; mw = np.r_[w, med, w]             # extend to both spines (annual cycle)
        ax.plot(xw, mw, '-', color=col, lw=2.2, zorder=4, path_effects=HALO3, label='model median')
    ax.set_xlim(0.5, 12.5); ax.set_xticks(_MO); ax.set_xticklabels(_MONTH_INI, fontsize=7)
    ax.set_xlabel('month')
    if metric == 'mcs':
        _mcs_axis(ax)
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_ylabel(MLAB.get(metric, metric) if show_ylabel else '', fontsize=8)
    ax.set_title(ERA_LAB[era], fontsize=9, loc='left', color=col)
    _style(ax)
    if show_legend:
        P = np.vstack([om[['mo', metric]].values, ym]) if len(ym) else om[['mo', metric]].values
        loc = _emptiest_corner(ax, P) if legend_loc == 'auto' else legend_loc
        ax.legend(fontsize=6.5, loc=loc, frameon=True, edgecolor='0.7', framealpha=0.9, borderpad=0.4)


# ============================================================ NPZD state time series (ONE era / ax)
def panel_npzd(ax, D, era, years=12, logy=False):
    """N / P(total) / Z(total) / D state-variable time series over the last `years` years."""
    clim, r = D['runs'][(era, ERAS[era])]
    t = r['t'] / 365.0
    m = t >= (t[-1] - years)
    for key, lab in [('N', 'N'), ('sumP', 'P'), ('Ztot', 'Z'), ('D', 'D')]:
        if key in r:
            ax.plot(t[m], np.asarray(r[key], float)[m], '-', color=NPZD_C[key], lw=1.0, label=lab)
    if logy:
        ax.set_yscale('log')
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_xlabel('year'); ax.set_ylabel('mmol N m$^{-3}$', fontsize=8)
    ax.set_title(f'NPZD — {ERA_LAB[era]}', fontsize=9, loc='left', color=ERA_COL[era])
    _style(ax); ax.legend(fontsize=7, ncol=4, loc='upper right')


# ============================================================ 4-5. bottom-up response
def panel_mcs_fn(ax, D, era, show_ylabel=True, show_legend=True, loop=True, annotate=True,
                 flag_mode='mark', legend_loc='auto', ymax=None):
    """ONE era's within-year (F_N, mcs) trajectory over that era's obs cloud. LINEAR mcs
    (the log flattened the comparison); model monthly points annotated with month number.
    Call once per era; independent y per panel (the cycle carries the shared-axis contrast).
    Obs transition / no-regime months marked per flag_mode."""
    col = ERA_COL[era]
    om = D['obs_m'][era].dropna(subset=['FN', 'mcs'])
    clim, r = D['runs'][(era, ERAS[era])]
    fn, mc = _monthly(r, 'FN'), _monthly(r, 'mcs')
    if loop:
        o = np.r_[np.arange(12), 0]
        ax.plot(fn[o], mc[o], '-', color=col, lw=1.3, zorder=3, path_effects=TINY_HALO)
    ax.plot(fn, mc, 'o', color=col, ms=4.5, zorder=4, path_effects=TINY_HALO)
    _plot_obs(ax, om, 'FN', 'mcs', col, flag_mode, s=18, z=5)   # obs ON TOP of model
    ax.set_xlabel('F$_N$ (mmol N m$^{-2}$ d$^{-1}$)')
    ax.set_ylabel('mean cell size (µm)' if show_ylabel else '', fontsize=8)
    ax.set_ylim(0, ymax) if ymax else ax.set_ylim(bottom=0)   # pass ymax (shared top) so limits are final
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5)); ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_title(ERA_LAB[era], fontsize=9, loc='left', color=col)
    ax.set_box_aspect(1); _style(ax)
    if annotate:
        _label_months(ax, fn, mc, col)                     # adjustText repel + leaders (after limits set)
    if show_legend:
        h, lbls = ax.get_legend_handles_labels()           # obs / transition / no-regime ...
        h.append(Line2D([], [], color=col, marker='o', ms=4, lw=1.3)); lbls.append('model')
        P = np.vstack([om[['FN', 'mcs']].values, np.column_stack([fn, mc])])
        loc = _emptiest_corner(ax, P) if legend_loc == 'auto' else legend_loc
        ax.legend(h, lbls, fontsize=6.5, loc=loc, frameon=True,
                  edgecolor='0.7', framealpha=0.9, borderpad=0.4, handletextpad=0.5)


# ============================================================ 6. the 2x2 with seasonal-range whiskers
def panel_2x2(ax, D, metric='mcs'):
    """{pre, post} forcing x {fish off (open), fish on (filled)}. Each cell: a marker at the
    median + a whisker spanning the seasonal range (monthly-clim min..max). Obs (star + range)
    only on the observed diagonal."""
    obs_fish = {'pre+recovery': 0.4, 'post': 0.0}
    for i, era in enumerate(('pre+recovery', 'post')):
        col = ERA_COL[era]
        for fish, dx, filled in [(0.0, -0.18, False), (0.4, 0.18, True)]:
            clim, r = D['runs'][(era, fish)]
            c = clim.get('clim_' + metric)
            md = clim.get(metric + '_med', np.nan)
            if c is None:
                continue
            ax.vlines(i + dx, np.nanmin(c), np.nanmax(c), color=col, lw=1.4, zorder=2)
            ax.plot(i + dx, md, 'o', color=col, mfc=(col if filled else 'white'), mew=1.5, ms=8, zorder=4)
        om = D['obs_m'][era].dropna(subset=[metric])
        if len(om):
            ocyc = om.groupby('mo')[metric].median().reindex(_MO).values
            ox = i + (0.18 if obs_fish[era] == 0.4 else -0.18)
            ax.vlines(ox, np.nanmin(ocyc), np.nanmax(ocyc), color='0.2', lw=2.5, alpha=0.45, zorder=3)
            ax.plot(ox, np.nanmedian(ocyc), '*', color='0.15', ms=13, zorder=5, path_effects=HALO3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['pre forcing', 'post forcing'], fontsize=8)
    ax.set_xlim(-0.5, 1.5); ax.set_ylabel(MLAB.get(metric, metric), fontsize=8)
    if metric == 'mcs':
        _mcs_axis(ax)
    ax.set_box_aspect(1); _style(ax)
    ax.legend(handles=[Line2D([], [], marker='o', color='0.4', ls='', label='fish on'),
                       Line2D([], [], marker='o', color='0.4', mfc='white', ls='', label='fish off'),
                       Line2D([], [], color='0.4', lw=1.4, label='seasonal range'),
                       Line2D([], [], marker='*', color='0.15', ls='', label='obs')],
              fontsize=6, loc='best')


# ============================================================ 9. bloom-duration bar (era contrast)
def panel_duration(ax, D, thresh=5.0):
    labels, mvals, ovals, cols = [], [], [], []
    for era in ERAS:
        col = ERA_COL[era]
        clim, r = D['runs'][(era, ERAS[era])]
        mc = np.asarray(clim.get('clim_mcs', np.full(12, np.nan)), float)
        ocyc = D['obs_m'][era].dropna(subset=['mcs']).groupby('mo')['mcs'].median().reindex(_MO).values
        mvals.append(int(np.nansum(mc > thresh))); ovals.append(int(np.nansum(ocyc > thresh)))
        labels.append(era.split('+')[0]); cols.append(col)
    x = np.arange(len(labels))
    ax.bar(x - 0.18, ovals, 0.34, color=cols, alpha=0.3, label='obs')
    ax.bar(x + 0.18, mvals, 0.34, color=cols, alpha=0.75, label='model')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(f'months mcs > {thresh:g} µm', fontsize=8); ax.set_ylim(0, 12)
    ax.set_box_aspect(1); _style(ax); ax.legend(fontsize=6.5)


# ============================================================ 12. mcs-Zoo>200 phase loop
def panel_phase(ax, D):
    for era in ERAS:
        col = ERA_COL[era]
        clim, r = D['runs'][(era, ERAS[era])]
        z, m = _monthly(r, 'Z200'), _monthly(r, 'mcs')
        o = np.r_[np.arange(12), 0]
        ax.plot(z[o], m[o], '-o', color=col, lw=1.4, ms=4, zorder=3, path_effects=HALO3, label=ERA_LAB[era])
    ax.set_xlabel('Zoo>200 (mmol N m$^{-3}$)'); ax.set_ylabel('mean cell size (µm)', fontsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4)); _mcs_axis(ax)
    ax.set_box_aspect(1); _style(ax); ax.legend(fontsize=6.5)


# ============================================================ 10. pre->post relative-change dumbbell (APPENDIX)
def panel_dumbbell(ax, D, metrics=('mcs', 'micro', 'sumP', 'PP', 'Export', 'N')):
    """Pre->post change as post/pre, model vs obs. Each source normalised to its OWN pre,
    so absolute model-obs offsets (e.g. net-tow Zoo) cancel and only the regime-shift
    direction/magnitude is compared. 1 = no change."""
    y = np.arange(len(metrics))[::-1]
    for j, mk in enumerate(metrics):
        opre = D['obs_t']['pre+recovery']['med'].get(mk, np.nan)
        opost = D['obs_t']['post']['med'].get(mk, np.nan)
        mpre = D['runs'][('pre+recovery', ERAS['pre+recovery'])][0].get(mk + '_med', np.nan)
        mpost = D['runs'][('post', ERAS['post'])][0].get(mk + '_med', np.nan)
        orat = opost / opre if opre else np.nan
        mrat = mpost / mpre if mpre else np.nan
        ax.plot([1, orat], [y[j], y[j]], '-', color='0.6', lw=4, alpha=0.35, zorder=1)
        ax.plot([1, mrat], [y[j] + 0.16, y[j] + 0.16], '-', color='0.35', lw=1.4, zorder=2)
        ax.scatter([orat], [y[j]], c='0.15', s=44, zorder=3, edgecolor='white', lw=0.6,
                   label='obs' if j == 0 else None)
        ax.scatter([mrat], [y[j] + 0.16], marker='s', c='#C0563F', s=28, zorder=4, edgecolor='white', lw=0.6,
                   label='model' if j == 0 else None)
    ax.axvline(1, color='0.7', ls=':', lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(metrics, fontsize=8)
    ax.set_xlabel('post / pre (relative change)', fontsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5)); _style(ax); ax.legend(fontsize=6.5, loc='best')


# ============================================================ 7-8/11. distributions + spectrum (APPENDIX)
def panel_box(ax, D, metric='Export'):
    data, cols, alphas, ticks = [], [], [], []
    for era in ERAS:
        col = ERA_COL[era]
        clim, r = D['runs'][(era, ERAS[era])]
        obs = D['obs_m'][era][metric].dropna().values if metric in D['obs_m'][era] else np.array([])
        mod = _postspin(r, metric)
        for v, who, a in ((obs, 'obs', 0.25), (mod, 'mod', 0.60)):
            if len(v):
                data.append(v); cols.append(col); alphas.append(a)
                ticks.append(f"{era.split('+')[0][:3]}\n{who}")
    if not data:
        ax.set_title(f"[no data] {metric}", fontsize=8); ax.set_box_aspect(1); return
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6, medianprops=dict(color='k', lw=1.0))
    for b, c, a in zip(bp['boxes'], cols, alphas):
        b.set_facecolor(c); b.set_alpha(a)
    ax.set_xticks(range(1, len(ticks) + 1)); ax.set_xticklabels(ticks, fontsize=6.5)
    ax.set_title(MLAB.get(metric, metric), fontsize=8); ax.set_ylim(bottom=0)
    ax.set_box_aspect(1); _style(ax)


def panel_violin(ax, D, metric='mcs'):
    p, ticks = 0, []
    for era in ERAS:
        col = ERA_COL[era]
        clim, r = D['runs'][(era, ERAS[era])]
        obs = D['obs_m'][era][metric].dropna().values if metric in D['obs_m'][era] else np.array([])
        mod = _postspin(r, metric)
        for v, who, a in ((obs, 'obs', 0.30), (mod, 'mod', 0.60)):
            if len(v) > 1:
                vp = ax.violinplot(v, positions=[p], widths=0.8, showmedians=True)
                for b in vp['bodies']:
                    b.set_facecolor(col); b.set_alpha(a); b.set_edgecolor('none')
                if 'cmedians' in vp:
                    vp['cmedians'].set_color('k')
            ticks.append(f"{era.split('+')[0][:3]}\n{who}"); p += 1
    ax.set_xticks(range(len(ticks))); ax.set_xticklabels(ticks, fontsize=6.5)
    ax.set_ylabel(MLAB.get(metric, metric), fontsize=8)
    if metric == 'mcs':
        _mcs_axis(ax)
    ax.set_box_aspect(1); _style(ax)


def panel_spectrum(ax, D):
    for era in ERAS:
        col = ERA_COL[era]
        clim, r = D['runs'][(era, ERAS[era])]
        msp = np.array([clim.get('pico', np.nan), clim.get('nano', np.nan),
                        clim.get('micro', np.nan)]) * clim.get('sumP', np.nan)
        o = D['obs_m'][era]
        osp = np.array([np.nanmedian(o['pico'] * o['sumP']), np.nanmedian(o['nano'] * o['sumP']),
                        np.nanmedian(o['micro'] * o['sumP'])])
        ax.plot(GEOM, osp, '--o', color=col, ms=4, alpha=0.5, zorder=2)
        ax.plot(GEOM, msp, '-s', color=col, ms=5, zorder=3, path_effects=HALO3, label=ERA_LAB[era])
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('cell size (µm ESD)'); ax.set_ylabel('biomass (mmol N m$^{-3}$)', fontsize=8)
    ax.set_box_aspect(1); _style(ax); ax.legend(fontsize=6.5)


# ============================================================ diagnostics (paste back)
def diag_fig4(D):
    """Numeric dump of exactly what the two Fig-4 panels plot, both eras -- the mcs seasonal
    cycle (model vs obs monthly median + obs coverage + peak/amplitude) and the F_N-mcs
    monthly trajectory + obs ranges. Run alongside the plots and paste the output back."""
    mo_hdr = "  ".join(f"{m:>5d}" for m in _MO)
    for era in ('pre+recovery', 'post'):
        clim, r = D['runs'][(era, ERAS[era])]
        om = D['obs_m'][era]
        oc = om.dropna(subset=['mcs'])
        ym = _peryear_month(r, 'mcs')
        mm = np.array([np.nanmedian(ym[ym[:, 0] == m, 1]) if np.any(ym[:, 0] == m) else np.nan for m in _MO])
        obm = oc.groupby('mo')['mcs'].median().reindex(_MO).values
        obn = oc.groupby('mo')['mcs'].count().reindex(_MO).fillna(0).astype(int).values
        print(f"\n=== {era} (fish={ERAS[era]}) ===")
        print(f"  mcs cycle   mo : {mo_hdr}")
        print(f"        model med: " + "  ".join(f"{v:5.1f}" if np.isfinite(v) else "   na" for v in mm))
        print(f"        obs   med: " + "  ".join(f"{v:5.1f}" if np.isfinite(v) else "   na" for v in obm))
        print(f"        obs   n  : " + "  ".join(f"{v:5d}" for v in obn))
        print(f"   peak: model mo={int(np.nanargmax(mm)) + 1} ({np.nanmax(mm):.1f}µm)  "
              f"obs mo={int(np.nanargmax(obm)) + 1} ({np.nanmax(obm):.1f})  "
              f"amp model={np.nanmax(mm) - np.nanmin(mm):.1f} obs={np.nanmax(obm) - np.nanmin(obm):.1f}")
        fn, mc = _monthly(r, 'FN'), _monthly(r, 'mcs')
        print(f"  F_N-mcs  model F_N: " + "  ".join(f"{v:5.2f}" for v in fn))
        print(f"           model mcs: " + "  ".join(f"{v:5.1f}" for v in mc))
        of = om.dropna(subset=['FN', 'mcs'])
        print(f"   obs  F_N [{of['FN'].min():.2f}, {of['FN'].max():.2f}]  "
              f"mcs [{of['mcs'].min():.1f}, {of['mcs'].max():.1f}]  n={len(of)}")
