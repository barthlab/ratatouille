import numpy as np
from scipy.stats import ttest_ind, ttest_rel, mannwhitneyu, wilcoxon, ks_2samp
from scipy.stats import linregress, pearsonr
from scipy.stats import f_oneway, tukey_hsd
from sklearn.linear_model import HuberRegressor


SIGNIFICANT_ALPHA = 0.05

def get_stars(p):
    return r"$\ast$"*3 if p < 0.001 else r"$\ast$"*2 if p < 0.01 else r"$\ast$" if p < 0.05 else 'n.s.'


def get_annotation_str(p_val: float, verbose: bool = True, double_line: bool = False):
    asterisks = get_stars(p_val)
    
    str_between = "\n" if double_line else " "

    if verbose:
        if p_val < 0.001:
            exponent = int(np.floor(np.log10(abs(p_val))))
            mantissa = p_val / 10 ** exponent
            return f"{asterisks}{str_between}p = {mantissa:.1f} × 10$^{{{exponent}}}$"
        else:
            return f"{asterisks}{str_between}p = {p_val:.3f}"
    return asterisks


def stats_ttest_ind(
        data1: np.ndarray | list,
        data2: np.ndarray | list,
):    
    p_val = ttest_ind(data1, data2).pvalue # type: ignore fucking stupid pylance
    return p_val, get_annotation_str(p_val)


def stats_ttest_rel(
        data1: np.ndarray | list,
        data2: np.ndarray | list,
):
    assert len(data1) == len(data2), "Cannot compare arrays of different lengths for paired t-test"
    p_val = ttest_rel(data1, data2).pvalue # type: ignore fucking stupid pylance
    return p_val, get_annotation_str(p_val)


def stats_mannwhitneyu(
        data1: np.ndarray | list,
        data2: np.ndarray | list,
):
    p_val = mannwhitneyu(data1, data2).pvalue # type: ignore fucking stupid pylance
    return p_val, get_annotation_str(p_val)


def stats_wilcoxon(
        data1: np.ndarray | list,
        data2: np.ndarray | list,
):
    assert len(data1) == len(data2), "Cannot compare arrays of different lengths for wilcoxon test"
    p_val = wilcoxon(data1, data2).pvalue # type: ignore fucking stupid pylance
    return p_val, get_annotation_str(p_val)


def stats_ks_2samp(
        data1: np.ndarray | list,
        data2: np.ndarray | list,
        **kwargs,
):
    p_val = ks_2samp(data1, data2, **kwargs).pvalue # type: ignore fucking stupid pylance
    return p_val, get_annotation_str(p_val)


def stats_huber_regress(
        x: np.ndarray | list,
        y: np.ndarray | list,
):
    assert len(x) == len(y), "Cannot compare arrays of different lengths for linear regression"
    assert not np.any(np.isnan(x)), "Cannot have NaN in x for linear regression"
    assert not np.any(np.isnan(y)), "Cannot have NaN in y for linear regression"
    huber_reg = HuberRegressor().fit(x.reshape(-1, 1), y)
    return {
        "intercept": huber_reg.intercept_,
        "slope": huber_reg.coef_[0],
    }


def stats_total_least_square_regress(
        x: np.ndarray | list,
        y: np.ndarray | list,
):
    assert len(x) == len(y), "Cannot compare arrays of different lengths for linear regression"
    assert not np.any(np.isnan(x)), "Cannot have NaN in x for linear regression"
    assert not np.any(np.isnan(y)), "Cannot have NaN in y for linear regression"
    # 1) stack and center
    XY = np.column_stack([x, y])
    mu = XY.mean(axis=0)
    XYc = XY - mu

    # 2) SVD
    # XYc = U S Vt
    U, S, Vt = np.linalg.svd(XYc, full_matrices=False)

    # 3) first right singular vector = principal direction
    vx, vy = Vt[0]

    # 4) convert direction vector to slope/intercept
    m = vy / vx
    b = mu[1] - m * mu[0]

    return {
        "intercept": b,
        "slope": m,
    }

from itertools import combinations
from statsmodels.stats.multitest import multipletests
def stats_pairwise_unpaired_ttest(data: dict[int, np.ndarray | list], ax, bon_ferroni_adj: int=1):
    keys = sorted(data)
    pairs = list(combinations(keys, 2))

    # Pairwise unpaired t-tests + multiple-testing correction
    pvals = [ttest_ind(data[a], data[b], equal_var=False).pvalue
             for a, b in pairs]
    padj = multipletests(pvals, method='bonferroni')[1] * bon_ferroni_adj

    means = [np.mean(data[k]) for k in keys]
    sems = [np.std(data[k], ddof=1) / np.sqrt(len(data[k]))
            for k in keys]
    
    # Significance brackets
    top = max(np.array(means) + np.array(sems))
    step = max(top * 0.15, 0.1)

    for i, ((a, b), p) in enumerate(zip(pairs, padj)):
        y = top + step * (i + 1)
        stars = get_stars(p)
        ax.plot([a, b],
                [y, y], c='k', lw=0.5)
        ax.text((a+b)/2, y, f'{stars}\np={p:.3g}',
                ha='center', va='bottom', fontsize=3)

def stats_pairwise_paired_ttest(data: dict[tuple[int, int], tuple[np.ndarray | list, np.ndarray | list]], ax):
    pairs = list(data)

    # Paired t-tests + FDR correction
    pvals = [ttest_rel(a, b).pvalue for a, b in data.values()]
    padj = multipletests(pvals, method='bonferroni')[1]

    # Significance brackets
    ymin, ymax = ax.get_ylim()
    step = (ymax - ymin) * 0.1

    for i, ((x1, x2), p) in enumerate(zip(pairs, padj)):
        y = ymax + step * (i + 1)

        stars = get_stars(p)
        ax.plot([x1, x2], [y, y], c='k', lw=0.5)
        ax.text((x1+x2)/2, y, f'{stars}\np={p:.3g}',
                ha='center', va='bottom', fontsize=3)


def stats_pairwise_ks(data: dict[str, np.ndarray | list], ax, loc='upper left'):
    pairs = list(combinations(data, 2))

    # Pairwise KS tests + FDR correction
    pvals = [ks_2samp(data[a], data[b]).pvalue for a, b in pairs]
    padj = multipletests(pvals, method='bonferroni')[1]

    # Format significance
    lines = [f'{a} vs {b}: {get_stars(p)} (p={p:.3g})'
             for (a, b), p in zip(pairs, padj)]

    # Position relative to axes
    x, y, ha, va = ((0.02, 0.98, 'left', 'top')
                    if loc == 'upper left'
                    else (0.98, 0.02, 'right', 'bottom'))

    ax.text(x, y, '\n'.join(lines),
            transform=ax.transAxes, ha=ha, va=va,
            fontsize=3)


def stats_anova(data: dict[int, np.ndarray | list], ax, bon_ferroni_adj: int =1):
    keys = list(data)
    groups = [np.asarray(data[k]) for k in keys]

    # One-way ANOVA + Tukey post-hoc
    F, p_anova = f_oneway(*groups)
    padj = tukey_hsd(*groups).pvalue * bon_ferroni_adj

    # Significance brackets
    ymin, ymax = ax.get_ylim()
    step = (ymax - ymin) * 0.1
    pairs = list(combinations(range(len(keys)), 2))

    for n, (i, j) in enumerate(pairs):
        p = padj[i, j]
        x1, x2 = keys[i], keys[j]
        y = ymax + step * (n + 1)

        stars = get_stars(p)

        ax.plot([x1, x2],
                [y, y], c='k', lw=0.3)
        ax.text((x1+x2)/2, y, f'{stars} (p={p:.3g})',
                ha='center', va='bottom', fontsize=3)


import statsmodels.api as sm

def add_linear_regression(ax, x_values, y_values, ci=0.95,
                          theme_color='black', offsets=0.):
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)

    # Remove invalid values
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    # Linear regression in the original scale
    model = sm.OLS(y, sm.add_constant(x)).fit()

    x_fit = np.linspace(x.min(), x.max(), 200)
    pred = model.get_prediction(
        sm.add_constant(x_fit)
    ).summary_frame(alpha=1 - ci)

    ax.plot(x_fit, pred["mean"].to_numpy(),
            color=theme_color, lw=1, zorder=-10)
    ax.fill_between(
        x_fit,
        pred["mean_ci_lower"].to_numpy(),
        pred["mean_ci_upper"].to_numpy(),
        color=theme_color, alpha=0.2, linewidth=0
    )

    p = model.pvalues[1]
    p_text = "$p < 0.001$" if p < 0.001 else f"$p = {p:.3f}$"
    slope_text = f"$\\mathrm{{slope}} = {model.params[1]:.3f}$"

    ax.text(
        0.98, 0.02 + offsets,
        f"{p_text}\n{slope_text}",
        transform=ax.transAxes,
        ha="right", va="bottom",
        color=theme_color, fontsize=3
    )

    return model


def add_loglog_regression(ax, x_values, y_values, ci=0.95, theme_color='black', offsets=0.):
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)

    # remove invalid / non-positive values
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[mask], y[mask]

    # linear regression in log-log space
    logx = np.log10(x)
    logy = np.log10(y)

    X = sm.add_constant(logx)
    model = sm.OLS(logy, X).fit()

    # smooth x coordinates in original scale
    x_fit = np.logspace(
        np.log10(x.min()),
        np.log10(x.max()),
        200
    )

    X_fit = sm.add_constant(np.log10(x_fit))

    # fitted mean + 95% CI in log space
    pred = model.get_prediction(X_fit).summary_frame(alpha=1-ci)

    # transform back to original y scale
    y_fit = 10 ** pred["mean"].to_numpy()
    y_low = 10 ** pred["mean_ci_lower"].to_numpy()
    y_high = 10 ** pred["mean_ci_upper"].to_numpy()

    # regression line
    line, = ax.plot(x_fit, y_fit, color=theme_color, lw=1, zorder=-10)

    # 95% CI band
    ax.fill_between(
        x_fit, y_low, y_high,
        color=theme_color,
        alpha=0.2,
        linewidth=0
    )

    # p-value for regression slope
    p = model.pvalues[1]
    p_text = f"$p < 0.001$" if p < 0.001 else f"$p = {p:.3f}$"
    slope = model.params[1]
    slop_text = f"$\\mathrm{{slope}} = {slope:.3f}$"
    
    ax.text(
        0.98, 0.02 + offsets,
        f"{p_text}\n{slop_text}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=theme_color,
        fontsize=3
    )

    return model








import numpy as np
from itertools import combinations
from scipy.spatial.distance import pdist
from statsmodels.stats.multitest import multipletests


def stats_pairwise_permutation(weight_matrix, cohort_name, ax=None, loc="upper left"):
    """Test differences in mean pairwise Pearson R by label permutation.

    Rows must be independent units with one cohort label each. Requires at
    least two cohorts, two rows per cohort, and finite pairwise correlations.

    Each cohort's statistic is its mean within-cohort R. Pooled within/cross
    statistics average row pairs, so larger cohorts receive more weight.
    Whole averages all row pairs, including within- and cross-cohort pairs.
    A vs B shuffles only A/B labels, preserving their sizes; comparisons
    involving pooled statistics shuffle all cohort labels. Labels must be
    exchangeable within the shuffled set under the corresponding null.

    Uses absolute mean differences, 9,999 shuffles (seed 42), and Bonferroni
    correction. Optionally annotates ax at loc. Returns similarities,
    comparisons, differences, raw/adjusted p-values, and null distributions.
    """
    X, cohort_name = map(np.asarray, (weight_matrix, cohort_name))
    if len(cohort_name) != len(X):
        raise ValueError("All inputs must have the same number of cells.")

    groups, labels = np.unique(cohort_name, return_inverse=True)
    group_names = [str(g) for g in groups]
    for g, name in enumerate(groups):
        if np.sum(labels == g) < 2:
            raise ValueError(f"Cohort {name} needs at least 2 independent units.")

    R = 1 - pdist(X, metric="correlation")
    if not np.all(np.isfinite(R)):
        raise ValueError(
            "Undefined pairwise correlations detected. "
            "Check for constant feature vectors or missing values."
        )
    i, j = np.triu_indices(len(X), k=1)
    valid = np.isfinite(R)
    pair_R, pair_u, pair_v = R[valid], i[valid], j[valid]

    def calculate_stats(labels):
        a, b = labels[pair_u], labels[pair_v]
        masks = {name: (a == g) & (b == g) for g, name in enumerate(group_names)}
        masks.update({"Within-cohort": a == b, "Cross-cohort": a != b})
        stats = {name: pair_R[mask].mean() if np.any(mask) else np.nan
                 for name, mask in masks.items()}
        stats["Whole"] = pair_R.mean()
        return stats

    observed = calculate_stats(labels)
    comparisons = list(combinations(group_names, 2))
    comparisons += [(name, "Cross-cohort") for name in group_names]
    # comparisons += [(name, "Within-cohort") for name in group_names]
    comparisons.append(("Within-cohort", "Cross-cohort"))
    comparisons += [(name, "Whole") for name in group_names]
    observed_diff = np.array([observed[a] - observed[b] for a, b in comparisons])

    n_perm = 9999
    rng = np.random.default_rng(42)
    perm_diffs = np.empty((n_perm, len(comparisons)))

    for c, (a, b) in enumerate(comparisons):
        if a in group_names and b in group_names:
            selected = np.where((labels == group_names.index(a)) | (labels == group_names.index(b)))[0]
        else:
            selected = np.arange(len(labels))

        for k in range(n_perm):
            shuffled = labels.copy()
            shuffled[selected] = rng.permutation(labels[selected])

            perm_stats = calculate_stats(shuffled)
            perm_diffs[k, c] = (
                perm_stats[a] - perm_stats[b]
            )

    pvals = (
        1 + (np.abs(perm_diffs) >= np.abs(observed_diff))
        .sum(axis=0)
    ) / (n_perm + 1)

    padj = multipletests(
        pvals, method="bonferroni"
    )[1]


    lines = [f"{a} vs {b}: {get_stars(p)} (p={p:.3g})"
             for (a, b), p in zip(comparisons, padj)]
    lines += [f"{name}: R = {observed[name]:.3f} \u00b1 "
              f"{pair_R[(labels[pair_u] == g) & (labels[pair_v] == g)].std():.3f}"
              for g, name in enumerate(group_names)]
    if ax is not None:
        x, y, ha, va = ((0.02, 0.98, "left", "top") if loc == "upper left"
                        else (0.98, 0.02, "right", "bottom"))
        ax.text(x, y, "\n".join(lines), transform=ax.transAxes,
                ha=ha, va=va, fontsize=3)

    return {
        "similarity": observed,
        "comparisons": comparisons,
        "delta_R": observed_diff,
        "pvals": pvals,
        "padj": padj,
        "null_distribution": perm_diffs,
    }
