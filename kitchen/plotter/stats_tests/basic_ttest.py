import numpy as np
from scipy.stats import ttest_ind, ttest_rel, mannwhitneyu, wilcoxon, ks_2samp
from scipy.stats import linregress, pearsonr
from sklearn.linear_model import HuberRegressor


SIGNIFICANT_ALPHA = 0.05

def get_annotation_str(p_val: float, verbose: bool = True, double_line: bool = False):
    if p_val < 0.001:
        asterisks = r"$\ast$"*3
    elif p_val < 0.01:
        asterisks = r"$\ast$"*2
    elif p_val < 0.05:
        asterisks = r"$\ast$"
    else:
        asterisks = 'n.s.'
    
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