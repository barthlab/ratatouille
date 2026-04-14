from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from snippet_type_analysis import analyze_snippet_types


@dataclass
class CaseResult:
    name: str
    metrics: Dict[str, float]
    top_channel: int
    top_window: Tuple[int, int]


def _to_snippet_list(X: np.ndarray) -> List[np.ndarray]:
    return [X[i] for i in range(X.shape[0])]


def _run_analysis(
    snippets: Sequence[np.ndarray],
    labels: np.ndarray,
    *,
    random_state: int = 0,
    n_permutations: int = 80,
) -> Dict:
    return analyze_snippet_types(
        snippets,
        labels,
        n_pca_components=20,
        n_umap_components=2,
        n_splits=5,
        n_repeats=3,
        n_permutations=n_permutations,
        window_size=20,
        random_state=random_state,
    )


def _check_core_shapes(result: Dict, n_samples: int, n_channels: int, n_time: int) -> None:
    required = {"X_shape", "pca", "umap", "classification", "permutation_test", "interpretability"}
    missing = required - set(result.keys())
    assert not missing, f"Missing keys: {missing}"

    assert result["pca"]["embedding_2d"].shape[0] == n_samples
    assert result["umap"]["embedding"].shape[0] == n_samples

    coef = result["interpretability"]["coefficient_map_outputs"]
    assert coef["mean"].shape == (n_channels, n_time)
    assert coef["std"].shape == (n_channels, n_time)
    assert coef["sign_consistency"].shape == (n_channels, n_time)

    ch_drop = result["interpretability"]["channel_importance"]["score_drop"]
    tw = result["interpretability"]["time_window_importance"]["windows"]
    tw_drop = result["interpretability"]["time_window_importance"]["score_drop"]

    assert ch_drop.shape == (n_channels,)
    assert tw.ndim == 2 and tw.shape[1] == 2
    assert tw_drop.shape[0] == tw.shape[0]


def _summarize_case(case_name: str, result: Dict) -> CaseResult:
    cls = result["classification"]
    perm = result["permutation_test"]
    ch_imp = result["interpretability"]["channel_importance"]["score_drop"]
    tw = result["interpretability"]["time_window_importance"]["windows"]
    tw_imp = result["interpretability"]["time_window_importance"]["score_drop"]

    top_channel = int(np.argmax(ch_imp))
    top_window_idx = int(np.argmax(tw_imp))
    top_window = tuple(tw[top_window_idx].tolist())

    print(f"\n=== {case_name} ===")
    print(f"balanced_accuracy : {cls['balanced_accuracy']:.3f}")
    print(f"roc_auc           : {cls['roc_auc']:.3f}")
    print(f"average_precision : {cls['average_precision']:.3f}")
    print(f"permutation p_val : {perm['p_value']:.4f}")
    print(f"top channel       : {top_channel} (drop={ch_imp[top_channel]:.4f})")
    print(f"top time window   : {top_window} (drop={tw_imp[top_window_idx]:.4f})")

    return CaseResult(
        name=case_name,
        metrics={
            "balanced_accuracy": float(cls["balanced_accuracy"]),
            "roc_auc": float(cls["roc_auc"]),
            "average_precision": float(cls["average_precision"]),
            "p_value": float(perm["p_value"]),
        },
        top_channel=top_channel,
        top_window=top_window,
    )


def _window_rank(result: Dict, target_window: Tuple[int, int]) -> int:
    windows = result["interpretability"]["time_window_importance"]["windows"]
    drops = result["interpretability"]["time_window_importance"]["score_drop"]
    order = np.argsort(drops)[::-1]

    matches = np.where((windows[:, 0] == target_window[0]) & (windows[:, 1] == target_window[1]))[0]
    assert matches.size == 1, f"Target window {target_window} not found in windows {windows.tolist()}"
    target_idx = int(matches[0])
    return int(np.where(order == target_idx)[0][0]) + 1


def _channel_rank(result: Dict, target_channel: int) -> int:
    drops = result["interpretability"]["channel_importance"]["score_drop"]
    order = np.argsort(drops)[::-1]
    return int(np.where(order == target_channel)[0][0]) + 1


def _make_balanced(
    *,
    n_per_class: int,
    n_channels: int,
    n_time: int,
    seed: int,
    noise_std: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X0 = rng.normal(0.0, noise_std, size=(n_per_class, n_channels, n_time))
    X1 = rng.normal(0.0, noise_std, size=(n_per_class, n_channels, n_time))
    X = np.concatenate([X0, X1], axis=0)
    y = np.concatenate([np.zeros(n_per_class, dtype=int), np.ones(n_per_class, dtype=int)])
    perm = rng.permutation(X.shape[0])
    return X[perm], y[perm]


def _make_imbalanced(
    *,
    n_major: int,
    n_minor: int,
    n_channels: int,
    n_time: int,
    seed: int,
    noise_std: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X_major = rng.normal(0.0, noise_std, size=(n_major, n_channels, n_time))
    X_minor = rng.normal(0.0, noise_std, size=(n_minor, n_channels, n_time))
    X = np.concatenate([X_major, X_minor], axis=0)
    y = np.concatenate([np.zeros(n_major, dtype=int), np.ones(n_minor, dtype=int)])
    perm = rng.permutation(X.shape[0])
    return X[perm], y[perm]


def _add_localized_signal(
    X: np.ndarray,
    y: np.ndarray,
    *,
    channel: int,
    start: int,
    end: int,
    amplitude: float,
) -> np.ndarray:
    X_mod = X.copy()
    X_mod[y == 1, channel, start:end] += amplitude
    return X_mod


def case_balanced_null(seed: int, n_permutations: int) -> CaseResult:
    print("Dataset: balanced null (same distribution for both classes).")
    X, y = _make_balanced(n_per_class=70, n_channels=8, n_time=120, seed=seed)
    result = _run_analysis(_to_snippet_list(X), y, random_state=seed, n_permutations=n_permutations)
    _check_core_shapes(result, X.shape[0], 8, 120)
    out = _summarize_case("Balanced Null", result)

    assert out.metrics["balanced_accuracy"] < 0.62
    assert out.metrics["p_value"] > 0.05
    return out


def case_balanced_easy(seed: int, n_permutations: int) -> Tuple[CaseResult, Tuple[int, int], int]:
    print("Dataset: balanced easy separable with strong localized minority-class signal.")
    implanted_channel = 3
    implanted_window = (40, 60)

    X, y = _make_balanced(n_per_class=60, n_channels=8, n_time=120, seed=seed)
    X = _add_localized_signal(X, y, channel=implanted_channel, start=40, end=60, amplitude=2.2)

    result = _run_analysis(_to_snippet_list(X), y, random_state=seed, n_permutations=n_permutations)
    _check_core_shapes(result, X.shape[0], 8, 120)
    out = _summarize_case("Balanced Easy", result)

    ch_rank = _channel_rank(result, implanted_channel)
    win_rank = _window_rank(result, implanted_window)
    print(f"implanted channel rank: {ch_rank}")
    print(f"implanted window rank: {win_rank}")

    assert out.metrics["balanced_accuracy"] > 0.80
    assert out.metrics["roc_auc"] > 0.90
    assert out.metrics["p_value"] < 0.05
    assert ch_rank <= 2
    assert win_rank <= 2
    return out, implanted_window, implanted_channel


def case_balanced_subtle(seed: int, n_permutations: int, easy_bal_acc: float) -> CaseResult:
    print("Dataset: balanced subtle separable with weaker localized signal.")
    X, y = _make_balanced(n_per_class=70, n_channels=8, n_time=120, seed=seed)
    X = _add_localized_signal(X, y, channel=3, start=40, end=60, amplitude=0.7)

    result = _run_analysis(_to_snippet_list(X), y, random_state=seed, n_permutations=n_permutations)
    _check_core_shapes(result, X.shape[0], 8, 120)
    out = _summarize_case("Balanced Subtle", result)

    assert out.metrics["balanced_accuracy"] > 0.56
    assert out.metrics["balanced_accuracy"] < easy_bal_acc
    return out


def case_balanced_temporal_pattern(seed: int, n_permutations: int) -> CaseResult:
    print("Dataset: balanced mean-preserving temporal-pattern difference.")
    target_channel = 2
    window = (45, 65)

    X, y = _make_balanced(n_per_class=80, n_channels=8, n_time=120, seed=seed)
    wlen = window[1] - window[0]
    half = wlen // 2
    pattern0 = np.concatenate([np.ones(half), -np.ones(wlen - half)])
    pattern1 = -pattern0

    X[y == 0, target_channel, window[0]:window[1]] += 1.6 * pattern0
    X[y == 1, target_channel, window[0]:window[1]] += 1.6 * pattern1

    result = _run_analysis(_to_snippet_list(X), y, random_state=seed, n_permutations=n_permutations)
    _check_core_shapes(result, X.shape[0], 8, 120)
    out = _summarize_case("Balanced Mean-Preserving Temporal", result)

    c0 = X[y == 0].mean(axis=0)
    c1 = X[y == 1].mean(axis=0)
    global_mean_diff = float(abs(c0.mean() - c1.mean()))
    print(f"global class-mean difference (all channels/time): {global_mean_diff:.5f}")
    print("Note: global averages remain similar while temporal structure differs.")

    assert global_mean_diff < 0.05
    assert out.metrics["balanced_accuracy"] > 0.65
    return out


def case_imbalanced_null(seed: int, n_permutations: int) -> CaseResult:
    print("Dataset: imbalanced null (about 90/10), no true class difference.")
    X, y = _make_imbalanced(n_major=180, n_minor=20, n_channels=8, n_time=120, seed=seed)

    result = _run_analysis(_to_snippet_list(X), y, random_state=seed, n_permutations=n_permutations)
    _check_core_shapes(result, X.shape[0], 8, 120)
    out = _summarize_case("Imbalanced Null (90/10)", result)

    assert out.metrics["balanced_accuracy"] < 0.62
    assert out.metrics["p_value"] > 0.05
    return out


def case_imbalanced_separable(seed: int, n_permutations: int) -> Tuple[CaseResult, Tuple[int, int], int]:
    print("Dataset: imbalanced separable (about 90/10) with localized minority-class signal.")
    implanted_channel = 4
    implanted_window = (60, 80)

    X, y = _make_imbalanced(n_major=180, n_minor=20, n_channels=8, n_time=120, seed=seed)
    X = _add_localized_signal(X, y, channel=implanted_channel, start=60, end=80, amplitude=2.0)

    result = _run_analysis(_to_snippet_list(X), y, random_state=seed, n_permutations=n_permutations)
    _check_core_shapes(result, X.shape[0], 8, 120)
    out = _summarize_case("Imbalanced Separable (90/10)", result)

    ch_rank = _channel_rank(result, implanted_channel)
    win_rank = _window_rank(result, implanted_window)
    print(f"implanted channel rank: {ch_rank}")
    print(f"implanted window rank: {win_rank}")

    assert out.metrics["balanced_accuracy"] > 0.70
    assert out.metrics["p_value"] < 0.05
    assert ch_rank <= 3
    assert win_rank <= 3
    return out, implanted_window, implanted_channel


def main() -> None:
    np.set_printoptions(precision=3, suppress=True)

    failures: List[str] = []
    passed: List[str] = []

    print("Running synthetic behavioral checks for analyze_snippet_types...\n")

    try:
        a = case_balanced_null(seed=101, n_permutations=80)
        passed.append(a.name)
    except AssertionError as e:
        failures.append(f"Balanced Null failed: {e}")

    easy_bal = np.nan
    try:
        b, _, _ = case_balanced_easy(seed=202, n_permutations=80)
        easy_bal = b.metrics["balanced_accuracy"]
        passed.append(b.name)
    except AssertionError as e:
        failures.append(f"Balanced Easy failed: {e}")

    try:
        c = case_balanced_subtle(seed=303, n_permutations=80, easy_bal_acc=easy_bal if np.isfinite(easy_bal) else 0.99)
        passed.append(c.name)
    except AssertionError as e:
        failures.append(f"Balanced Subtle failed: {e}")

    try:
        d = case_balanced_temporal_pattern(seed=404, n_permutations=80)
        passed.append(d.name)
    except AssertionError as e:
        failures.append(f"Balanced Temporal failed: {e}")

    try:
        e = case_imbalanced_null(seed=505, n_permutations=80)
        passed.append(e.name)
    except AssertionError as e:
        failures.append(f"Imbalanced Null failed: {e}")

    try:
        f, _, _ = case_imbalanced_separable(seed=606, n_permutations=80)
        passed.append(f.name)
    except AssertionError as e:
        failures.append(f"Imbalanced Separable failed: {e}")

    print("\n=== Conclusion ===")
    if not failures:
        print("Overall verdict: implementation appears correct on synthetic recovery behavior.")
        print(f"Passed cases: {len(passed)}")
    elif len(failures) <= 2:
        print("Overall verdict: implementation appears partially correct; some checks are unstable or failing.")
        print("Failures:")
        for msg in failures:
            print(f"- {msg}")
    else:
        print("Overall verdict: implementation looks suspicious based on multiple synthetic failures.")
        print("Failures:")
        for msg in failures:
            print(f"- {msg}")


if __name__ == "__main__":
    main()
