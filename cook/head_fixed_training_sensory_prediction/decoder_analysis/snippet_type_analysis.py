from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import umap.umap_ as umap
except ImportError as exc:  # pragma: no cover
    raise ImportError("umap-learn is required. Install with `pip install umap-learn`.") from exc


ArrayLike = np.ndarray


def _validate_inputs(snippets: Sequence[ArrayLike], labels: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Validate snippet/label inputs and return stacked snippets with binary labels."""
    if not isinstance(snippets, Sequence) or len(snippets) == 0:
        raise ValueError("`snippets` must be a non-empty sequence of arrays.")

    first_shape = None
    processed: List[np.ndarray] = []

    for i, snip in enumerate(snippets):
        arr = np.asarray(snip)
        if arr.ndim != 2:
            raise ValueError(f"Snippet at index {i} must be 2D, got shape {arr.shape}.")
        if first_shape is None:
            first_shape = arr.shape
        elif arr.shape != first_shape:
            raise ValueError(
                f"All snippets must have the same shape. Expected {first_shape}, got {arr.shape} at index {i}."
            )
        processed.append(arr.astype(float, copy=False))

    X3 = np.stack(processed, axis=0)

    y = np.asarray(labels).astype(int, copy=False)
    if y.ndim != 1 or y.shape[0] != X3.shape[0]:
        raise ValueError("`labels` must be 1D with one label per snippet.")

    uniq = np.unique(y)
    if not np.array_equal(uniq, [0, 1]) and not np.array_equal(uniq, [0]) and not np.array_equal(uniq, [1]):
        raise ValueError("`labels` must be binary with values in {0, 1}.")
    if uniq.size < 2:
        raise ValueError("Both classes must be present in `labels` for analysis.")

    return X3, y


def _flatten_snippets(X3: np.ndarray) -> np.ndarray:
    """Flatten snippets from (n_samples, n_channels, n_time) to (n_samples, n_channels*n_time)."""
    return X3.reshape(X3.shape[0], -1)


def _make_decoder(random_state: int) -> Pipeline:
    """Build a class-balanced linear decoder for interpretable binary classification."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    l1_ratio=0,
                    C=1.0,
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _generate_splits(y: np.ndarray, n_splits: int, n_repeats: int, random_state: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate repeated stratified CV splits."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    return list(cv.split(np.zeros_like(y), y))


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _cross_validated_predictions(
    X: np.ndarray,
    y: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    random_state: int,
    collect_coefficients: bool = False,
    collect_oof: bool = True,
) -> Dict[str, Any]:
    """Run repeated stratified CV and return fold metrics (primary) plus optional OOF outputs.

    Fold-wise metrics are primary for repeated CV. Aggregated OOF predictions are for
    visualization/qualitative inspection and not the primary score summary.
    """
    n_samples = X.shape[0]
    prob_sum = np.zeros(n_samples, dtype=float) if collect_oof else None
    counts = np.zeros(n_samples, dtype=int) if collect_oof else None
    fold_scores: List[Dict[str, float]] = []
    coefs: List[np.ndarray] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        model = _make_decoder(random_state=random_state + fold_idx)
        model.fit(X[train_idx], y[train_idx])

        prob = model.predict_proba(X[test_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)

        if collect_oof:
            prob_sum[test_idx] += prob
            counts[test_idx] += 1

        fold_scores.append(
            {
                "fold": int(fold_idx),
                "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], pred)),
                "roc_auc": _safe_roc_auc(y[test_idx], prob),
                "average_precision": _safe_average_precision(y[test_idx], prob),
            }
        )

        if collect_coefficients:
            scaler: StandardScaler = model.named_steps["scaler"]
            clf: LogisticRegression = model.named_steps["clf"]
            scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
            coefs.append(clf.coef_.ravel() / scale)

    bal_scores = np.asarray([row["balanced_accuracy"] for row in fold_scores], dtype=float)
    roc_scores = np.asarray([row["roc_auc"] for row in fold_scores], dtype=float)
    ap_scores = np.asarray([row["average_precision"] for row in fold_scores], dtype=float)

    out: Dict[str, Any] = {
        "cv_metrics_summary": {
            "balanced_accuracy": {"mean": float(np.nanmean(bal_scores)), "std": float(np.nanstd(bal_scores))},
            "roc_auc": {"mean": float(np.nanmean(roc_scores)), "std": float(np.nanstd(roc_scores))},
            "average_precision": {"mean": float(np.nanmean(ap_scores)), "std": float(np.nanstd(ap_scores))},
        },
        "fold_scores": fold_scores,
        "balanced_accuracy": float(np.nanmean(bal_scores)),
        "roc_auc": float(np.nanmean(roc_scores)),
        "average_precision": float(np.nanmean(ap_scores)),
    }

    if collect_oof:
        assert prob_sum is not None and counts is not None
        mean_prob = prob_sum / np.maximum(counts, 1)
        out["oof_predictions"] = {
            "pred_proba": mean_prob,
            "pred_label": (mean_prob >= 0.5).astype(int),
            "counts": counts,
            "note": "Aggregated repeated-CV OOF predictions; mainly for plotting/inspection.",
        }

    if collect_coefficients:
        out["coefficients"] = np.stack(coefs, axis=0)

    return out


def _cv_balanced_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    random_state: int,
) -> float:
    """Mean fold-wise balanced accuracy for a fixed CV split geometry."""
    out = _cross_validated_predictions(X, y, splits, random_state, collect_oof=False)
    return float(out["cv_metrics_summary"]["balanced_accuracy"]["mean"])


def _window_slices(n_time: int, window_size: int) -> List[Tuple[int, int]]:
    if window_size <= 0:
        raise ValueError("`window_size` must be > 0.")
    return [(start, min(start + window_size, n_time)) for start in range(0, n_time, window_size)]


def analyze_snippet_types(
    snippets: Sequence[ArrayLike],
    labels: Sequence[int],
    *,
    n_pca_components: int = 20,
    n_umap_components: int = 2,
    n_splits: int = 5,
    n_repeats: int = 5,
    n_permutations: int = 200,
    window_size: int = 30,
    random_state: int = 0,
) -> Dict[str, Any]:
    """Analyze whether binary snippet types are separable using one raw-snippet pipeline.

    Pipeline:
    1) Validate inputs and flatten raw snippets to X_flat.
    2) Build PCA/UMAP embeddings from standardized X_flat.
    3) Decode with repeated stratified CV on X_flat using class-balanced logistic regression.
    4) Report fold-wise metrics as primary outputs and OOF predictions for visualization.
    5) Run a permutation test (balanced accuracy) with fixed split geometry.
    6) Compute coefficient maps and occlusion importances over channels/time windows.

    Notes:
    - `class_weight="balanced"` handles class imbalance during decoding.
    - Fold-wise repeated-CV metrics are primary; OOF outputs are secondary.
    - Occlusion importance can be weak/degenerate when score drops are near zero.
    """
    if n_pca_components <= 0:
        raise ValueError("`n_pca_components` must be > 0.")
    if n_umap_components <= 0:
        raise ValueError("`n_umap_components` must be > 0.")
    if n_splits <= 1:
        raise ValueError("`n_splits` must be >= 2.")
    if n_repeats <= 0:
        raise ValueError("`n_repeats` must be > 0.")
    if n_permutations < 0:
        raise ValueError("`n_permutations` must be >= 0.")

    X3, y = _validate_inputs(snippets, labels)
    n_samples, n_channels, n_time = X3.shape
    X_flat = _flatten_snippets(X3)

    # Embeddings on standardized raw flattened features.
    X_scaled = StandardScaler().fit_transform(X_flat)
    pca_n = max(1, min(n_pca_components, X_scaled.shape[0], X_scaled.shape[1]))
    pca_model = PCA(n_components=pca_n, random_state=random_state)
    X_pca = pca_model.fit_transform(X_scaled)
    pca_2d = X_pca[:, :2] if X_pca.shape[1] >= 2 else np.column_stack([X_pca[:, 0], np.zeros(n_samples)])

    umap_model = umap.UMAP(
        n_components=n_umap_components,
        random_state=random_state,
        init="spectral",
        n_neighbors=min(15, max(2, n_samples - 1)),
    )
    X_umap = umap_model.fit_transform(X_pca)

    try:
        splits = _generate_splits(y, n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    except ValueError as exc:
        raise ValueError(
            "`n_splits` is too large for the minority class size; each fold must contain both classes."
        ) from exc

    cls = _cross_validated_predictions(X_flat, y, splits, random_state=random_state, collect_coefficients=False)

    # Permutation test: fixed fold geometry, shuffled labels.
    rng = np.random.default_rng(random_state)
    perm_metric = "balanced_accuracy"
    true_score = float(cls["cv_metrics_summary"][perm_metric]["mean"])
    null_scores = np.zeros(n_permutations, dtype=float)
    for i in range(n_permutations):
        null_scores[i] = _cv_balanced_accuracy(X_flat, rng.permutation(y), splits, random_state=random_state + i + 1)
    p_value = float((1.0 + np.sum(null_scores >= true_score)) / (n_permutations + 1.0))

    interp_cv = _cross_validated_predictions(X_flat, y, splits, random_state=random_state, collect_coefficients=True)
    coef_maps = interp_cv["coefficients"].reshape(len(splits), n_channels, n_time)

    mean_coef_map = coef_maps.mean(axis=0)
    std_coef_map = coef_maps.std(axis=0)
    sign_consistency_map = np.abs(np.sign(coef_maps).mean(axis=0))

    baseline_score = float(interp_cv["cv_metrics_summary"]["balanced_accuracy"]["mean"])

    channel_score_drop = np.zeros(n_channels, dtype=float)
    for ch in range(n_channels):
        X_mod = X3.copy()
        X_mod[:, ch, :] = 0.0
        channel_score_drop[ch] = baseline_score - _cv_balanced_accuracy(
            _flatten_snippets(X_mod), y, splits, random_state=random_state
        )

    windows = _window_slices(n_time, window_size=window_size)
    window_score_drop = np.zeros(len(windows), dtype=float)
    for i, (start, end) in enumerate(windows):
        X_mod = X3.copy()
        X_mod[:, :, start:end] = 0.0
        window_score_drop[i] = baseline_score - _cv_balanced_accuracy(
            _flatten_snippets(X_mod), y, splits, random_state=random_state
        )

    importance_eps = 1e-6
    importance_warnings: List[str] = []
    if np.nanmax(np.abs(channel_score_drop)) <= importance_eps:
        importance_warnings.append(
            "Channel occlusion score drops are near zero (<=1e-6); importance may be numerically degenerate."
        )
    if np.nanmax(np.abs(window_score_drop)) <= importance_eps:
        importance_warnings.append(
            "Time-window occlusion score drops are near zero (<=1e-6); importance may be numerically degenerate."
        )

    return {
        "X_shape": {
            "n_samples": n_samples,
            "n_channels": n_channels,
            "n_time": n_time,
            "X_flat": X_flat.shape,
        },
        "pca": {
            "embedding_2d": pca_2d,
            "embedding": X_pca,
            "explained_variance_ratio": pca_model.explained_variance_ratio_,
            "n_components": pca_n,
        },
        "umap": {
            "embedding": X_umap,
            "n_components": n_umap_components,
        },
        "classification": {
            "primary_evaluation": "fold-wise repeated-stratified-CV metrics",
            "cv_metrics_summary": cls["cv_metrics_summary"],
            "fold_scores": cls["fold_scores"],
            "oof_predictions": cls["oof_predictions"],
            # Convenience aliases (fold-wise means).
            "balanced_accuracy": cls["balanced_accuracy"],
            "roc_auc": cls["roc_auc"],
            "average_precision": cls["average_precision"],
        },
        "permutation_test": {
            "true_score": true_score,
            "null_scores": null_scores,
            "p_value": p_value,
            "n_permutations": n_permutations,
            "metric": perm_metric,
        },
        "interpretability": {
            "coefficient_map_outputs": {
                "mean": mean_coef_map,
                "std": std_coef_map,
                "sign_consistency": sign_consistency_map,
                "all_folds": coef_maps,
            },
            "channel_importance": {
                "score_drop": channel_score_drop,
            },
            "time_window_importance": {
                "windows": np.asarray(windows, dtype=int),
                "score_drop": window_score_drop,
                "window_size": window_size,
            },
            "importance_warning": None if not importance_warnings else " | ".join(importance_warnings),
            "importance_warnings": importance_warnings,
            "degenerate_threshold": importance_eps,
        },
    }


# Usage example:
# import numpy as np
# from snippet_type_analysis import analyze_snippet_types
#
# snippets = [np.random.randn(8, 200) for _ in range(20)]
# labels = np.array([0] * 10 + [1] * 10)
# results = analyze_snippet_types(snippets, labels)
