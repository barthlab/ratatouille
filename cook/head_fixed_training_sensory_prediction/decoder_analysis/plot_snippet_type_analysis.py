from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _validate_inputs(snippets: Sequence[np.ndarray], labels: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Validate plotting inputs and return stacked snippets and labels."""
    if not isinstance(snippets, Sequence) or len(snippets) == 0:
        raise ValueError("`snippets` must be a non-empty sequence of arrays.")

    arrs: List[np.ndarray] = []
    first_shape = None
    for i, s in enumerate(snippets):
        a = np.asarray(s)
        if a.ndim != 2:
            raise ValueError(f"Snippet {i} must be 2D, got shape {a.shape}.")
        if first_shape is None:
            first_shape = a.shape
        elif a.shape != first_shape:
            raise ValueError(f"All snippets must share shape {first_shape}; got {a.shape} at index {i}.")
        arrs.append(a)

    X = np.stack(arrs, axis=0)
    y = np.asarray(labels)
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("`labels` must be 1D with one label per snippet.")
    if not np.all(np.isin(np.unique(y), [0, 1])):
        raise ValueError("`labels` must be binary values in {0, 1}.")
    if np.unique(y).size < 2:
        raise ValueError("Both classes must be present for comparative plotting.")

    return X, y.astype(int)


def _require_path(d: Dict[str, Any], path: str) -> Any:
    """Get nested key path like 'a.b.c' or raise a clear error."""
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(f"Missing required results field: '{path}'")
        cur = cur[key]
    return cur


def _confusion_matrix_binary(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def _roc_curve_binary(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if np.unique(y_true).size < 2:
        raise ValueError("ROC curve is undefined with one class.")

    order = np.argsort(-y_score)
    y = y_true[order]
    p = np.sum(y == 1)
    n = np.sum(y == 0)

    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)

    tpr = np.concatenate([[0.0], tps / max(p, 1), [1.0]])
    fpr = np.concatenate([[0.0], fps / max(n, 1), [1.0]])
    return fpr, tpr


def _pr_curve_binary(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if np.unique(y_true).size < 2:
        raise ValueError("PR curve is undefined with one class.")

    order = np.argsort(-y_score)
    y = y_true[order]

    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / max(np.sum(y == 1), 1)

    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return recall, precision


def plot_dataset_heatmaps(snippets: Sequence[np.ndarray], labels: Sequence[int]) -> plt.Figure:
    """Plot class mean heatmaps and class-difference heatmap."""
    X, y = _validate_inputs(snippets, labels)

    m0 = X[y == 0].mean(axis=0)
    m1 = X[y == 1].mean(axis=0)
    d = m1 - m0

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    panels = [(m0, "Class 0 Mean"), (m1, "Class 1 Mean"), (d, "Class 1 - Class 0")]

    for ax, (img, title) in zip(axes, panels):
        vmax = np.max(np.abs(img)) if "-" in title else None
        vmin = -vmax if vmax is not None else None
        im = ax.imshow(img, aspect="auto", origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Channel")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Dataset-Level Class Average Heatmaps", fontsize=12)
    return fig


def plot_channel_average_traces(
    snippets: Sequence[np.ndarray], labels: Sequence[int], max_channels: int = 6
) -> plt.Figure:
    """Plot per-channel class-average traces over time."""
    X, y = _validate_inputs(snippets, labels)
    _, n_channels, n_time = X.shape

    m0 = X[y == 0].mean(axis=0)
    m1 = X[y == 1].mean(axis=0)

    n_plot = min(n_channels, max_channels)
    fig, axes = plt.subplots(n_plot, 1, figsize=(10, 2.2 * n_plot), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    t = np.arange(n_time)

    for ch in range(n_plot):
        ax = axes[ch]
        ax.plot(t, m0[ch], label="Class 0", lw=1.8)
        ax.plot(t, m1[ch], label="Class 1", lw=1.8)
        ax.set_ylabel(f"Ch {ch}")
        if ch == 0:
            ax.legend(loc="upper right", ncol=2)

    axes[-1].set_xlabel("Time")
    fig.suptitle("Per-Channel Class-Average Traces", fontsize=12)
    return fig


def plot_example_snippets(
    snippets: Sequence[np.ndarray],
    labels: Sequence[int],
    channels: Sequence[int] | None = None,
    n_examples_per_class: int = 5,
    random_state: int = 0,
) -> plt.Figure:
    """Overlay a few example traces from each class for selected channels."""
    X, y = _validate_inputs(snippets, labels)
    _, n_channels, n_time = X.shape
    rng = np.random.default_rng(random_state)

    if channels is None:
        channels = list(range(min(3, n_channels)))
    channels = [c for c in channels if 0 <= c < n_channels]
    if len(channels) == 0:
        raise ValueError("`channels` must contain at least one valid channel index.")

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    e0 = rng.choice(idx0, size=min(n_examples_per_class, len(idx0)), replace=False)
    e1 = rng.choice(idx1, size=min(n_examples_per_class, len(idx1)), replace=False)

    fig, axes = plt.subplots(len(channels), 1, figsize=(10, 2.5 * len(channels)), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    t = np.arange(n_time)

    for ax, ch in zip(axes, channels):
        for i in e0:
            ax.plot(t, X[i, ch], color="tab:blue", alpha=0.25, lw=1.0)
        for i in e1:
            ax.plot(t, X[i, ch], color="tab:orange", alpha=0.25, lw=1.0)

        ax.plot(t, X[y == 0, ch].mean(axis=0), color="tab:blue", lw=2.0, label="Class 0 mean")
        ax.plot(t, X[y == 1, ch].mean(axis=0), color="tab:orange", lw=2.0, label="Class 1 mean")
        ax.set_ylabel(f"Ch {ch}")

    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Time")
    fig.suptitle("Example Snippet Overlays (Selected Channels)", fontsize=12)
    return fig


def plot_embeddings(labels: Sequence[int], results: Dict[str, Any]) -> plt.Figure:
    """Plot PCA and UMAP 2D scatters colored by class label."""
    y = np.asarray(labels).astype(int)
    pca2 = np.asarray(_require_path(results, "pca.embedding_2d"))
    um = np.asarray(_require_path(results, "umap.embedding"))

    if pca2.ndim != 2 or pca2.shape[0] != y.shape[0]:
        raise ValueError("`results['pca']['embedding_2d']` must have shape (n_samples, >=2).")
    if um.ndim != 2 or um.shape[0] != y.shape[0]:
        raise ValueError("`results['umap']['embedding']` must have shape (n_samples, n_components).")

    pca_xy = pca2[:, :2]
    um_xy = um[:, :2] if um.shape[1] >= 2 else np.column_stack([um[:, 0], np.zeros(um.shape[0])])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    for ax, emb, title in [(axes[0], pca_xy, "PCA"), (axes[1], um_xy, "UMAP")]:
        for cls, color, label in [(0, "tab:blue", "Class 0"), (1, "tab:orange", "Class 1")]:
            idx = y == cls
            ax.scatter(emb[idx, 0], emb[idx, 1], s=20, alpha=0.8, c=color, label=label)
        ax.set_title(f"{title} Embedding")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.legend(loc="best")

    return fig


def plot_decoder_outputs(labels: Sequence[int], results: Dict[str, Any]) -> plt.Figure:
    """Plot OOF probability histograms, confusion matrix, ROC, and PR curves."""
    y = np.asarray(labels).astype(int)
    proba = np.asarray(_require_path(results, "classification.oof_predictions.pred_proba"))
    pred = np.asarray(_require_path(results, "classification.oof_predictions.pred_label"))

    if proba.shape[0] != y.shape[0] or pred.shape[0] != y.shape[0]:
        raise ValueError("OOF predictions must have one value per sample.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    # Histogram by class
    ax = axes[0, 0]
    ax.hist(proba[y == 0], bins=20, alpha=0.7, label="Class 0", color="tab:blue")
    ax.hist(proba[y == 1], bins=20, alpha=0.7, label="Class 1", color="tab:orange")
    ax.set_title("OOF Predicted Probability Histogram")
    ax.set_xlabel("Predicted P(class=1)")
    ax.set_ylabel("Count")
    ax.legend()

    # Confusion matrix
    ax = axes[0, 1]
    cm = _confusion_matrix_binary(y, pred)
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title("Confusion Matrix (OOF)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ROC curve
    ax = axes[1, 0]
    try:
        fpr, tpr = _roc_curve_binary(y, proba)
        ax.plot(fpr, tpr, lw=2)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title("ROC Curve (OOF)")
    except ValueError:
        ax.text(0.5, 0.5, "ROC unavailable\n(only one class)", ha="center", va="center")
        ax.set_title("ROC Curve")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")

    # PR curve
    ax = axes[1, 1]
    try:
        recall, precision = _pr_curve_binary(y, proba)
        ax.plot(recall, precision, lw=2)
        ax.set_title("Precision-Recall Curve (OOF)")
    except ValueError:
        ax.text(0.5, 0.5, "PR unavailable\n(only one class)", ha="center", va="center")
        ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    return fig


def plot_interpretability(results: Dict[str, Any]) -> plt.Figure:
    """Plot coefficient maps and occlusion importances with warnings."""
    coef_mean = np.asarray(_require_path(results, "interpretability.coefficient_map_outputs.mean"))
    coef_std = np.asarray(_require_path(results, "interpretability.coefficient_map_outputs.std"))
    coef_sign = np.asarray(_require_path(results, "interpretability.coefficient_map_outputs.sign_consistency"))
    ch_drop = np.asarray(_require_path(results, "interpretability.channel_importance.score_drop"))
    windows = np.asarray(_require_path(results, "interpretability.time_window_importance.windows"))
    tw_drop = np.asarray(_require_path(results, "interpretability.time_window_importance.score_drop"))

    if coef_mean.shape != coef_std.shape or coef_mean.shape != coef_sign.shape:
        raise ValueError("Coefficient mean/std/sign_consistency must have identical shapes (channels, time).")
    if ch_drop.ndim != 1:
        raise ValueError("Channel importance score_drop must be 1D.")
    if windows.ndim != 2 or windows.shape[1] != 2:
        raise ValueError("Time-window `windows` must have shape (n_windows, 2).")
    if tw_drop.shape[0] != windows.shape[0]:
        raise ValueError("Time-window score_drop length must match number of windows.")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

    hm_specs = [
        (coef_mean, "Coefficient Mean", "RdBu_r"),
        (coef_std, "Coefficient Std", "viridis"),
        (coef_sign, "Sign Consistency", "magma"),
    ]
    for ax, (img, title, cmap) in zip(axes[0], hm_specs):
        vmax = np.max(np.abs(img)) if cmap == "RdBu_r" else None
        vmin = -vmax if vmax is not None else None
        im = ax.imshow(img, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Channel")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    ax.bar(np.arange(ch_drop.shape[0]), ch_drop)
    ax.set_title("Channel Importance (Score Drop)")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Delta Balanced Accuracy")

    ax = axes[1, 1]
    x = np.arange(windows.shape[0])
    labels = [f"{int(s)}-{int(e)}" for s, e in windows]
    ax.bar(x, tw_drop)
    ax.set_title("Time-Window Importance (Score Drop)")
    ax.set_xlabel("Window [start-end]")
    ax.set_ylabel("Delta Balanced Accuracy")
    if len(labels) <= 20:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")

    ax = axes[1, 2]
    warn_texts: List[str] = []
    warn_single = results.get("interpretability", {}).get("importance_warning", None)
    warn_list = results.get("interpretability", {}).get("importance_warnings", None)
    if isinstance(warn_single, str) and warn_single:
        warn_texts.append(warn_single)
    if isinstance(warn_list, list):
        warn_texts.extend([str(w) for w in warn_list if str(w)])

    if warn_texts:
        ax.text(0.02, 0.98, "\n".join(warn_texts), va="top", ha="left", wrap=True)
    else:
        ax.text(0.02, 0.98, "No interpretability warnings.", va="top", ha="left")
    ax.set_title("Interpretability Notes")
    ax.axis("off")

    return fig


def plot_permutation_test(results: Dict[str, Any]) -> plt.Figure:
    """Plot null-score histogram and true-score line for permutation test."""
    null_scores = np.asarray(_require_path(results, "permutation_test.null_scores"))
    true_score = float(_require_path(results, "permutation_test.true_score"))
    p_value = float(_require_path(results, "permutation_test.p_value"))

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5), constrained_layout=True)
    ax.hist(null_scores, bins=20, alpha=0.8, color="gray", edgecolor="black")
    ax.axvline(true_score, color="red", linestyle="--", linewidth=2, label=f"True score = {true_score:.3f}")
    ax.set_xlabel("Balanced Accuracy")
    ax.set_ylabel("Count")
    ax.set_title(f"Permutation Test Null Distribution (p = {p_value:.4f})")
    ax.legend()
    return fig


def plot_analysis_dashboard(
    snippets: Sequence[np.ndarray], labels: Sequence[int], results: Dict[str, Any]
) -> Dict[str, plt.Figure]:
    """Create all requested analysis plots and return a dict of figures."""
    figs = {
        "dataset_heatmaps": plot_dataset_heatmaps(snippets, labels),
        "channel_traces": plot_channel_average_traces(snippets, labels),
        "example_overlays": plot_example_snippets(snippets, labels),
        "embeddings": plot_embeddings(labels, results),
        "decoder": plot_decoder_outputs(labels, results),
        "interpretability": plot_interpretability(results),
        "permutation": plot_permutation_test(results),
    }
    return figs


if __name__ == "__main__":
    # Minimal usage example:
    
    import warnings 
    warnings.filterwarnings("ignore")
    import numpy as np
    import matplotlib.pyplot as plt
    from snippet_type_analysis import analyze_snippet_types
    from plot_snippet_type_analysis import plot_analysis_dashboard
    
    rng = np.random.default_rng(0)
    snippets = [rng.normal(size=(8, 120)) for _ in range(120)]
    labels = np.array([0] * 60 + [1] * 60)
    for i in range(60, 120):
        snippets[i][3, 40:60] += 1.5
    
    # Run the analysis and plot the dashboard
    results = analyze_snippet_types(snippets, labels, random_state=0)
    plot_analysis_dashboard(snippets, labels, results)
    plt.show()
