import logging
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt     
import seaborn as sns
import matplotlib.patheffects as pe
import numpy as np
import os.path as path
import os
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA, TruncatedSVD, FactorAnalysis, SparsePCA, NMF, FastICA

from kitchen.calculator import calculate_physiology
from kitchen.configs import routing
from kitchen.loader.general_loader_interface import load_dataset
from kitchen.operator.grouping import grouping_timeseries
from kitchen.plotter import color_scheme
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_FeatureSpace import get_weight_tuple_PSTH
from kitchen.plotter.utils.tick_labels import add_textonly_legend
from kitchen.utils.ordering_kit import linkage_order
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_Settings import CLUSTER_COLORS, COHORT_COLORS, COHORT_BRIGHTER_COLORS
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_SingleCell import get_500ms_puff_trials
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_SummaryMetric import get_all_cellsession_PSTH_mean, get_all_cellsession_Waveform_mean, get_saving_path
from kitchen.settings.potential import NARROW_SPIKE_RANGE_FOR_VISUALIZATION, SPIKE_ANNOTATION_EARLY_WINDOW
from kitchen.structure.neural_data_structure import TimeSeries
from kitchen.utils import numpy_kit

logger = logging.getLogger(__name__)


def variability_visualize(variant='raw',
                          task_name: str="VariabilityVisualize", **kwargs):
    PSTH_weight_matrix, node_names, cohort_names = get_weight_tuple_PSTH(variant, **kwargs)

    bins = np.arange(-1, 1.5 + kwargs["binsize"], kwargs["binsize"])
    bin_centers = (bins[:-1] + bins[1:]) / 2


    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 6
    # n_sep = 5

    n_cells, n_timepoints = PSTH_weight_matrix.shape
    logger.info(f"Performing stability test for SVD with on "
                f"PSTH data ({variant}) with {n_cells} cells and {n_timepoints} timepoints.")
    
    fig, axs = plt.subplots(1, 2, figsize=(3.9, 0.9), constrained_layout=True)
    for ax in axs:
        ax.spines[['right', 'top',]].set_visible(False)
        ax.tick_params(length=1, pad=1, which='both')
    variances = []
    for cohort_name in ("PYR", "PV", "SST"):
        cohort_mask = np.logical_or(np.array(cohort_names) == cohort_name + "_JUX",
                                    np.array(cohort_names) == cohort_name + "_WC")
        sub_psth = PSTH_weight_matrix[cohort_mask]
        variance = np.var(sub_psth, axis=0, ddof=1)
        variances.append(variance)
    variances = np.array(variances)
    total_variance = np.sum(variances, axis=0)   
    global_variance = np.std(PSTH_weight_matrix, axis=0, ddof=1)
    percentages = np.divide(
        variances * 100,
        total_variance,
        out=np.zeros_like(variances),
        where=total_variance > 0
    )

    axs[0].plot(bin_centers, global_variance, color='black', lw=0.5, alpha=1)
    axs[0].set_yticks([0, 50, 100], ["0", "50", "100"])
    axs[0].set_ylabel("SD (Hz)", labelpad=1)
    axs[0].set_xticks([0, 0.5], ["0", "0.5"])
    axs[0].set_xlim(-0.5, 1)
    axs[0].set_xlabel("Time from puff onset (s)", labelpad=1)    
    axs[0].axvspan(0, 0.5, 
                alpha=0.3, color=color_scheme.PUFF_COLOR, lw=0, zorder=-10)
    # axs[0].set_yscale('symlog', linthresh=1, linscale=0.1)

    
    axs[1].stackplot(bin_centers, *percentages, 
                     linewidth=0.2, edgecolor=(0, 0, 0, 0.6), linestyle='-', 
                    alpha=1, colors=[COHORT_BRIGHTER_COLORS[cohort_name+"_JUX"] 
                                        for cohort_name in ("PYR", "PV", "SST")])
    axs[1].axvline(0, color='black', lw=0.5, alpha=0.5, ls='--')
    axs[1].axvline(0.5, color='black', lw=0.5, alpha=0.5, ls='--')
    axs[1].set_xlim(-0.5, 1)
    axs[1].set_xticks([0, 0.5], ["0", "0.5"])
    axs[1].set_xlabel("Time from puff onset (s)", labelpad=1)
    axs[1].set_yticks([0, 50, 100], ["0", "50", "100"], rotation=90, ha='right', va='center')
    axs[1].set_ylim(0, 100)    
    axs[1].set_ylabel("Variance ratio (%)", labelpad=1)
    axs[1].spines[['right', 'top', 'bottom', 'left']].set_visible(True)

    save_path = path.join(get_saving_path(), task_name, 
                        f"VarianceDist_{variant}_SVD.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)


def stability_test_SVD(variant,
                       n_sep: int = 5,
                       task_name: str="StabilityTest", **kwargs):
    PSTH_weight_matrix, node_names, cohort_names = get_weight_tuple_PSTH(variant, **kwargs)

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 6
    # n_sep = 5

    n_cells, n_timepoints = PSTH_weight_matrix.shape
    logger.info(f"Performing stability test for SVD with on "
                f"PSTH data ({variant}) with {n_cells} cells and {n_timepoints} timepoints.")


    def visualize_raw_and_reconstruct_matrix():
        bins = np.arange(-1, 1.5 + kwargs["binsize"], kwargs["binsize"])
        bin_centers = (bins[:-1] + bins[1:]) / 2

        solver = TruncatedSVD(n_components=n_sep, random_state=42)
        component_projection = solver.fit_transform(PSTH_weight_matrix)
        reconstruct_matrix = solver.inverse_transform(component_projection)
        
        mask = (bin_centers > 0) & (bin_centers < 0.5)
        sorted_order = np.argsort(np.mean(PSTH_weight_matrix[:, mask], axis=1))

        fig, axs = plt.subplots(1, 2, figsize=(2.2, 1.2), constrained_layout=True)

        for ax, matrix in zip(axs, [PSTH_weight_matrix, reconstruct_matrix]):
            sns.heatmap(matrix[sorted_order],
                                ax=ax, cbar=False,
                                cmap='Spectral', 
                                vmin=-2, vmax=2,)
                                # norm=SymLogNorm(linthresh=1., vmin=0, vmax=100, linscale=0.1),)
                    
            desired_ticks = [0, 0.5]
            tick_positions = [np.argmin(np.abs(bin_centers - tick)) for tick in desired_ticks]
            ax.set_xticks(tick_positions, desired_ticks, rotation=0)  

            ax.invert_yaxis()
            ax.set_xlabel("Time (s)", labelpad=1)
            ax.set_yticks([0.5, len(PSTH_weight_matrix)-0.5], 
                            [f"{len(PSTH_weight_matrix)}", "1"], rotation=0)
            ax.spines[['right', 'top', 'left', 'bottom']].set_visible(True)
            ax.tick_params(length=1, pad=1)

        save_path = path.join(get_saving_path(), task_name, 
                            f"RawVSReconstruct_{variant}_SVD_{n_sep}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=500, transparent=True)
        plt.close(fig)
        logger.info("Plot saved to " + save_path)


    visualize_raw_and_reconstruct_matrix()

    def evr_sv_overview():
        n_components = 15
        solver = TruncatedSVD(n_components=n_components, random_state=42)
        component_projection = solver.fit_transform(PSTH_weight_matrix)
        
        fig, ax = plt.subplots(1, 1, figsize=(1.4, 1.1), constrained_layout=True)
        
        ax.tick_params(
            length=1,
            pad=1,
        )
        ax.plot(np.arange(n_components) + 1, solver.explained_variance_ratio_, 
                marker='o', color='black', lw=1, markersize=3, alpha=0.8)
        ax.set_xlabel("Component ID", labelpad=1)
        ax.set_ylabel("Explained Variance Ratio", color='black', labelpad=1)
        ax.spines['top'].set_visible(False)
        ax.set_xticks([1, n_sep, n_sep+1, 10, 15])
        ax.set_yticks([0, 0.1, 0.2])


        ax2 = ax.twinx()
        ax2.tick_params(
            length=1,
            pad=1,
        )
        ax2.plot(np.arange(n_components) + 1, solver.singular_values_, 
                marker='o', color='gray', lw=1, ls='--', markersize=1.5, zorder=-1, alpha=0.8)
        ax2.set_ylabel("Singular Values", color='gray', labelpad=1)
        ax2.set_yticks([0, 50, 100], ["0", "50", "100"], color='gray')
                    
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_color('gray')

        ax.axvline(x=n_sep + 0.5, color='red', lw=1, ls='--', alpha=0.8)

        save_path = path.join(get_saving_path(), task_name, 
                            f"ExplainedVariance_{variant}_SVD_{n_components}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=500, transparent=True)
        plt.close(fig)
        logger.info("Plot saved to " + save_path)

    evr_sv_overview()


    n_components=15
    n_repeats=1000
    random_state=42
    
    X = np.asarray(PSTH_weight_matrix, dtype=float)
    n_samples, n_features = X.shape
    
    rng_stability = np.random.default_rng(random_state + 1)
    svd_seed = random_state

    pcs = np.arange(1, n_components + 1)
    full_svd = TruncatedSVD(
        n_components=n_components,
        random_state=svd_seed,
    )

    full_svd.fit(X)
    full_evr = full_svd.explained_variance_ratio_.copy()
    full_components = full_svd.components_.copy()

    # ============================================================
    # FIGURE 2
    # 5-fold matrix bi-cross-validation
    # ============================================================
    #
    # For each fold, split BOTH rows and features:
    #
    #                    feature columns
    #
    #                train          held out
    #
    # train rows        A              B
    #
    # held-out rows     C              D
    #
    #
    # SVD is fit ONLY to A.
    #
    # Then:
    #
    #       D_hat = C @ A_k^+ @ B
    #
    # where A_k^+ is the rank-k pseudoinverse of A.
    #
    # We compare D_hat with the completely held-out block D.
    #
    # ------------------------------------------------------------
    def fold_matrix_bi_cross_validation(temp_random_seed, cv_folds=5):
        # Approximate smallest training block in BCV
        min_train_rows = (
            n_samples - int(np.ceil(n_samples / cv_folds))
        )
        min_train_cols = (
            n_features - int(np.ceil(n_features / cv_folds))
        )

        if n_components >= min(min_train_rows, min_train_cols):
            raise ValueError(
                "n_components is too large for the bi-cross-validation "
                f"training blocks. Use n_components < "
                f"{min(min_train_rows, min_train_cols)}."
            )

        
        row_kfold = KFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=temp_random_seed,
        )

        feature_kfold = KFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=temp_random_seed + 1,
        )

        row_splits = list(row_kfold.split(np.arange(n_samples)))
        feature_splits = list(feature_kfold.split(np.arange(n_features)))

        # rows = folds
        # columns = candidate ranks
        cv_errors = np.empty((cv_folds, n_components), dtype=float,)

        for fold in range(cv_folds):
            train_rows, test_rows = row_splits[fold]
            train_features, test_features = feature_splits[fold]

            A = X[np.ix_(
                train_rows,
                train_features,
            )]

            B = X[np.ix_(
                train_rows,
                test_features,
            )]

            C = X[np.ix_(
                test_rows,
                train_features,
            )]

            D = X[np.ix_(
                test_rows,
                test_features,
            )]

            svd = TruncatedSVD(
                n_components=n_components,
                random_state=svd_seed,
            )
            A_scores = svd.fit_transform(A)
            s = svd.singular_values_
            V = svd.components_.T
            U = A_scores / s

            denominator = np.linalg.norm(D, ord="fro",)

            if denominator == 0:
                raise ValueError(f"Held-out block in fold {fold + 1} has zero Frobenius norm." )

            for k in range(1, n_components + 1,):
                Vk = V[:, :k]
                Uk = U[:, :k]
                sk = s[:k]

                # Rank-k pseudoinverse:
                #
                # A_k^+ =
                # V_k @ diag(1 / s_k) @ U_k.T

                A_pinv_k = Vk @ np.diag(1.0 / sk) @ Uk.T

                # Predict completely held-out block
                D_hat = C @ A_pinv_k  @ B

                # Normalized Frobenius error
                error = np.linalg.norm(D - D_hat, ord="fro", ) / denominator
                cv_errors[fold, k - 1] = error
        return cv_errors


    def generalization_plot():
        fig, ax = plt.subplots(
            1, 1, 
            figsize=(1.2, 1.1),
            constrained_layout=True,
        )
                
        ax.tick_params(
            length=1,
            pad=1,
        )

        all_cv_errors = []
        for seed in range(n_repeats):
            errors = fold_matrix_bi_cross_validation(seed, cv_folds=5)
            all_cv_errors.append(errors)

        all_cv_errors = np.concatenate(all_cv_errors, axis=0)
        mean_error = all_cv_errors.mean(axis=0)
        std_error = all_cv_errors.std(axis=0, ddof=1)

        ax.errorbar(pcs, 
                    mean_error, yerr=std_error,
                    fmt="o", color='black',
                    markersize=3, lw=1, alpha=0.9, markeredgewidth=0.5,
                    ecolor='black', elinewidth=0.5,
                    capsize=1.5, capthick=0.5,)

        ax.axvline(x=n_sep + 0.5, color='red', lw=1, ls='--', alpha=0.8)
        ax.set_xlabel("Retained rank (k)", labelpad=1)
        ax.set_xticks([1, n_sep, n_sep+1, 10, 15])
        ax.set_ylabel("Held-out\nReconstruction error", color='black', labelpad=1)
        minimal_error = mean_error.min()
        ax.axhline(y=minimal_error, color='gray', lw=1, ls='--', alpha=0.8)
        # ax.set_title("5-fold Bi-cross-validation")
        ax.spines[['top', 'right']].set_visible(False)


        save_path = path.join(get_saving_path(), task_name, 
                            f"Fold_Bicrossvalidation_{variant}_SVD_{n_components}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=500, transparent=True)
        plt.close(fig)
        logger.info("Plot saved to " + save_path)


    generalization_plot()


    # ============================================================
    # FIGURE 3
    # Loading / eigenvector stability
    # ============================================================
    #
    # EXACTLY as requested:
    #
    # full PC1 <-> subset PC1
    # full PC2 <-> subset PC2
    # ...
    #
    # No Hungarian matching.
    # No PC switching.
    # No subspace correction.
    #
    # Similarity:
    #
    #     | v_full dot v_subset |
    #
    # TruncatedSVD loading vectors are unit vectors, so this is
    # absolute cosine similarity.
    #
    # Absolute value handles arbitrary sign flips.
    # ============================================================

    subset_fracs=(0.90, 0.75, 0.50)
    stability = {}

    # Smallest row-subsampled dataset
    smallest_subset_n = min(
        int(np.floor(f * n_samples))
        for f in subset_fracs
    )

    if n_components >= min(smallest_subset_n, n_features):
        raise ValueError(
            "n_components is too large for the smallest requested "
            f"sample subset. Use n_components < "
            f"{min(smallest_subset_n, n_features)}."
        )

    
    for frac in subset_fracs:
        n_sub = int(np.floor(frac * n_samples))

        similarities = np.empty(
            (n_repeats, n_components), dtype=float,)

        for repeat in range(n_repeats):
            sample_idx = rng_stability.choice(n_samples, size=n_sub, replace=False)

            svd = TruncatedSVD(n_components=n_components, random_state=svd_seed,)

            svd.fit(X[sample_idx])

            subset_components = svd.components_

            # PCj <-> PCj directly
            #
            # Each row of components_ is unit norm.
            similarities[repeat] = np.abs(np.sum(full_components * subset_components, axis=1))

        stability[frac] = similarities

    def stability_summary():

        fig, ax = plt.subplots(
                    1, 1, 
                    figsize=(1.4, 1.1),
                    constrained_layout=True,
                )
                        
        ax.tick_params(
            length=1,
            pad=1,
        )
        n_fractions = len(subset_fracs)
        total_bar_width = 0.5
        bar_width = total_bar_width / (n_fractions - 1)

        for frac_id, frac in zip(
            range(len(subset_fracs)),
            subset_fracs,
        ):

            result = stability[frac]
            mean_similarity = np.median(result, axis=0)
            upper_similarity = np.percentile(result, 75, axis=0)    
            lower_similarity = np.percentile(result, 25, axis=0)
            ax.errorbar(
                pcs + frac_id * bar_width, mean_similarity, yerr=[mean_similarity - lower_similarity, upper_similarity - mean_similarity],
                fmt="o", color=f"C{frac_id}",
                markersize=2, lw=1, alpha=0.9, markeredgewidth=0.5,
                ecolor=f"C{frac_id}", elinewidth=0.5, capsize=1, capthick=0.5, label=f"{int(frac * 100)}% subset",
            )

        ax.set_xlabel("Component ID", labelpad=1)
        ax.set_xticks([1 + total_bar_width / 2, n_sep + total_bar_width / 2, n_sep + 1 + total_bar_width / 2, 10 + total_bar_width / 2, 15 + total_bar_width / 2],
                      ["1", f"{n_sep}", f"{n_sep+1}", "10", "15"])

        ax.set_ylabel("Similarity to PCs\n of full dataset", color='black', labelpad=1)
        ax.set_yticks([0, 0.5, 1.0])
        ax.axvline(x=n_sep + 0.5 + total_bar_width / 2, color='red', lw=1, ls='--', alpha=0.8)
        # add_textonly_legend(ax, {f"{int(frac * 100)}% ": {"color": f"C{frac_id}"} 
        #                          for frac_id, frac in enumerate(subset_fracs)}, loc='best', 
        #                     fontsize=5, frameon=False, title="Subset Frac")

        ax.spines[['top', 'right']].set_visible(False)

        save_path = path.join(get_saving_path(), task_name, 
                            f"Stability_{variant}_SVD_{n_components}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=500, transparent=True)
        plt.close(fig)
        logger.info("Plot saved to " + save_path)

    stability_summary()