from collections import defaultdict, Counter
import numpy as np
import logging
from dataclasses import dataclass

from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_ClusteringAnalysis import Visualize_ClusteringAnalysis
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_FeatureSpace import get_weight_tuple_Decomposition_Weights, get_weight_tuple_PSTH, get_weight_tuple_Physiology_Fingerprint, get_weight_tuple_Variability_Physiology, get_weight_tuple_Waveform, simple_Landscope_feature_space
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_SPVtypes import CellFeature, clustering_lineage, clustering_quality_test, clustering_stability_test, consensus_grid_search, eigen_gap_quality_test, feature_contribution_analysis, grand_co_clustering_analysis
from kitchen.utils.sequence_kit import find_only_one



logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# handler = logging.FileHandler(path.join(path.dirname(__file__), "overview.log"), mode="w")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s:\n%(message)s\n')
handler.setFormatter(formatter)
logger.addHandler(handler)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)


def tuple2featurelist(feature_tuple) -> list[CellFeature]:
    cf_list = []
    for features, node_name, cohort_name in zip(feature_tuple[0], feature_tuple[1], feature_tuple[2]):
        cf_list.append(CellFeature(features=features, node_name=node_name, cohort_name=cohort_name))
    print(f"Total cells: {len(cf_list)}")
    print(Counter([cf.cohort_name for cf in cf_list]))
    return cf_list


def combine_feature_lists(*feature_lists):
    shared_node_names = set.intersection(*(set(cf.node_name for cf in fl) for fl in feature_lists))
    combined_fl = []
    for single_shared_node_name in shared_node_names:
        combined_feature_vec, combined_cohort_name = [], []
        for fl in feature_lists:
            corresponding_cf = find_only_one(fl, node_name=single_shared_node_name)
            combined_feature_vec.append(corresponding_cf.features)
            combined_cohort_name.append(corresponding_cf.cohort_name)
        assert len(set(combined_cohort_name)) == 1, f"Inconsistent cohort names for node {single_shared_node_name}"
        combined_fl.append(CellFeature(features=np.concatenate(combined_feature_vec),
                                       node_name=single_shared_node_name, cohort_name=combined_cohort_name[0]))
        assert np.all(np.isfinite(combined_fl[-1].features)), f"Non-finite values found in features for node {single_shared_node_name}"

    combined_fl = sorted(combined_fl, key=lambda cf: cf.node_name)
    return combined_fl
        
           
# def old_main_on_PSTH():    
#     # PSTH_tuple = get_weight_tuple_PSTH("raw", 5/1000)    
#     svd_5_tuple = get_weight_tuple_Decomposition_Weights("zscore", "SVD", 5, True, _visualize=False)
#     cluster_names, (n_clusters, embedding, normalized_weight_matrix) = get_putative_labels(svd_5_tuple)
#     for x, y,z in zip(svd_5_tuple[2], cluster_names, svd_5_tuple[1]):
#         print(x, y, z)
#     Visualize_ClusteringAnalysis(svd_5_tuple, PSTH_tuple, feature_space_name="SVD5")


def mainloop():
    # svd_flag = 5  # or 6
    # spk_feature_flag = True  # or False
    # acg_flag = False  # or True, spk feature has to be True

    # for svd_flag in [5, 6]:
    #     for spk_feature_flag in [True, False]:
    #         for acg_flag in [False, True]:  # spk feature has to be True
    #             if not spk_feature_flag and acg_flag:
    #                 continue
    #             main(svd_flag, spk_feature_flag, acg_flag)
    
    # main(svd_flag=5, spk_feature_flag=True, acg_flag=False)
    main(svd_flag=5, spk_feature_flag=False, acg_flag=False)

def main(svd_flag, spk_feature_flag, acg_flag):
    task_name = f"zscore_SVD{svd_flag}_bin{10 if svd_flag == 5 else 5}ms_{'include' if spk_feature_flag else 'exclude'}SpkFeature_{'include' if acg_flag else 'exclude'}ACG"

    
    svd_tuple = get_weight_tuple_Decomposition_Weights("zscore", "SVD", n_components=svd_flag, 
                                                         append_baseline=not spk_feature_flag, 
                                                         BINSIZE=10/1000 if svd_flag == 5 else 5/1000,
                                                         _visualize=False)
    print(f"SVD {svd_flag} Feature List")
    svd_cf_list = tuple2featurelist(svd_tuple)
    if spk_feature_flag:
        ephys_tuple = get_weight_tuple_Physiology_Fingerprint(with_spont_FR=True, _drop_ACG_rise_time=not acg_flag)
        print("Ephys Feature List")
        ephys_cf_list = tuple2featurelist(ephys_tuple)           
        combined_cf_list = combine_feature_lists(svd_cf_list, ephys_cf_list)
    else:
        combined_cf_list = svd_cf_list

    
    print(f"After combined, total cells: {len(combined_cf_list)}")
    print(Counter([cf.cohort_name for cf in combined_cf_list]))

    n_feature = len(combined_cf_list[0].features)
    
    feature_names = [
        [f"PC{i+1}" for i in range(svd_flag)],
    ]
    if spk_feature_flag:
        feature_names.append([
            "Spont. FR",
            "Spike Width",
            "Spike Asym.",
            "CV2",
        ])
        if acg_flag:
            feature_names[-1].append("ACG Rise tau")
    else:
        feature_names = [["Spont. FR",],] + feature_names

    def p1():
        eigen_gap_quality_test(combined_cf_list, task_id=task_name)
        # for weight_alpha in [2, 5, 10, 30, 50]:
        # for weight_alpha in [5, 10, 30, 50]:
        for weight_alpha in [50, ]:
            # for n_cluster in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
            #     consensus_grid_search(combined_cf_list, 
            #                           task_id=task_name, n_features=n_feature,
            #                           n_weight=100,
            #                           num_of_cluster=n_cluster, weight_alpha=weight_alpha,
            #                             _visualize=True)

            # clustering_lineage(combined_cf_list, task_id=task_name, weight_alpha=weight_alpha, n_features=n_feature)
            clustering_lineage(combined_cf_list, task_id=task_name, weight_alpha=weight_alpha, n_features=n_feature, 
                               top_k=7, figsize=(1.9, 1.2))
            clustering_quality_test(combined_cf_list, task_id=task_name, weight_alpha=weight_alpha, n_features=n_feature)

    def p2():
        grand_co_clustering_analysis(combined_cf_list, weight_alpha=50, task_id=task_name, feature_names=feature_names, 
                                     n_features=n_feature)

    def p3():
        for weight_alpha in [50,]:
            for n_cluster in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
                for feature_drop_id in range(n_feature):
                    consensus_grid_search(combined_cf_list, 
                                          task_id=task_name, n_features=n_feature,
                                            num_of_cluster=n_cluster, weight_alpha=weight_alpha,
                                            n_weight=100,
                                            _visualize=True, _feature_drop_id=feature_drop_id)
        feature_contribution_analysis(combined_cf_list, task_id=task_name, n_feature=n_feature, weight_alpha=50, feature_names=feature_names)

    def p4():
        for resample_rate in [0.9, 0.75, 0.5]:
            for resample_seed in range(100):
                for n_cluster in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
                    consensus_grid_search(combined_cf_list, 
                                          task_id=task_name,
                                          n_features=n_feature,
                                          num_of_cluster=n_cluster, weight_alpha=50,
                                            n_weight=10,
                                           _visualize=False,
                                          _resample_tuple=(resample_rate, resample_seed))

        clustering_stability_test(combined_cf_list, weight_alpha=50, task_id=task_name, n_feature=n_feature, num_fcluster=6)


    # p1()
    p2()
    # p3()
    # p4()


if __name__ == "__main__":
    # old_main_on_PSTH()
    mainloop()