from os.path import join
import numpy as np

import hydra

from recovery_skills.utils import *
from recovery_skills.recovery.value_mab_planner import *


@hydra.main(config_path="../cfg", config_name="evaluate_recovery_skills")
def main(cfg):
    strategy =  'iter-mono-mab'
    path_to_dir = 'data/door_opening/debug/recovery_skills/test'
    path_to_subdir = join(path_to_dir, strategy)
    p_low = pkl_load(cfg.path_to_transition_low, True)
    # failure_clusters = pkl_load(cfg.path_to_failure_clusters, True)
    nclusters = 6
    nsubgoals = 4
    train_tasks = pkl_load(join(
        cfg.path_to_tasks_root, f"train_tasks_0.pkl"), True)
    cluster_sizes = np.array([len(cluster[0]) for cluster in train_tasks])
    ntotal_failures = sum(cluster_sizes)
    cluster_ids = np.arange(nclusters)
    cluster_weights = cluster_sizes / ntotal_failures

    print(strategy)
    print("===========")
    fail_vals = []
    for i in range(5):
    # for i in [1]:
        print(f"{i}:")
        print("----------")
        path_to_skills = join(path_to_subdir, f"{i}", "knn_skills.pkl")
        recovery_skills = pkl_load(path_to_skills, True)
        skill_accs = []
        for cluster in recovery_skills:
            accs = []
            for skill in cluster:
                if skill:
                    accs.append(skill['accuracy'])
                else:
                    accs.append(0.0)
            skill_accs.append(np.array(accs))
        skill_accs = np.array(skill_accs)
        if cfg.path_to_hl_planner:
            planner = pkl_load(cfg.path_to_hl_planner, True)
        else:
            planner = MonoValueMABPlanner(nclusters, nsubgoals, cluster_weights, p_low)
        planner.update_transition_matrix(skill_accs)
        V = planner.compute_value_function()
        failure_value = planner.compute_failure_value(V)
        # print(f"  Failure value: {failure_value}")
        fail_vals.append(failure_value)
        policy = planner.policy
        state_ids = [planner.get_state_id(i, cluster=True) for i in range(nclusters)]
        hl_policy = np.array(policy[state_ids]) - 1
        hl_policy_succ = [skill_accs[i, hl_policy[i]] for i in range(nclusters)]
        # print(f"High level Policy: {hl_policy}")
        # print(f"High level Policy Success: {np.round(hl_policy_succ, 2)}")

    print("-------------------")
    print(f"  Value: {np.mean(fail_vals)}")
    print("-------------------")
    print("\n")

if __name__ == "__main__":
    main()
