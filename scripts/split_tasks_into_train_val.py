import hydra
from recovery_skills.utils.data_processing import *
from recovery_skills.skills.utils import *


@hydra.main(config_path="../cfg", config_name="learn_recovery_skills")
def main(cfg):
    if cfg.seed is None:
        seed = int(time.time())
    else:
        seed = cfg.seed

    failure_clusters = pkl_load(cfg.path_to_failure_clusters, True)
    skill_chain = load_nominal_skill_chain(cfg)
    subgoals = pkl_load(cfg.path_to_preconds, True)
    tasks = failures_to_tasks(failure_clusters, skill_chain, subgoals)
    nclusters = len(failure_clusters)
    nsubgoals = len(subgoals)
    cluster_ids = np.arange(nclusters)

    N = 6

    for i in range(N):
        tasks_train, tasks_val = split_into_train_and_val_tasks(tasks,
                                                                cluster_ids,
                                                                nclusters,
                                                                nsubgoals)
        pkl_dump(tasks_train, f"train_tasks_{i}.pkl")
        pkl_dump(tasks_val, f"val_tasks_{i}.pkl")


if __name__ == "__main__":
    main()
