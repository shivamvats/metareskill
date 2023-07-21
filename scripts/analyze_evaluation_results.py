"""Analyze and compare the results of evaluate_recovery_skills.py scripts"""

from os.path import join

import hydra
from hydra.utils import to_absolute_path
from recovery_skills.utils import *


@hydra.main(config_path="../cfg", config_name="evaluate_recovery_skills")
def main(cfg):
    exp_dates = ["2022-09-06", "2022-09-06"]
    exp_times = ["12-49-39", "12-49-41"]
    exp_labels = ["rr", "iter-mab"]
    infos = []
    for date, time in zip(exp_dates, exp_times):
        path = join("outputs", date, time, 'info.pkl')
        infos.append(pkl_load(path, True))

    print("Collected stats:")
    print(infos[0].keys())
    from IPython import embed; embed();

if __name__ == "__main__":
    main()
