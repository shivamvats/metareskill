from itertools import cycle
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np


def reps_solve_info_analysis(input_path, verbose=False):

    path_to_reps_info = Path(input_path)
    assert (
        path_to_reps_info.exists()
    ), f'Expected input_path "{path_to_reps_info}" to exist, but it does not.'

    with open(path_to_reps_info, "rb") as f:
        reps_solve_info = pickle.load(f)

    # TODO: change to dict-like
    assert isinstance(
        reps_solve_info, dict
    ), f"Expected reps_solve_info to be a dict, but it is a {type(reps_solve_info)}."

    reps_converged = reps_solve_info["reps_converged"]
    policy_params_mean = reps_solve_info["policy_params_mean"]
    mean_param_hist = reps_solve_info["history"]["policy_params_mean"]
    var_diag_param_hist = reps_solve_info["history"]["policy_params_var_diag"]
    mean_rew_hist = reps_solve_info["history"]["mean_reward"]

    num_params = len(reps_solve_info["policy_params_mean"])
    num_reps_attempts = reps_solve_info["num_reps_attempts"]

    # this might be a ragged array, so we flatten it
    assert len(mean_rew_hist) == num_reps_attempts
    mean_rew_hist_all_attempts = np.hstack(
        [mean_rew_hist[a] for a in range(num_reps_attempts)]
    )
    iter_param_updates = range(len(mean_rew_hist_all_attempts))

    if verbose:
        print(f'REPS solve info for "{path_to_reps_info}":')
        print(f" -> Solved: {reps_converged}")
        print(f" -> Parameters (mean): {policy_params_mean}")

    num_subplots = num_params + 1
    fig, ax = plt.subplots(num_subplots, 1, sharex=True)
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = cycle(prop_cycle.by_key()["color"])
    for p in range(num_subplots):

        if p == 0:
            # show reward
            ax[p].plot(
                iter_param_updates,
                mean_rew_hist_all_attempts,
                ".-",
                color=next(colors),
            )
            ax[p].grid()
            ax[p].set_ylabel(f"Reward")

        else:
            idx_p = p - 1
            # this might be a ragged array, so we flatten it
            assert len(mean_param_hist) == num_reps_attempts
            mean_param_hist_all_attempts = np.hstack(
                [
                    np.array(mean_param_hist[a])[:, idx_p]
                    for a in range(num_reps_attempts)
                ]
            )
            # assert len(mean_param_hist_all_attempts) == (num_reps_param_updates + 1)

            assert len(var_diag_param_hist) == num_reps_attempts
            var_diag_param_hist_all_attempts = np.hstack(
                [
                    np.array(var_diag_param_hist[a])[:, idx_p]
                    for a in range(num_reps_attempts)
                ]
            )

            stdev_diag_param_hist_all_attempts = np.sqrt(
                var_diag_param_hist_all_attempts
            )

            assert len(mean_param_hist_all_attempts) == len(
                stdev_diag_param_hist_all_attempts
            )
            assert len(mean_param_hist_all_attempts) == len(iter_param_updates)

            mean_p_stdev = (
                mean_param_hist_all_attempts + stdev_diag_param_hist_all_attempts
            )
            mean_m_stdev = (
                mean_param_hist_all_attempts - stdev_diag_param_hist_all_attempts
            )

            this_color = next(colors)

            ax[p].plot(
                iter_param_updates,
                mean_param_hist_all_attempts,
                ".-",
                color=this_color,
            )
            ax[p].fill_between(
                iter_param_updates,
                mean_p_stdev,
                mean_m_stdev,
                alpha=0.25,
                color=this_color,
            )
            ax[p].grid()
            ax[p].set_ylabel(f"Parameter {idx_p}")

        if p == (num_subplots - 1):
            ax[p].set_xlabel("parameter update iteration")

    plt.xlim((iter_param_updates[0], iter_param_updates[-1]))
    plt.show()

    return fig, ax
