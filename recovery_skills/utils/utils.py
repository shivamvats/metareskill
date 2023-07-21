import copy
import glob
import logging
import os
from os.path import join, isfile
import time

from hydra.utils import to_absolute_path
from icecream import ic
import numpy as np
from omegaconf import OmegaConf
import pickle as pkl
from rl_utils import Reps
from stable_baselines3.common.callbacks import BaseCallback
import wandb


logger = logging.getLogger(__name__)
logger.setLevel('INFO')
# logger.setLevel('DEBUG')


KNOWN_ENV_CONVERGENCE_CRITERIA = [
    "policy_var_diag",
    "env_solved",
    "mean_reward",
    "min_num_total_updates",
]


class WandbCallback(BaseCallback):
    """
    Callback for saving a model every ``save_freq`` calls
    to ``env.step()``.
    .. warning::
      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``
    :param save_freq:
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param verbose:
    """

    def __init__(
        self,
        save_freq,
        save_path,
        eval_callback=None,
        base_path="",
        name_prefix="rl_model",
        verbose=0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = join(base_path, save_path)
        self.eval_callback = eval_callback
        # self.save_path = save_path
        self.base_path = base_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # log training progress

        # log model
        if self.n_calls % self.save_freq == 0:
            path = join(
                self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.ckpt"
            )
            self.model.save(path)
            wandb.save(path, self.base_path)
            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")

        if self.n_calls % self.eval_callback.eval_freq == 0:
            wandb.log(
                {
                    "eval/best_mean_reward": self.eval_callback.best_mean_reward,
                    "eval/last_mean_reward": self.eval_callback.last_mean_reward,
                },
            )
        return True


def solve_env_using_reps(
    env,
    policy,
    policy_params_mean_init,
    policy_params_var_init,
    num_policy_rollouts_before_reps_update,
    max_reps_param_updates,
    env_convergence_criteria,
    start_state=None,
    reps_hyperparams={},
    max_num_reps_attempts=1,
    enable_sum_rewards_over_rollout=True,
    debug_info=False,
    verbose=False,
    render=False,
    plot=True,
):

    run_id = time.time()

    # Input argument handling
    assert isinstance(
        env_convergence_criteria, dict
    ), "Expected env_convergence_criteria to be a dict, but it is a {}.".format(
        type(env_convergence_criteria)
    )

    for criterion in env_convergence_criteria.keys():
        assert (
            criterion in KNOWN_ENV_CONVERGENCE_CRITERIA
        ), 'env_convergence_criteria type "{}" is not recognized.'.format(criterion)

    check_policy_var_diag = (
        True if "policy_var_diag" in env_convergence_criteria else False
    )
    policy_var_diag_thresh = (
        env_convergence_criteria["policy_var_diag"] if check_policy_var_diag else np.inf
    )
    check_env_solved = True if "env_solved" in env_convergence_criteria else False
    env_solved_thresh = (
        env_convergence_criteria["env_solved"] if check_env_solved else 0.0
    )
    check_mean_reward = True if "mean_reward" in env_convergence_criteria else False
    mean_reward_thresh = (
        env_convergence_criteria["mean_reward"] if check_mean_reward else -np.inf
    )
    check_min_num_total_updates = (
        True if "min_num_total_updates" in env_convergence_criteria else False
    )
    min_num_total_updates = (
        env_convergence_criteria["min_num_total_updates"]
        if check_min_num_total_updates
        else -np.inf
    )

    assert (
        isinstance(max_num_reps_attempts, int) and max_num_reps_attempts > 0
    ), "Expected max_num_reps_attempts to be a positive integer, but it is not."

    # Set up REPS
    reps_hyperparams_to_use = {
        "rel_entropy_bound": 0.5,
        "min_temperature": 0.00001,
    }
    reps_hyperparams_to_use.update(reps_hyperparams)
    reps = Reps(**reps_hyperparams_to_use)

    reps_converged = False

    policy_params_mean = copy.deepcopy(policy_params_mean_init)
    policy_params_var = copy.deepcopy(policy_params_var_init)

    num_reps_attempts = 0
    num_reps_param_total_updates = 0

    num_policy_rollouts_attempts = []
    num_reps_param_updates_attempts = []
    policy_params_mean_attempts = []
    policy_params_var_attempts = []
    mean_reward_obtained_attempts = []
    solved_frac_obtained_attempts = []

    policy_params = {'mean': policy_params_mean,
              'cov': policy_params_var}
    policy.update_policy(policy_params)

    while not reps_converged and num_reps_attempts < max_num_reps_attempts:

        num_policy_rollouts = 0
        num_reps_param_updates = 0
        policy_params_for_reps = []
        rewards_for_reps = []
        policy_solved_env = []
        policy_params_mean_this_attempt = []
        policy_params_var_this_attempt = []
        mean_reward_obtained_this_attempt = []
        solved_frac_obtained_this_attempt = []

        policy_params_mean_init_this_attempt = copy.deepcopy(policy_params_mean)
        policy_params_var_init_this_attempt = copy.deepcopy(policy_params_var)

        if verbose:
            print(
                "REPS attempt {} of {}: ".format(
                    num_reps_attempts + 1, max_num_reps_attempts
                )
            )
            print("policy_params_mean_init_this_attempt:")
            print(policy_params_mean_init_this_attempt)
            print("policy_params_var_init_this_attempt:")
            print(policy_params_var_init_this_attempt)
            print()

        policy_params_mean_this_attempt.append(
            policy_params_mean_init_this_attempt.tolist()
        )
        if debug_info:
            init_var = policy_params_var_init_this_attempt.tolist()
        else:
            init_var = np.diag(policy_params_var_init_this_attempt).tolist()
        policy_params_var_this_attempt.append(init_var)

        while not reps_converged and num_reps_param_updates <= max_reps_param_updates:

            if verbose:
                print(
                    f"Number of REPS param updates: {num_reps_param_updates} (current attempt); {num_reps_param_total_updates} (total over all attempts)"
                )

            # Reset env
            if hasattr(env, "num_envs"): #and env.num_envs > 1:
                # VecEnv
                if start_state:
                    observations = env.reset_from_state(start_state)
                else:
                    observations = env.reset()

                num_env_steps = 0
                # __import__('ipdb').set_trace()
                # Just works for one-step envs for now
                obsv_envs, reward_envs, is_done_envs, info_envs = policy.apply(
                    env, observations, render=render, local_frame=True)
                # l2 regularization
                regularization = True
                if regularization:
                    params = np.array([info['policy_params'] for info in info_envs])
                    params = params.reshape(-1, 3, 7)
                    # xyz_params = params[:, :, [0, 1, 2]]
                    pose_params = params[:, :, :-1]
                    pose_params = pose_params.reshape(pose_params.shape[0], -1)
                    pose_norms = np.linalg.norm(pose_params, axis=1)
                    reward_envs -= 10.0*pose_norms

                policy_params_envs = [info['policy_params'] for info in
                                      info_envs]

                # assert np.all(is_done_envs)

                envs_are_solved = np.array(
                    [info_envs[e]["is_solved"] for e in range(env.num_envs)]
                )

                policy_params_for_reps.extend(policy_params_envs)
                rewards_for_reps.extend(reward_envs)
                policy_solved_env.extend(envs_are_solved)
                num_policy_rollouts += env.num_envs

            else:
                observation = env.reset()
                if start_state:
                    observation = env.reset_from_state(start_state)

                if render:
                    for _ in range(50):
                        env.render()

                # Sample policy parameters
                # TODO Use reps apply
                policy_params = np.random.multivariate_normal(
                    mean=policy_params_mean, cov=policy_params_var
                )

                logger.debug(f"Sampled params: {policy_params.round(3)}")

                # Rollout policy
                env_is_done = False
                env_is_solved = False
                num_env_steps = 0
                env_reward_for_reps = 0.0

                while not env_is_done and num_env_steps < policy.num_steps:

                    # Calculate new action based on policy here
                    # action = policy.action_from_state(num_env_steps, observation, env.context())
                    observation, this_step_reward, env_is_done, info = policy.apply(
                        env, obs=observation, context=env.context(),
                        render=render, local_frame=True
                    )
                    # this_step_reward += policy.reward_from_state(num_env_steps, observation, env.context())
                    logger.debug(f"    reward: {this_step_reward}")
                    # logger.debug(f"    done: {env_is_done}")

                    if enable_sum_rewards_over_rollout:
                        env_reward_for_reps += this_step_reward
                    else:
                        env_reward_for_reps = this_step_reward

                    env_is_solved = info["is_solved"]
                    num_env_steps += policy.num_steps

                # Add to REPS buffers
                policy_params_for_reps.append(policy_params.tolist())
                rewards_for_reps.append(env_reward_for_reps)
                policy_solved_env.append(env_is_solved)
                num_policy_rollouts += 1

            num_policy_rollouts_this_batch = len(policy_params_for_reps)

            if verbose:
                print(
                    f"Number of policy samples obtained so far this batch: {num_policy_rollouts_this_batch} out of {num_policy_rollouts_before_reps_update} needed."
                )
                print()

            if num_policy_rollouts_this_batch >= num_policy_rollouts_before_reps_update:

                if verbose:
                    print("Batch size reached.")

                n_times_solved = np.sum(policy_solved_env)
                env_solved_frac = n_times_solved / num_policy_rollouts_this_batch

                # Check if we've converged - do not update if so
                policy_var_diag_under_thresh = np.all(
                    np.diag(policy_params_var) <= policy_var_diag_thresh
                )
                env_solved_over_thresh = env_solved_frac >= env_solved_thresh
                mean_reward_for_policy = np.mean(rewards_for_reps)
                mean_reward_over_thresh = mean_reward_for_policy >= mean_reward_thresh
                min_num_total_updates_over_thresh = (
                    num_reps_param_total_updates >= min_num_total_updates
                )

                if verbose:
                    print(
                        "Rewards for this policy: {} +- {} (1 stdev, n={})".format(
                            mean_reward_for_policy,
                            np.std(rewards_for_reps),
                            len(rewards_for_reps),
                        )
                    )
                    print(
                        "Solved success rate: {} ({}/{})".format(
                            env_solved_frac,
                            n_times_solved,
                            num_policy_rollouts_this_batch,
                        )
                    )
                    logger.info(
                        "Solved success rate: {} ({}/{})".format(
                            env_solved_frac,
                            n_times_solved,
                            num_policy_rollouts_this_batch,
                        )
                    )

                    if check_policy_var_diag and policy_var_diag_under_thresh:
                        print(" -> Policy parameter diagonal variance has converged.")

                    if check_env_solved and env_solved_over_thresh:
                        print(" -> Environment solved fraction has converged.")

                    if check_mean_reward and mean_reward_over_thresh:
                        print(" -> Mean reward for policy has converged.")

                    if (
                        check_min_num_total_updates
                        and min_num_total_updates_over_thresh
                    ):
                        print(f" -> Total number of parameter updates has converged.")

                    print("")

                mean_reward_obtained_this_attempt.append(mean_reward_for_policy)
                solved_frac_obtained_this_attempt.append(env_solved_frac)

                if plot:
                    import matplotlib.pyplot as plt

                    plt.clf()
                    plt.plot(np.array(mean_reward_obtained_this_attempt).flatten())
                    plt.xlabel("Updates")
                    plt.ylabel("Mean Reward")
                    plt.savefig(f"rewards_{run_id}.png")

                # XXX Hack
                # if mean_reward_for_policy < -500:
                    # logger.warning(f"  Reward of {mean_reward_for_policy} too low. Early Breaking!")
                    # raise ValueError

                if (
                    policy_var_diag_under_thresh
                    and env_solved_over_thresh
                    and mean_reward_over_thresh
                    and min_num_total_updates_over_thresh
                ):
                    reps_converged = True

                    if verbose:
                        logger.info(f"REPS has converged with success rate: {env_solved_frac}")

                else:
                    if verbose:
                        print("REPS has not yet converged. Updating policy...")

                    # Run the REPS update
                    (
                        policy_params_mean,
                        policy_params_var,
                        reps_info,
                    ) = reps.policy_from_samples_and_rewards(
                        policy_params_for_reps, rewards_for_reps
                    )
                    policy_params_var_diag = np.diag(policy_params_var)
                    policy_params = {'mean': policy_params_mean,
                                     'cov': policy_params_var}
                    policy.update_policy(policy_params, env.context())
                    pkl.dump(policy, open("latest_reps_policy.pkl", "wb"))

                    # Debug
                    if verbose:
                        print("New policy param mean:")
                        print(policy_params_mean)

                        print("New policy param var diag:")
                        print(policy_params_var_diag)
                        print("")

                    policy_params_mean_this_attempt.append(policy_params_mean.tolist())
                    if debug_info:
                        var_to_keep = policy_params_var.tolist()
                    else:
                        var_to_keep = np.diag(policy_params_var).tolist()
                    policy_params_var_this_attempt.append(var_to_keep)

                    # Reset buffers
                    policy_params_for_reps = []
                    rewards_for_reps = []
                    policy_solved_env = []
                    num_reps_param_updates += 1
                    num_reps_param_total_updates += 1

        num_policy_rollouts_attempts.append(num_policy_rollouts)
        num_reps_param_updates_attempts.append(num_reps_param_updates)
        policy_params_mean_attempts.append(policy_params_mean_this_attempt)
        policy_params_var_attempts.append(policy_params_var_this_attempt)
        mean_reward_obtained_attempts.append(mean_reward_obtained_this_attempt)
        solved_frac_obtained_attempts.append(solved_frac_obtained_this_attempt)

        num_reps_attempts += 1
        if not reps_converged and num_reps_attempts < max_num_reps_attempts:
            # Seed the next attempt with this result
            # Keep the same mean and divide the covariance
            policy_params_var = policy_params_var_init_this_attempt / 2.0

    # We keep lists as lists and don't convert to np arrays
    # The dimensions don't always agree depending on how many updates each attempt takes

    if debug_info:
        policy_params_var_diag_attempts = [
            [np.diag(var) for var in param_var_attempt]
            for param_var_attempt in policy_params_var_attempts
        ]
    else:
        policy_params_var_diag_attempts = policy_params_var_attempts

    solve_env_info = {
        "reps_converged": reps_converged,
        "policy_params_mean": policy_params_mean,
        "policy_params_var": policy_params_var,
        "env_convergence_criteria": env_convergence_criteria,
        "max_num_reps_attempts": max_num_reps_attempts,
        "num_reps_attempts": num_reps_attempts,
        "num_policy_rollouts": num_policy_rollouts_attempts,
        "num_reps_param_updates": num_reps_param_updates_attempts,
        "history": {
            "policy_params_mean": policy_params_mean_attempts,
            "policy_params_var_diag": policy_params_var_diag_attempts,
            "mean_reward": mean_reward_obtained_attempts,
            "solved_frac": solved_frac_obtained_attempts,
        },
    }

    if debug_info:
        solve_env_info["history"]["policy_params_var"] = policy_params_var_attempts

    return reps_converged, policy_params_mean, policy_params_var, solve_env_info


def load_reps_params(path_to_file):
    with open(path_to_file, "rb") as f:
        _, params_mean, _, _ = pkl.load(f)
        return params_mean


def get_rand_in_range(a, b, shape=None):
    if shape is None:
        return a + (b - a) * np.random.rand()
    else:

        return a + (b - a) * np.random.rand(*shape)


def load_pillar_states(path_to_file):
    import pillar_state as ps
    serialized_strs = pkl.load(open(path_to_file, "rb"))
    states = []
    for str in serialized_strs:
        state = ps.State.create_from_serialized_string(str)
        states.append(state)
    return states


def load_demos(path_to_demo_dir):
    """
    Loads all demos and corresponding results saved during recording from a
    directory.
    """
    assert os.path.exists(path_to_demo_dir), f"Demo dir {path_to_demo_dir} does not exist"

    skill_chain_files = glob.glob(join(path_to_demo_dir, 'skill_chain_*.pkl'))
    demo_results_files = glob.glob(join(path_to_demo_dir, '1_demo_results_*.pkl'))

    skill_chains, demo_results = [], []

    for chain_file, results_file in zip(skill_chain_files, demo_results_files):
        skill_chain = pkl.load(open(chain_file, 'rb'))
        results = pkl.load(open(results_file, 'rb'))
        skill_chains.append(skill_chain)
        demo_results.append(results)

    return skill_chains, demo_results


def load_skill(path_to_skill):
    assert os.path.exists(path_to_skill), f"Skill dir {path_to_skill} does not exist"

    cfg = OmegaConf.load(join(path_to_skill, 'config.yaml'))
    # load policy
    if cfg.type == 'IdentityEESpaceSkill':
        from recovery_skills.skills.identity_ee_space_skill import IdentityEESpaceSkill

        try:
            skill = IdentityEESpaceSkill(path_to_policy=join(path_to_skill,
                                                            'policy.pkl'))
        except:
            skill = pkl.load(open(join(path_to_skill, 'skill.pkl'), 'rb'))
    else:
        raise NotImplementedError

    # load preconditions
    # TODO

    return skill


def pkl_load(filename, hydra=False):
    if hydra:
        filename = to_absolute_path(filename)
    return pkl.load(open(filename, 'rb'))


def pkl_dump(object, filename, hydra=False):
    if hydra:
        filename = to_absolute_path(filename)
    return pkl.dump(object, open(filename, 'wb'))


def is_vec_env(env):
    if hasattr(env, 'num_envs'):
        return True
    else:
        return False


def load_subgoals(subgoals_dir, ground_truth=True):
    root_dir = to_absolute_path(subgoals_dir)

    def load_subgoals_from_dir(path_to_dir):
        if ground_truth:
            subgoals = pkl_load(join(path_to_dir, "all_gt_subgoals.pkl"))
        else:
            subgoals = pkl_load(join(path_to_dir, "all_subgoals.pkl"))
        labels = pkl_load(join(path_to_dir, "all_labels.pkl"))

        return subgoals, labels

    # first check if init_sets file exits
    if isfile(join(root_dir, 'all_subgoals.pkl')):
        subgoals, labels = load_subgoals_from_dir(root_dir)
    else:
        # assume subdirs exist
        subgoal_dirs = glob.glob(
            join(
                root_dir, "*/",
            )
        )
        subgoals, labels = None, None
        for subgoal_dir in subgoal_dirs:
            _subgoals, _labels = load_subgoals_from_dir(subgoal_dir)
            nsubgoals = len(_subgoals)

            if subgoals is None:
                subgoals  = [[] for _ in range(nsubgoals)]
                labels = []

            for i in range(nsubgoals):
                subgoals[i].extend(_subgoals[i])
            labels.extend(_labels)
        __import__('ipdb').set_trace()

    return subgoals, labels
