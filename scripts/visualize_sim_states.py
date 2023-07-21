from copy import deepcopy
import logging
from os.path import join
import time

import hydra
from hydra.utils import to_absolute_path
from icecream import ic
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import pickle as pkl

from stat_utils import sample_from_distrib as sample
from robosuite.wrappers import VisualizationWrapper
from recovery_skills.envs.franka_door_env import FrankaDoorEnv
from recovery_skills.envs.franka_door_subtask_env import FrankaDoorSubtaskEnv
from recovery_skills.envs.improved_gym_wrapper import ImprovedGymWrapper
from recovery_skills.graph import State
from recovery_skills.utils import *
import robosuite.utils.transform_utils as T

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


@hydra.main(config_path="../cfg", config_name="visualize_sim_states.yaml")
def main(cfg):
    if cfg.seed is None:
        seed = int(time.time())
    else:
        seed = cfg.seed

    controller_cfg = OmegaConf.to_container(cfg.env.controller)
    env_cfg = cfg.env

    # dummy
    # goal_state = State(FrankaDoorEnv.state_vars,
                       # FrankaDoorEnv.state_var_ndims).from_array([0])

    # goal_constraint = GoalConstraint(goal_state)

    env = FrankaDoorSubtaskEnv(
            goal_constraint=None,
            controller_configs=controller_cfg,
            has_renderer=cfg.render,
            horizon=env_cfg.horizon,
            context_cfg=env_cfg.context,
            obs_uncertainty=env_cfg.obs_uncertainty,
            timestep=env_cfg.timestep,
            eef_start_region_cfg=env_cfg.eef_start_region,
            )
    env.set_obs_corruption(False)

    path_to_file = to_absolute_path(join(cfg.path_to_dir, cfg.filename))
    data = pkl.load(open(path_to_file, 'rb'))

    if cfg.view_term_states:
        logger.info("Visualizing skill terminal states")
        nskills = len(data[0])
        term_states = [[] for _ in range(nskills)]
        for datum in data:
            for i in range(nskills):
                term_states[i].append(datum[i])

    elif cfg.view_subgoals:
        opts = cfg['subgoals']
        all_subgoals = pkl_load(opts.subgoal_file, True)
        labels = np.where(pkl_load(opts.label_file, True))

        logger.info("Visualizing task subgoals")
        subgoal_id = 1

        subgoals = np.array(all_subgoals[subgoal_id])

        if opts.viz_pos:
            subgoals = subgoals[labels]
            logger.info("Visualizing positive subgoals")
        elif opts.viz_neg:
            subgoals = subgoals[np.logical_not(labels)]
            logger.info("Visualizing negative subgoals")
        else:
            pass

        for state in subgoals:
            env.reset_from_state(state)
            for _ in range(50):
                env.render()

    elif cfg.view_skill_failures:
        logger.info("Visualizing failures")
        skill_id = 2
        all_failures = data
        failures = all_failures[skill_id]
        for i, failure in enumerate(failures):
            obs = failure['obs']
            env.reset_from_state(obs)
            logger.info("Start")
            env.render_by(25)

    elif cfg.view_task_failures:
        logger.info("Visualizing task failures")
        task_failures = data['tasks']
        for failure in task_failures:
            start, skill_chain = failure
            env.reset()
            obs = env.reset_from_state(start)
            for skill in skill_chain:
                env.env.goal_constraint = skill.goal
                obs, rew, done, info = skill.apply(env, obs, None, render=cfg.render)
                success = info['is_solved']
                logger.info(f"Skill success: {success}")
            task_success = env.check_task_success()
            logger.info("-------")
            logger.info(f"Task success: {task_success}")
            logger.info("-------")

    elif cfg.view_learnt_subgoals:
        train_skill_chains, _ = load_demos(to_absolute_path('./data/door_opening/debug/demos_1'))
        skill_chains, _ = load_demos(to_absolute_path('./data/door_opening/debug/demos_2'))
        reg = data
        nskills= len(skill_chains[0])

        # context of the task
        contexts = [[] for _ in range(nskills)]
        for chain in skill_chains:
            for i in range(nskills):
                contexts[i].append(State(FrankaDoorEnv.context_vars,
                                         FrankaDoorEnv.context_var_ndims).from_dict(
                                             chain[i].start_state.as_ordered_dict()).as_array())
        contexts = np.array(contexts)

        # subgoals
        subgoals =  [[] for  _ in range(nskills)]
        subgoal_arr =  [[] for  _ in range(nskills)]
        for chain in skill_chains:
            for i in range(nskills):
                subgoal_arr[i].append(chain[i].goal.goal.as_array())
                subgoals[i].append(chain[i].goal.goal)
        subgoal_arr = np.array(subgoal_arr)


        skill_id = 0
        X, y = contexts[skill_id], subgoal_arr[skill_id]
        preds = reg.predict(X)
        pred_states = [State(FrankaDoorEnv.state_vars,
                             FrankaDoorEnv.state_var_ndims).from_array(pred)
                       for pred in preds]
        __import__('ipdb').set_trace()

        for i, (subgoal, pred_state) in enumerate(zip(subgoals[skill_id], pred_states)):
            ic(i)
            ic("-----------")
            # viz actual subgoal
            env.reset()
            obs = env.reset_from_state(subgoal)
            ic("Ground truth subgoal")
            for _ in range(100):
                env.render()

            # viz predcted subgoal
            env.reset()
            obs = env.reset_from_state(pred_state)
            ic("Predicted subgoal")
            for _ in range(100):
                env.render()

    elif cfg.view_learnt_rl_subgoals:
        train_skill_chains, _ = load_demos(to_absolute_path('./data/door_opening/debug/demos_train'))
        skill_chains, _ = load_demos(to_absolute_path('./data/door_opening/debug/demos_test'))
        nskills = len(skill_chains[0])

        skill_id = 0
        env = env.env

        # train
        subgoals =  [[] for  _ in range(nskills)]
        subgoal_arr =  [[] for  _ in range(nskills)]
        for chain in train_skill_chains:
            for i in range(nskills):
                rl_goal = env.state_to_rl_state(chain[i].goal.goal)
                subgoals[i].append(rl_goal)
                subgoal_arr[i].append(rl_goal.as_array())

        subgoal_arr = np.array(subgoal_arr)

        learnt_goal_constraint = LearntGoalConstraint(subgoals[skill_id])

        # test
        subgoals =  [[] for  _ in range(nskills)]
        rl_subgoals =  [[] for  _ in range(nskills)]
        for chain in train_skill_chains:
            for i in range(nskills):
                subgoals[i].append(chain[i].goal.goal)
                rl_goal = env.state_to_rl_state(chain[i].goal.goal)
                rl_subgoals[i].append(rl_goal)

        viz_env = VisualizationWrapper(env, 'default')

        for i, (subgoal, rl_subgoal) in enumerate(zip(subgoals[skill_id],
                                                      rl_subgoals[skill_id])):
            ic(i)
            ic("-----------")
            # viz actual subgoal
            env.reset()
            obs = env.reset_from_state(subgoal)
            # ee_pos = rl_subgoal.as_ordered_dict()['robot_eef:pose/position']
            ee_pos = obs['robot_eef:pose/position']
            viz_env.set_indicator_pos('indicator0', ee_pos)
            env.sim.forward()
            ic("Ground truth subgoal")
            ic(ee_pos)
            for _ in range(100):
                env.render()

            # __import__('ipdb').set_trace()

            # viz predcted subgoal
            ic("Predicted subgoal")
            rl_goal = learnt_goal_constraint.goal(subgoal)
            # ee_pos = rl_goal.as_ordered_dict()['robot_eef:pose/position']
            fk_goal = env.state_to_rl_state(subgoal)
            ee_pos = fk_goal.as_ordered_dict()['robot_eef:pose/position']
            ic(ee_pos)
            viz_env.set_indicator_pos('indicator0', ee_pos)
            env.sim.forward()
            for _ in range(100):
                env.render()

    elif cfg.view_backchaining_states:
        states = pkl_load(cfg.backchaining.obs_file, True)
        labels = pkl_load(cfg.backchaining.label_file, True)

        def plot_states(states, ax=None, **kwargs):
            eef_poss = [state['robot_eef:pose/position'] for state in states]
            handle_poss = [state['handle:pose/position'] for state in states]
            eef_to_handle = np.array([eef - handle for eef, handle in zip(eef_poss, handle_poss)])
            if ax is None:
                ax = plt.axes(projection='3d')
            # ax.plot([-0.03, 0.07], [0.0, 0.0], [0.0, 0.0], color='k', linewidth=5)
            ax.plot([0], [0], [0], 'kx')
            scatter = ax.scatter(eef_to_handle[:, 0], eef_to_handle[:, 1], eef_to_handle[:, 2],
                    **kwargs)
            return scatter

        ax = plt.axes(projection='3d')
        ax.set_xlim3d(-0.25, 0.25)
        ax.set_ylim3d(-0.25, 0.25)
        ax.set_zlim3d(-0.25, 0.25)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        states = np.array(states)
        labels = np.array(labels).astype(bool)
        pos_states = states[labels]
        neg_states = states[np.logical_not(labels)]

        plot_states(pos_states, ax=ax, c='g', alpha=1)
        plot_states(neg_states, ax=ax, c='r', alpha=0.1)
        plt.tight_layout()
        plt.show()

    elif cfg.view_learnt_preconds:
        # env = VisualizationWrapper(
            # FrankaDoorEnv(
                # # goal_constraint=None,
                # controller_configs=controller_cfg,
                # has_renderer=cfg.render,
                # horizon=env_cfg.horizon,
                # context_cfg=env_cfg.context,
                # obs_uncertainty=env_cfg.obs_uncertainty,
                # timestep=env_cfg.timestep,
                # eef_start_region_cfg=env_cfg.eef_start_region,
            # ),
            # 'default')

        # sample a state
        # sample end-effector states around the handle
        skill_id = 3
        precond = data[skill_id]
        __import__('ipdb').set_trace()
        # __import__('ipdb').set_trace()
        # states = precond.sample(FrankaDoorEnv, 25, env)
        states = precond.sample(FrankaDoorEnv, 50)
        # states = precond.sample(FrankaDoorEnv, 25)
        for state in states:
            obs = env.reset_from_state(state)
            print(f"Success prob: {np.around(precond.prob(obs), 3)}")
            env.render_by(25)

    elif cfg.view_learnt_preconds_probs:
        # skill_id = 0
        # skill_ids = [0, 1, 2]
        skill_ids = [0]
        ax = plt.axes(projection='3d')
        ax.set_xlim3d(-0.25, 0.25)
        ax.set_ylim3d(-0.25, 0.25)
        ax.set_zlim3d(-0.25, 0.25)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        def sample_states_around_handle(ref_state, n):
            samples_x = np.random.uniform(low=-0.1, high=0.1, size=n)
            samples_y = np.random.uniform(low=-0.1, high=0.05, size=n)
            samples_z = np.random.uniform(low=-0.1, high=0.1, size=n)
            X = np.column_stack([samples_x, samples_y, samples_z])
            # X[:, 2] = 0.05
            handle_pos = ref_state['handle:pose/position']

            states = []
            for x in X:
                state = deepcopy(ref_state)
                state['robot_eef:pose/position'] = handle_pos + x
                states.append(state)

            return np.array(states)

        for skill_id in skill_ids:
            precond = data[skill_id]
            # sample end-effector states around the handle
            # sample the joint space dist
            # states = precond.sample(FrankaDoorEnv, 500)
            ref_state = precond.sample(FrankaDoorEnv, 1, env)[0]
            ref_rl_state = env.state_to_rl_state(ref_state)

            rl_states = sample_states_around_handle(ref_rl_state, 500)
            # rl_states = np.array([env.state_to_rl_state(state) for state in states])
            precond_probs = np.array([precond.prob(state) for state in rl_states])
            # rl_states = rl_states[precond_probs > 0.7]

            def plot_states(states, ax=None, **kwargs):
                eef_poss = [state['robot_eef:pose/position'] for state in states]
                handle_poss = [state['handle:pose/position'] for state in states]
                eef_to_handle = np.array([eef - handle for eef, handle in zip(eef_poss, handle_poss)])
                if ax is None:
                    ax = plt.axes(projection='3d')
                # ax.plot([-0.03, 0.07], [0.0, 0.0], [0.0, 0.0], color='k', linewidth=5)
                ax.plot([0], [0], [0], 'kx')
                scatter = ax.scatter(eef_to_handle[:, 0], eef_to_handle[:, 1], eef_to_handle[:, 2],
                        **kwargs)
                return scatter

            scatter = plot_states(rl_states, ax=ax, c=precond_probs, cmap='RdYlGn', vmin=0, vmax=1)
            # scatter = plot_states(rl_states, ax=ax) #, vmin=0, vmax=1)
        # plt.colorbar(scatter)
        plt.show()


    elif cfg.view_precond_prediction:
        clfs = data
        subgoal_id = 1
        clf = clfs[subgoal_id]

        pos_site_config = {
            "name": 'pos',
            "type": "sphere",
            "size": [0.03],
            "rgba": [0, 0, 1, 0.5],
        }
        neg_site_config = {
            "name": 'neg',
            "type": "sphere",
            "size": [0.03],
            "rgba": [1, 0, 0, 0.5],
        }

        env = VisualizationWrapper(
            FrankaDoorEnv(
                controller_configs=controller_cfg,
                has_renderer=cfg.render,
                horizon=env_cfg.horizon,
                context_cfg=env_cfg.context,
                obs_uncertainty=env_cfg.obs_uncertainty,
                timestep=env_cfg.timestep,
                eef_start_region_cfg=env_cfg.eef_start_region,
            ),
            [pos_site_config, neg_site_config])
        env.set_obs_corruption(False)

        pos_recoveries = pkl_load(
            'data/door_opening/debug/recovery_skills/recovery_skills.pkl', True)[subgoal_id]
        neg_recoveries = pkl_load(
            'data/door_opening/debug/recovery_skills/failed_recovery_skills.pkl', True)[subgoal_id]
        pos_recovery_states = [recovery.start for recovery in pos_recoveries]
        neg_recovery_states = [recovery.start for recovery in neg_recoveries]
        recovery_states = pos_recovery_states + neg_recovery_states
        np.random.shuffle(recovery_states)

        pos_X = [state.as_array() for state in pos_recovery_states]
        neg_X = [state.as_array() for state in neg_recovery_states]
        X = [state.as_array() for state in recovery_states]

        pos_preds = clf.predict(pos_X)
        neg_preds = clf.predict(neg_X)
        preds = clf.predict(X)

        # for state, pred in zip(pos_recovery_states, pos_preds):
        for state, pred in zip(recovery_states, preds):
            obs = env.reset_from_state(state)
            if pred == 1:
                env.set_indicator_pos('pos', obs['robot_eef:pose/position'])
            else:
                env.set_indicator_pos('neg', obs['robot_eef:pose/position'])

            env.sim.step()
            for _ in range(100):
                env.render()

        # ref_state = recovery_states[0]


        # eef_region_cfg = cfg.eef_region

        # def sample_eef_perturbation():
            # x = sample(eef_region_cfg.x)
            # y = sample(eef_region_cfg.y)
            # z = sample(eef_region_cfg.z)

            # roll = sample(eef_region_cfg.roll)
            # pitch = sample(eef_region_cfg.pitch)
            # yaw = sample(eef_region_cfg.yaw)

            # eef_pos = [x, y, z]
            # eef_quat = T.mat2quat(T.euler2mat([roll, pitch, yaw]))
            # return eef_pos, eef_quat

        # # generate states with same door pose and 
        # states = []
        # for i in range(cfg.nevals):
            # state = deepcopy(ref_state)
            # eef_pos, eef_quat = sample_eef_perturbation()
            # state['robot_eef:pose/position'] = eef_pos
            # state['robot_eef:pose/quat'] = eef_quat

    elif cfg.view_failure_clusters:
        clusters = data

        __import__('ipdb').set_trace()
        for cluster_id, cluster in enumerate(clusters):
        # for cluster_id, cluster in enumerate(clusters[-1:]):
            logger.info(f"Cluster {cluster_id}")
            for state in cluster[:5]:
            # for state in cluster:
                obs = env.reset_from_state(state)
                logger.info(f"Hinge: {obs['hinge:pose/theta']}")
                env.render_by(40)


if __name__ == "__main__":
    main()
