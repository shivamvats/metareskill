from .robot_skill import RobotSkill
import numpy as np


class SkillChain(RobotSkill):
    """
    Represents a list of robot skills to be executed in sequence.

        Input
        -----
        * List of ``RobotSkill``

    """

    def __init__(self, robot_skills, goal_constraint=None):
        super().__init__(goal_constraint)

        self.skills = robot_skills

    def precondition_satisfied(self, obs):
        precond_sat = [skill.precondition_satisfied(obs) for skill in self.skills]
        return max(precond_sat)

    def apply_skill(self, env, obs, context=None, **kwargs):
        """Applies the best skill"""

        probs = [skill.preconds.prob(obs) for skill in self.skills]
        skill_id = np.argmax(probs)
        skill = self.skills[skill_id]
        obs, rew, done, info = skill.apply(env, obs, context, **kwargs)
        info['applied_skill'] = skill
        info['applied_skill_id'] = skill_id
        return obs, rew, done, info

    def apply(self, env, obs, context=None, precond_check=True, **kwargs):
        """Applies a chain of skills."""

        obs_hist, rew_hist, done_hist, info_hist = [], [], [], []
        obs, gt_obs = env.obs(), env.obs(ground_truth=True)
        subgoals, gt_subgoals = [obs], [gt_obs]

        for i in range(len(self.skills)):
            skill = self.skills[i]
            state = env.rl_state()
            if precond_check:
                precond_satisfied = skill.precondition_satisfied(state, context)
            else:
                precond_satisfied = True

            if precond_satisfied:
                obs, rew, done, info = skill.apply(env, obs, context, **kwargs)
                subgoal, gt_subgoal = obs, env.obs(ground_truth=True)

                subgoals.append(subgoal)
                gt_subgoals.append(gt_subgoal)

            else:
                rew = None
                done = False
                info = {"is_solved": False}
                break

            obs_hist.append(obs)
            rew_hist.append(rew)
            done_hist.append(done)
            info_hist.append(info)
            # __import__('ipdb').set_trace()

        info["hist"] = dict(obs=obs_hist, rew=rew_hist, done=done_hist, info=info_hist)
        info["subgoals"] = subgoals
        info["gt_subgoals"] = gt_subgoals

        return obs, rew, done, info

    def apply_with_state_estimation(self, env, obs, context=None,
                                    precond_check=True, **kwargs):
        """Assumes there is a state estimation module that converges to the
        true state after the first skill execution."""

        env.set_obs_corruption(True)

        obs_hist, rew_hist, done_hist, info_hist = [], [], [], []
        obs, gt_obs = env.obs(), env.obs(ground_truth=True)
        subgoals, gt_subgoals = [obs], [gt_obs]


        for skill_id, skill in enumerate(self.skills):
            if skill_id > 0:
                # State estimation converges after first skill
                env.set_obs_corruption(False)

            state = env.rl_state()
            if precond_check:
                precond_satisfied = skill.precondition_satisfied(state, context)
            else:
                precond_satisfied = True

            if precond_satisfied:
                obs, rew, done, info = skill.apply(env, obs, context, **kwargs)
                subgoal, gt_subgoal = obs, env.obs(ground_truth=True)

                subgoals.append(subgoal)
                gt_subgoals.append(gt_subgoal)

            else:
                rew = None
                done = False
                info = {"is_solved": False}
                break

            obs_hist.append(obs)
            rew_hist.append(rew)
            done_hist.append(done)
            info_hist.append(info)
            # __import__('ipdb').set_trace()

        info["hist"] = dict(obs=obs_hist, rew=rew_hist, done=done_hist, info=info_hist)
        info["subgoals"] = subgoals
        info["gt_subgoals"] = gt_subgoals

        return obs, rew, done, info

    def goal(self):
        raise NotImplementedError

    def make_policy(self, state, context=None):
        raise NotImplementedError

    @property
    def size(self):
        return len(self.skills)
