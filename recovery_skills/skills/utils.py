from os.path import join
import logging
from sklearn.utils import *
from recovery_skills.skills.nominal_door_opening_skills import (
    ReachAndGraspHandleSkill,
    RotateHandleSkill,
    PullHandleSkill,
)
from recovery_skills.graph.preconditions import PreconditionClassifier
from recovery_skills.skills import SkillChain
from recovery_skills.utils import *

logger = logging.getLogger(__name__)


def load_nominal_skill_chain(cfg):
    if cfg.use_pretrained_nominal_skill_chain:
        skill_chain = pkl_load(join(cfg.nominal_skills_dir,
                                    cfg.pretrained_nominal_skill_chain),
                               True)

    elif cfg.load_preconds:

        skills = [
            ReachAndGraspHandleSkill(),
            RotateHandleSkill(),
            PullHandleSkill(),
        ]

        if cfg.subgoals.gmm:
            # load preconditions
            subgoals = pkl_load(cfg.path_to_preconds, True)

        else:
            # svm
            init_sets, rl_init_sets = load_preconds(
                join(cfg.nominal_skills_dir, 'preconds'))
            svm_params = cfg.preconds.one_class_svm

            subgoals = []
            if cfg.preconds.one_class:
                logger.info("Training one class SVM preconditions")
                for init_set, rl_init_set in zip(init_sets, rl_init_sets):
                        y = np.ones(len(init_set))
                        preconds = PreconditionClassifier(
                            init_set, rl_init_set, y, one_class_svm_params=svm_params
                        )
                        subgoals.append(preconds)

            else:
                logger.info("Training SVM preconditions")

                for i in range(len(rl_init_sets)):
                    rl_X_pos = rl_init_sets[i]
                    X_pos = init_sets[i]
                    y_pos = np.ones(len(X_pos))
                    rl_X_neg = np.concatenate(rl_init_sets[:i] + rl_init_sets[i+1:])
                    X_neg = np.concatenate(init_sets[:i] + init_sets[i+1:])
                    y_neg = np.zeros(len(X_neg))
                    X = np.concatenate([X_pos, X_neg])
                    rl_X = np.concatenate([rl_X_pos, rl_X_neg])
                    y = np.concatenate([y_pos, y_neg])
                    X, rl_X, y = shuffle(X, rl_X, y)
                    precond = PreconditionClassifier(X, rl_X, y)
                    subgoals.append(precond)

        for i in range(len(skills)):
            skill = skills[i]
            skill.preconds = subgoals[i]
            skill.goal_constraint = subgoals[i + 1]

        skill_chain = SkillChain(skills)
        pkl_dump(skill_chain, 'trained_skill_chain.pkl')

    else:

        skills = [
            ReachAndGraspHandleSkill(),
            RotateHandleSkill(),
            PullHandleSkill(),
        ]
        skill_chain = SkillChain(skills)

    return skill_chain
