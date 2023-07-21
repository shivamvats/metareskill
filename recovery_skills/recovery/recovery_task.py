from dataclasses import dataclass

from recovery_skills.graph import State, PreconditionClassifier


@dataclass
class RecoveryTask(object):
    """
    Contains all the information needed for learning a recovery skill.
    """

    start: State
    goal_constraint: PreconditionClassifier
    goal_id: int
