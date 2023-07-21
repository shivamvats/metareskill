from abc import ABC, abstractmethod
from enum import Enum
from itertools import cycle


class Resource(Enum):
    HUMAN = 1
    RL = 2


class TaskAllocator(ABC):
    """
    Given a set of tasks, it allocates resources to a subset of tasks so that
    all of them are solved.
    """

    def __init__(self):
        pass

    @abstractmethod
    def allocate(self, all_tasks: list):
        """
        Order of tasks is important only for book-keeping. We assume the tasks
        are unordered.
        """
        pass


class RoundRobinTaskAllocator(TaskAllocator):
    """
    Cycles through the resources to allocate tasks.
    """

    def __init__(self):
        super().__init__()
        self._resource_cycle = cycle(Resource)

    def allocate(self, all_tasks: list):
        allocations = [next(self._resource_cycle) for _ in all_tasks]
        return allocations


class FixedTaskAllocator(TaskAllocator):
    """Allocates to a fixed resource."""

    def __init__(self, resource):
        self._resource = resource

    def allocate(self):
        return self._resource
