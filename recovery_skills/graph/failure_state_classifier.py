from recovery_skills.utils.data_processing import *


class FailureStateClassifier():
    """Wrapper around preconditions and failures gmms to predict whether a
    state is a failure state or not and its label."""

    def __init__(self, gmms, cluster_sizes):
        self.gmms = gmms
        self.nclusters = [len(gmms[0]['gmm'].weights_),
                          len(gmms[1]['gmm'].weights_)]
        self.cluster_sizes = cluster_sizes
        self.invalid_cluster_ids = []
        for sizes in cluster_sizes:
            invalid = np.argwhere(np.array(sizes) < 10).flatten()
            self.invalid_cluster_ids.append(invalid)

    def predict(self, state):
        state = [state]
        door_closed, door_open = split_based_on_hinge_angle(state)
        if len(door_closed):
            gmm_id = 0
            cluster_id = self.gmms[0].predict(state)
        elif len(door_open):
            gmm_id = 1
            cluster_id = self.gmms[1].predict(state)
        else:
            raise ValueError
        return self.get_failure_id(gmm_id, cluster_id)

    def get_failure_id(self, gmm_id, cluster_id):
        if gmm_id == 0:
            if cluster_id in self.invalid_cluster_ids[0]:
                return -1
            else:
                failure_id = cluster_id
        elif gmm_id == 1:
            if cluster_id in self.invalid_cluster_ids[1]:
                return -1
            else:
                offset = self.nclusters[0] - len(self.invalid_cluster_ids[0])
                failure_id = cluster_id + offset
        else:
            raise ValueError
        return failure_id[0]

    @property
    def num_fail_modes(self):
        return sum(self.nclusters)
