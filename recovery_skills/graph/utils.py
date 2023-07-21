from copy import deepcopy
import numpy as np

def compute_transition_matrix(counts):
    fail = 10
    # s1, s2, s3, goal, t1, t2, t3, t4, t5
    counts = counts[:, :, 0]
    T = counts / np.sum(counts, axis=1, keepdims=True)

    # onlye the nominal subgoals
    T = T[:4, :]
    # goal
    row = np.zeros(counts.shape[0])
    row[3] = 1.0
    T[3, :] = row

    T_subgoals = T[:, :4]
    T_failures = T[:, 4:]
    T_failures = np.sum(T_failures, axis=1, keepdims=True)
    # s1, s2, s3, goal, fail
    T = np.hstack([T_subgoals, T_failures])

    fail_row = np.zeros(5)
    fail_row[4] = 1.0
    T = np.vstack([T, fail_row])

    # A x S x S
    T = T[np.newaxis, :, :]

    print(T)
    return T

def compute_transition_matrix_belief(counts):
    fail = 10
    # s1, s2, s3, goal, t1, t2, t3, t4, t5
    T = counts / np.sum(counts, axis=1, keepdims=True)
    # goal
    row = np.zeros(counts.shape[0])
    row[3] = 1.0
    T[3, :, 0] = row

    # failures
    row = np.zeros(counts.shape[0])
    row[fail] = 1.0
    for i in range(4, 11):
        T[i, :, 0] = row
        T[i, :, 1] = row

    # low noise start states
    T = np.hstack([T, np.zeros((11, 3, 2))])
    T = np.vstack([T, np.zeros((3, 14, 2))])

    # s1' - > s2'
    T[11, 12, 0] = 1.0
    # s2' -> s3'
    T[12, 13, 0] = 1.0
    # s3' -> goal
    T[13, 3, 0] = 1.0

    # second action
    T[:, :, 1] = 0.0
    T[:, fail, 1] = 1.0

    # A x S x S
    T = np.moveaxis(T, 2, 0)

    print(T)
    return T


def update_transition_matrix(T, probs):
    assert probs.shape == (6, 2, 4)
    T = deepcopy(T)
    probs = np.moveaxis(probs, 1, 0)

    # 4 - 9 are failures states
    failure_ids = np.arange(4, 10)
    # 11 - 13 + 3 are low noise subgoals + goal
    subgoal_ids = np.array([11, 12, 13, 3])

    for i, failure_id  in enumerate(failure_ids):
        T[:, failure_id, subgoal_ids] = probs[:, i, :]

    # for row in T[1]:
        # print(row)
    fail = 10
    # reset fail prob
    T[:, failure_ids, fail] = 0.0
    # normalize
    for i, failure_id in enumerate(failure_ids):
        row_sum = np.sum(T[:, failure_id], axis=1)
        fail_prob = 1.0 - row_sum
        T[:, failure_id, fail] = fail_prob

    for row in T[1]:
        print(row)

    return T
