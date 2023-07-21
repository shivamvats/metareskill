"""This script clusters and visualizes all subgoals obseved during the
execution of a skill chain with respect to the door handle."""

import argparse
import sys
from os.path import join

import matplotlib.pyplot as plt
from recovery_skills.utils import *
from recovery_skills.envs import *
from sklearn.mixture import GaussianMixture
from sklearn.cluster import *
from sklearn.decomposition import *


def parse_arguments(input_args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default='./data/door_opening/debug/nominal_skills/subgoals/0',
        help="Input files dir"
    )
    args = parser.parse_args(input_args)
    return args


def plot_states(states, ax=None, **kwargs):
    eef_poss = [state['robot_eef:pose/position'] for state in states]
    handle_poss = [state['handle:pose/position'] for state in states]
    eef_to_handle = np.array([eef - handle for eef, handle in zip(eef_poss, handle_poss)])
    if ax is None:
        ax = plt.axes(projection='3d')
    ax.plot([-0.03, 0.07], [0.0, 0.0], [0.0, 0.0], color='k', linewidth=5)
    ax.scatter(eef_to_handle[:, 0], eef_to_handle[:, 1], eef_to_handle[:, 2],
               **kwargs)


def cluster_states(states):
    """Cluster using GMM"""
    X = np.array([state.as_array() for state in states])

    nclusters = 4

    # plot_states(states)
    # kmeans = KMeans(n_clusters=nclusters).fit(X)
    # dbscan = DBSCAN(eps=0.01, min_samples=5).fit(X)
    # meanshift = MeanShift().fit(X)
    gmm = GaussianMixture(n_components=nclusters, covariance_type='full').fit(X)

    # preds = kmeans.predict(X)
    # preds = dbscan.labels_
    # preds = meanshift.labels_
    preds = gmm.predict(X)

    nclusters = len(np.unique(preds))
    print(f"# clusters: {nclusters}")

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # colors = ['r', 'g', 'b']
    colors = plt.cm.rainbow(np.linspace(0.5, 1, nclusters))

    for cluster_id in range(nclusters):
        cluster_states = states[preds == cluster_id]
        plot_states(cluster_states, ax, color=colors[cluster_id])

    plt.show()


def main(input_args):
    args = parse_arguments(input_args)
    all_gt_states = pkl_load(join(args.input_dir, 'all_gt_subgoals.pkl'))
    all_states = pkl_load(join(args.input_dir, 'all_subgoals.pkl'))
    labels = np.array(pkl_load(join(args.input_dir, 'all_labels.pkl')), dtype=bool)

    states = np.concatenate([np.array(s_)[labels] for s_ in all_states])
    states = np.array([FrankaDoorEnv.create_rl_state().from_dict(state) for
                       state in states])

    gt_states = np.concatenate([np.array(s_)[labels] for s_ in all_gt_states])
    gt_states = np.array([FrankaDoorEnv.create_rl_state().from_dict(state) for
                       state in gt_states])
    # pos_states
    clusters = cluster_states(states)
    # plt.savefig('noisy_state_clusters.png')
    clusters = cluster_states(gt_states)
    # plt.savefig('gt_state_clusters.png')


if __name__ == "__main__":
    main(sys.argv[1:])
