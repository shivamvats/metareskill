import argparse
import sys

import matplotlib.pyplot as plt
from recovery_skills.utils import *
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import *
from sklearn.decomposition import *
from sklearn.preprocessing import *
from sklearn.pipeline import Pipeline
from recovery_skills.utils.data_processing import *
from recovery_skills.graph.failure_state_classifier import *


def parse_arguments(input_args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i",
        "--input-filename",
        type=str,
        required=True,
        help="Input file path"
    )
    parser.add_argument(
        "-o",
        "--outpur-dir",
        type=str,
        default="./data/door_opening/debug/failures",
        help="Output directory"
    )
    args = parser.parse_args(input_args)
    return args


def plot_states(states, ax=None, **kwargs):
    eef_poss = [state['robot_eef:pose/position'] for state in states]
    handle_poss = [state['handle:pose/position'] for state in states]
    eef_to_handle = np.array([eef - handle for eef, handle in zip(eef_poss, handle_poss)])
    thetas = np.array([state['handle:pose/theta'] for state in states])
    if ax is None:
        ax = plt.axes(projection='3d')
    # ax.plot([-0.03, 0.07], [0.0, 0.0], [0.0, 0.0], color='k', linewidth=5)
    if len(eef_to_handle) >=    2 :
        ax[0].scatter(eef_to_handle[:, 0], eef_to_handle[:, 1], eef_to_handle[:, 2],
                **kwargs)
        ax[1].scatter(eef_to_handle[:, 0], eef_to_handle[:, 1], thetas,
                **kwargs)
    ax[0].plot([0], [0], [0], 'kX', markersize=8)
    ax[1].plot([0], [0], [0], 'kX', markersize=8)


def filter_outliers(states):
    sub_states = [state['robot_eef:pose/position'] - state['handle:pose/position'] for state in states]
    X = np.array(sub_states)
    inliers = states[X[:, 2] < 0.06]
    return inliers

def cluster_failures(failures):
    """
    Cluster based on eef position wrt the handle

    Cluster using GMM"""
    if len(failures) == 0:
        return None

    # states = np.array([failure['obs'] for failure in failures])
    states = []
    # for failure in failures:
        # state = failure['obs']
        # state['sim_state'] = failure['sim_state']
        # states.append(state)
    states = failures
    states = np.array(states)
    states = filter_outliers(states)
    X = states
    states_door_closed, states_door_open = split_based_on_hinge_angle(states)
    print(f"Door closed: {len(states_door_closed)}")
    print(f"Door open: {len(states_door_open)}")

    all_failure_clusters = []
    all_nclusters = [3, 2]
    # all_nclusters = [10, 2]
    gmms = []
    for X, nclusters in zip([states_door_closed, states_door_open],
                            all_nclusters):
        failure_clusters = []
        states = X
        # X = np.array([state.as_array() for state in states])
        # X = sub_states

        # pca = PCA(n_components='mle').fit(X)
        # pca = PCA(n_components=11).fit(X)
        # X = pca.transform(X)
        # vars = pca.explained_variance_ratio_
        # print(f"Explained var ratios: {vars}")

        # plot_states(states)
        # kmeans = KMeans(n_clusters=nclusters).fit(X)
        # dbscan = DBSCAN(eps=0.01, min_samples=5).fit(X)
        # meanshift = MeanShift().fit(X)
        scores = []
        # for nclusters in np.arange(5, 20):
        # for nclusters in np.arange(10, 11):
        gmm = Pipeline([
                ('select_vars', SelectVarsTransformer()),
                ('preprocess', StandardScaler()),
                ('gmm', GaussianMixture(n_components=nclusters,
                                        covariance_type='full',
                                        n_init=50))
                # ('gmm', BayesianGaussianMixture(n_components=nclusters,
                                        # covariance_type='full',))
                # ('dbscan', DBSCAN(eps=0.7, min_samples=10))
                # ('dbscan', DBSCAN(eps=1.5, min_samples=5))
            ])
        gmm.fit(X)
        gmms.append(gmm)
        # score = gmm.score(X)
        # scores.append(score)
        # print(f"nclusters: {nclusters}, score: {score}")

        # gmm = gmms[np.argmax(scores)]
        # weights = gmm['gmm'].weights_
        preds = gmm.predict(X)
        # preds = gmm['dbscan'].labels_

        cluster_ids, cluster_sizes = np.unique(preds, return_counts=True)
        for cluster_id in cluster_ids:
            if cluster_id == -1:
                continue
            fs = states[preds == cluster_id]
            failure_clusters.append(fs)

        all_failure_clusters.append(failure_clusters)
        noise = list(preds).count(-1)
        print(f"# noisy samples: {noise}")
        print(f"cluster sizes: {cluster_sizes}")

        # outlier_ids = np.where(weights < 0.01)[0]

        # preds = kmeans.predict(X)
        # preds = dbscan.labels_
        # preds = meanshift.labels_

        # inliers = []
        # for x, label in zip(X, preds):
            # if label not in outlier_ids:
                # inliers.append(x)
        # X = np.array(inliers)

        # gmm.fit(X)

        nclusters = len(np.unique(preds))
        print(f"# clusters: {nclusters}")

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('z')
        ax = [ax1, ax2]

        # colors = ['r', 'g', 'b']
        # colors = plt.cm.rainbow(np.linspace(0, 1, nclusters))

        for cluster_id in range(nclusters):
            cluster_states = states[preds == cluster_id]
            plot_states(cluster_states, ax, alpha=0.5) #, color=colors[cluster_id])
        plot_states(states[preds == -1], ax, c='k', alpha=0.1) #, color=colors[cluster_id])

        plt.show()

    pkl_dump(gmms, 'gmms.pkl')

    return all_failure_clusters, gmms


def main(input_args):
    args = parse_arguments(input_args)
    all_failures = pkl_load(args.input_filename)
    # failure_clusters = []

    # failures = np.concatenate(all_failures)
    failures = all_failures
    # __import__('ipdb').set_trace()
    # for failures in all_failures:
    all_clusters, gmms = cluster_failures(failures)
    clusters = np.concatenate(all_clusters)
    # failure_clusters.append(clusters)

    for cluster in clusters:
        print(len(cluster))

    cluster_sizes = []
    for clusters in all_clusters:
        sizes = [len(cluster) for cluster in clusters]
        cluster_sizes.append(sizes)
    failure_classifier = FailureStateClassifier(gmms, cluster_sizes)

    i = 0
    for clusters in all_clusters:
        for cluster in clusters:
            state = cluster[0]
            pred = failure_classifier.predict(state)
            i = i+1
            logger.info(f"i: {i}, pred: {pred}")

    pkl_dump(failure_classifier, "failure_classifier.pkl")

    all_clusters = np.concatenate(all_clusters)
    pkl_dump(all_clusters, 'failure_clusters.pkl')



if __name__ == "__main__":
    main(sys.argv[1:])
