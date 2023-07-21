import numpy as np
import os
import pickle
import pytest

import rl_utils

# ---------------------------------------------------------

def test_reps():

    correct_thresh = 1e-5

    # Load in ground truth test data.
    # Assume it's located in the same directory as this file
    test_file = 'test_reps_data.pkl'
    test_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  test_file)
    with open(test_file_path, 'rb') as f:
        test_data = pickle.load(f)

    # Create REPS object
    reps = rl_utils.Reps(rel_entropy_bound=test_data['rel_entropy_bound'],
                         min_temperature=test_data['min_temperature'])

    # Test reps_weights_from_rewards function (without object)
    weights, temperature = rl_utils.reps_weights_from_rewards(
        test_data['rewards'],
        test_data['rel_entropy_bound'],
        test_data['min_temperature'])
    
    assert np.linalg.norm(weights -
                          test_data['weights_gt']) < correct_thresh
    assert np.linalg.norm(temperature -
                          test_data['temperature_gt']) < correct_thresh

    # Test object version of REPS
    policy_params_mean, policy_params_var, reps_info = \
        reps.policy_from_samples_and_rewards(test_data['policy_param_samples'],
                                             test_data['rewards'])

    assert np.allclose(policy_params_mean, test_data['policy_params_mean_gt'], correct_thresh)
    assert np.allclose(policy_params_var, test_data['policy_params_var_gt'], correct_thresh)
    assert np.allclose(reps_info['weights'], test_data['weights_gt'], correct_thresh)
    assert np.allclose(reps_info['temperature'], test_data['temperature_gt'], correct_thresh)
