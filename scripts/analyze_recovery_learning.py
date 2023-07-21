from os.path import join

import matplotlib.pyplot as plt
from recovery_skills.utils import *


def main():
    # path_to_results = 'data/door_opening/debug/recovery_skills/8-Sep/rr/budget-300'
    # path_to_results = 'data/door_opening/debug/recovery_skills/9-Sep/rr/budget-500'
    # path_to_results = 'data/door_opening/debug/recovery_skills/13-Sep/rr/0'
    # path_to_results = 'data/door_opening/debug/recovery_skills/13-Sep/rr/budget-400'
    path_to_results = 'outputs/2022-09-14/21-57-49'
    val_accs = pkl_load(join(path_to_results, 'val_accuracy_hist.pkl'), True)
    all_accs = []

    for accs in val_accs:
        all_accs += accs

    # pi = np.array([0, 3, 0, 3, 3])
    # for accs, act in zip(val_accs, pi):
        # all_accs.append(accs[act])

    max_len = max(len(accs) for accs in all_accs)
    max_ids = []

    plt.title("Transition Probs")
    for i, accs in enumerate(all_accs):
        if not len(accs):
            accs = np.zeros(max_len)
        else:
            accs = np.pad(accs, (0, max_len - len(accs)), constant_values=accs[-1])
            accs = np.pad(accs, (1, 0), constant_values=0)
        max_id = np.argmax(accs)
        max_val = accs[max_id]
        if accs[-1] > 0.05:
            max_ids.append(max_id)
            plt.plot(accs, label=f"{i}")
            plt.vlines(max_id, 0, max_val, color='k', linestyles='--')
        # else:
            # plt.plot(accs)
    print("Max ids: ", max_ids)

    plt.legend()

    plt.show()

    plt.title("Delta Transition Probs")
    for i, accs in enumerate(all_accs):
        if not len(accs):
            accs = np.zeros(max_len)
        print(accs)
        delta_accs = []
        for j in range(1, len(accs)):
            delta_accs.append(max(accs[j] - accs[j-1], 0.0))
        delta_accs = np.array(delta_accs)
        if any(delta_accs > 0.01):
            plt.plot(delta_accs, label=f"{i}")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
