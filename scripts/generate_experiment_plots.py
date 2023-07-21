import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from recovery_skills.utils import *

sns.set()
# sns.color_palette('Set2')

def main_corl():
    rr_stats_6 = {
        'failure_rate': np.array([28, 27, 34.5, 19, 12]),
        'failure_value': np.array([-2.87, -3.91, -3.84, -3.22, -3.43]),
        'cost_on_success': [0.898, 0.844, 0.838, 0.86, 0.93]
    }
    mab_stats_6 = {
        'failure_rate': np.array([15, 29.5, 24.5, 24.5, 27]),
        'failure_value': np.array([-2.73, -3.31, -3.34, -3.03, -3.31]),
        'cost_on_success': [0.961, 0.861, 0.88, 0.855, 0.848]
    }
    rr_stats_10 = {
        'failure_rate': np.array([21, 25, 29.5, 8.0, 13]),
        'failure_value': np.array([-2.87, -3.91, -3.84, -3.22, -3.43]),
        'cost_on_success': [0.98, 0.93, 0.89, 0.865, 0.995]
    }
    mab_stats_10 = {
        'failure_rate': np.array([11, 22, 22, 17, 14.5]),
        'failure_value': np.array([-2.73, -3.31, -3.34, -3.03, -3.31]),
        'cost_on_success': [0.89, 0.87, 0.96, 1.0, 0.914]
    }
    # __import__('ipdb').set_trace()

    path_rr = 'data/door_opening/debug/recovery_skills/test/rr/0/blackbox_info.pkl'
    path_wrr = 'data/door_opening/debug/recovery_skills/test/wrr/0/blackbox_info.pkl'
    path_mab = 'data/door_opening/debug/recovery_skills/test/iter-mono-mab/1/mab_info.pkl'
    info_rr = pkl_load(path_rr)
    info_wrr = pkl_load(path_wrr)
    info_mab = pkl_load(path_mab)

    rr_alloc= []
    for row in info_rr:
        line = []
        for elem in row:
            line.append(elem['pos_calls'] + elem['neg_calls'])
        rr_alloc.append(line)

    wrr_alloc= []
    for row in info_wrr:
        line = []
        for elem in row:
            line.append(elem['pos_calls'] + elem['neg_calls'])
        wrr_alloc.append(line)
    rr_alloc = np.array(rr_alloc)
    wrr_alloc = np.array(wrr_alloc)
    mab_alloc = np.array(info_mab['arm_pulls']).reshape(6, 4)
    vmax = np.max([np.max(rr_alloc), np.max(wrr_alloc), np.max(mab_alloc)])
    print(f"Vmax: {vmax}")

    __import__('ipdb').set_trace()
    plt.imshow(rr_alloc, vmin=0, vmax=vmax, cmap='Reds')
    plt.savefig("data/door_opening/debug/results/rr_alloc.png")
    plt.imshow(wrr_alloc, vmin=0,vmax=vmax, cmap='Reds')
    plt.savefig("data/door_opening/debug/results/wrr_alloc.png")
    plt.imshow(mab_alloc, vmin=0, vmax=vmax, cmap='Reds')
    plt.savefig("data/door_opening/debug/results/mab_alloc.png")
    # __import__('ipdb').set_trace()

    rr_stats = rr_stats_6
    mab_stats = mab_stats_6

    fig, ax = plt.subplots()
    # ax = axs[0]
    ax.bar(np.arange(5), 100 - mab_stats['failure_rate'], width=0.35, label='Value-MAB',
           alpha=1.)
    ax.bar(np.arange(5) + 0.4, 100 - rr_stats['failure_rate'], width=0.35, label='Weighted RR',
           alpha=1.)
    # ax.set_title("# max actions = 6")
    # ax.set_xlabel("Trial")
    # ax.set_ylabel("Success (%)")
    # fig.legend()
    fig.tight_layout()
    fig.savefig("data/door_opening/debug/results/success_rate_6.png")
    plt.close(fig)

    rr_stats = rr_stats_10
    mab_stats = mab_stats_10
    # ax = axs[1]
    fig, ax = plt.subplots()
    ax.bar(np.arange(5), 100 - mab_stats['failure_rate'], width=0.35, label='Value-MAB',
           alpha=1.0)
    ax.bar(np.arange(5) + 0.4, 100 - rr_stats['failure_rate'], width=0.35, label='Weighted RR',
           alpha=1.0)
    # ax.set_title("# max actions = 10")
    # ax.set_xlabel("Trial")
    # ax.set_ylabel("Success (%)")
    fig.tight_layout()
    # fig.legend()
    fig.savefig("data/door_opening/debug/results/success_rate_10.png")
    # plt.show()
    plt.close(fig)

    # fig, ax = plt.subplots()
    fig, ax = plt.subplots()
    ax.bar(np.arange(5), mab_stats['failure_value'], width=0.35, label='Value-MAB',
           alpha=1.0)
    ax.bar(np.arange(5) + 0.4, rr_stats['failure_value'], width=0.35, label='Weighted RR',
           alpha=1.0)
    # ax.bar([5], [-6], width=0.35, label='No recovery')
    # ax.set_title("Failure Value V(F)")
    # ax.set_xlabel("Trial")
    # ax.set_ylabel("Value")
    fig.legend()
    fig.tight_layout()
    fig.savefig("data/door_opening/debug/results/failure_value.png")
    # plt.show()
    # fig.savefig("data/door_opening/debug/results/failure_value_and_rate.png")
    plt.close()

    # fig, ax = plt.subplots()
    # ax.bar
    # plt.errorbar(x, y, yerr=dy, fmt='o', color='black',
                # ecolor='lightgray', elinewidth=3, capsize=0);


def main():
    path_to_rr = 'data/door_opening/debug/recovery_skills/13-Sep/rr/3'
    path_to_uci = 'data/door_opening/debug/recovery_skills/13-Sep/v-uci/1'
    extent = 1, 4, 1, 5

    info_uci = pkl_load(join(path_to_uci, 'mab_info.pkl'), True)
    info_rr = pkl_load(join(path_to_rr, 'mab_info.pkl'), True)

    vmax = max(np.max(info_uci['arm_pulls']), np.max(info_rr['arm_pulls']))
    vmin = min(np.min(info_uci['arm_pulls']), np.min(info_rr['arm_pulls']))

    uci_alloc = np.array(info_uci['arm_pulls']).reshape((5, 4))
    plt.xticks([], [])
    plt.yticks([], [])
    plt.imshow(uci_alloc, vmin=vmin, vmax=vmax, cmap='Reds')
    plt.savefig("data/door_opening/debug/results/icra/uci_alloc.png")

    rr_alloc = np.array(info_rr['arm_pulls']).reshape((5, 4))
    plt.xticks([], [])
    plt.yticks([], [])
    plt.imshow(rr_alloc, vmin=vmin, vmax=vmax, cmap='Reds', extent=extent)
    plt.savefig("data/door_opening/debug/results/icra/rr_alloc.png")


    uci_planner = pkl_load(join(path_to_uci, 'planner.pkl'), True)
    uci_val = uci_planner.p_recovery().reshape((5, 4))
    plt.xticks([], [])
    plt.yticks([], [])
    plt.imshow(uci_val, vmin=0,vmax=1.0, cmap='Reds')
    plt.savefig("data/door_opening/debug/results/icra/uci_acc.png")


    # RR vs Value-UCI
    # fig, axs = plt.subplots(5)
    rr_best, uci_best = [], []
    for i in range(1, 6):
    # for i in range(1, 2):
        hist_rr = pkl_load(f'data/door_opening/debug/recovery_skills/13-Sep/rr/{i}/failure_value_hist.pkl')
        hist_uci = pkl_load(f'data/door_opening/debug/recovery_skills/13-Sep/v-uci/{i}/failure_value_hist.pkl')
        rr_best.append(max(hist_rr[:100]))
        uci_best.append(max(hist_uci[:100]))

        # ax = axs[i-1]
        fig, ax = plt.subplots()

        # color = next(ax._get_lines.prop_cycler)['color']
        if i == 1:
            ax.plot(np.arange(35, 100), hist_rr[35:100], linestyle='--',
                    label="Round-robin",
                    linewidth=4,
                    # color=color)
                    color='r')
            ax.plot(np.arange(35, 100), hist_uci[35:100], label='Value-UCI',
                    # color=color)
                    linewidth=4,
                    color='b')
            fig.legend(loc='upper center')
        else:
            ax.plot(np.arange(35, 100), hist_rr[35:100], linestyle='--',
                    linewidth=4,
                    color='r')
            ax.plot(np.arange(35, 100), hist_uci[35:100],
                    linewidth=4,
                    color='b')
        # ax.axhline(max(rr_best[:100]), 35, 100, color='k', linestyle='--')
        ax.axhline(max(hist_rr[:100]), color='k', linestyle='--')
        ax.set_xlabel("Episodes")
        # ax.set_ylabel("Failure Value")
        plt.savefig(f'data/door_opening/debug/results/icra/rr_vs_vuci_{i}.png')

    print("RR: ", np.mean(rr_best))
    print(np.round(rr_best, 3))
    print("UCI: ", np.mean(uci_best))
    print(np.round(uci_best, 3))
    print("Difference: ", (np.mean(uci_best) -
                           np.mean(rr_best))/np.mean(uci_best) * -100 )


if __name__ == "__main__":
    main()
