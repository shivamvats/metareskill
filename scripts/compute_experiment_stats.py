from os.path import join
from recovery_skills.utils import *

def main():
    path_to_dir = 'results/door_opening/corl'
    n_experiments = 5
    # baselines
    print("Baselines:")
    print("=============")

    for recovery_strategy in ['open-loop', 'no-recovery', 'retry', 'go_to_prev', 'go_to_start']:
        print("\n")
        print(recovery_strategy)
        print("-------------\n")
        path_to_results = join(path_to_dir, 'baselines', recovery_strategy)
        task_succs, recovery_succs, exec_costs = [], [], []

        for i in range(n_experiments):
            info = pkl_load(join(path_to_results, f'{i}/info.pkl'))
            task_succs += info['success']
            exec_costs += info['rewards']
            recovery_succs += info['recovery']['success']
        task_succs = np.array(task_succs)
        exec_costs = np.array(exec_costs)
        recovery_succs = np.array(recovery_succs)
        succ_exec_costs = exec_costs[task_succs]

        print(f"Success Rate: {np.mean(task_succs)*100}")
        print(f"Recovery Rate: {np.mean(recovery_succs)*100}")
        print(f"Execution cost mean (on success): {np.mean(succ_exec_costs)}")
        print(f"Execution cost std (on success): {np.std(succ_exec_costs)}")

    # Learnt
    print("\n")
    print("Weighted RR")
    print("-------------")
    path_to_results = join(path_to_dir, 'mab, wrr')
    task_succs, recovery_succs, exec_costs = [], [], []

    for i in range(n_experiments):
        info = pkl_load(join(path_to_results, f'{i}/0/info.pkl'))
        task_succs += info['success']
        exec_costs += info['rewards']
        recovery_succs += info['recovery']['success']
    task_succs = np.array(task_succs)
    exec_costs = np.array(exec_costs)
    recovery_succs = np.array(recovery_succs)
    succ_exec_costs = exec_costs[task_succs]

    print(f"Success Rate: {np.mean(task_succs)*100}")
    print(f"Recovery Rate: {np.mean(recovery_succs)*100}")
    print(f"Execution cost mean (on success): {np.mean(succ_exec_costs)}")
    print(f"Execution cost std (on success): {np.std(succ_exec_costs)}")

    print("\n")
    print("Value-MAB")
    print("-------------")
    path_to_results = join(path_to_dir, 'mab, wrr')
    task_succs, recovery_succs, exec_costs = [], [], []

    for i in range(n_experiments):
        info = pkl_load(join(path_to_results, f'{i}/1/info.pkl'))
        task_succs += info['success']
        exec_costs += info['rewards']
        recovery_succs += info['recovery']['success']
        print(np.mean(info['success']),
              np.mean(info['recovery']['success']),
              np.mean(np.array(info['rewards'])[info['success']]))
    task_succs = np.array(task_succs)
    exec_costs = np.array(exec_costs)
    recovery_succs = np.array(recovery_succs)
    succ_exec_costs = exec_costs[task_succs]

    print(f"Success Rate: {np.mean(task_succs)*100}")
    print(f"Recovery Rate: {np.mean(recovery_succs)*100}")
    print(f"Execution cost mean (on success): {np.mean(succ_exec_costs)}")
    print(f"Execution cost std (on success): {np.std(succ_exec_costs)}")


    # ablation
    print("\n")
    print("Round Robin")
    print("-------------")
    path_to_results = join(path_to_dir, 'rr')
    task_succs, recovery_succs, exec_costs = [], [], []

    for i in range(n_experiments):
        info = pkl_load(join(path_to_results, f'{i}/info.pkl'))
        task_succs += info['success']
        exec_costs += info['rewards']
        recovery_succs += info['recovery']['success']
        print(np.mean(info['success']),
              np.mean(info['recovery']['success']),
              np.mean(np.array(info['rewards'])[info['success']]))
    task_succs = np.array(task_succs)
    exec_costs = np.array(exec_costs)
    recovery_succs = np.array(recovery_succs)
    succ_exec_costs = exec_costs[task_succs]

    print(f"Success Rate: {np.mean(task_succs)*100}")
    print(f"Recovery Rate: {np.mean(recovery_succs)*100}")
    print(f"Execution cost mean (on success): {np.mean(succ_exec_costs)}")
    print(f"Execution cost std (on success): {np.std(succ_exec_costs)}")

    print("\n")
    print("Iter-Value-MAB")
    print("-------------")
    path_to_results = join(path_to_dir, 'iter-mab')
    task_succs, recovery_succs, exec_costs = [], [], []

    for i in range(n_experiments):
        info = pkl_load(join(path_to_results, f'{i}/info.pkl'))
        task_succs += info['success']
        exec_costs += info['exec_costs']
        recovery_succs += info['recovery']['success']
        print(np.mean(info['success']),
              np.mean(info['recovery']['success']),
              np.mean(np.array(info['exec_costs'])[info['success']]))
    task_succs = np.array(task_succs)
    exec_costs = np.array(exec_costs)
    recovery_succs = np.array(recovery_succs)
    succ_exec_costs = exec_costs[task_succs]

    print(f"Success Rate: {np.mean(task_succs)*100}")
    print(f"Recovery Rate: {np.mean(recovery_succs)*100}")
    print(f"Execution cost mean (on success): {np.mean(succ_exec_costs)}")
    print(f"Execution cost std (on success): {np.std(succ_exec_costs)}")

    print("\n")
    print("MAB (300)")
    print("-------------")
    path_to_results = join(path_to_dir, 'mab-300')
    task_succs, recovery_succs, exec_costs = [], [], []

    for i in range(n_experiments):
        info = pkl_load(join(path_to_results, f'{i}/info.pkl'))
        task_succs += info['success']
        exec_costs += info['rewards']
        recovery_succs += info['recovery']['success']
    task_succs = np.array(task_succs)
    exec_costs = np.array(exec_costs)
    recovery_succs = np.array(recovery_succs)
    succ_exec_costs = exec_costs[task_succs]

    print(f"Success Rate: {np.mean(task_succs)*100}")
    print(f"Recovery Rate: {np.mean(recovery_succs)*100}")
    print(f"Execution cost mean (on success): {np.mean(succ_exec_costs)}")
    print(f"Execution cost std (on success): {np.std(succ_exec_costs)}")
if __name__ == "__main__":
    main()
