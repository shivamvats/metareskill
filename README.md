# Installation Instructions

## Dependencies
1. Download [MuJoCo200](https://www.roboti.us/download.html) if you want to run the door opening environment.

## Steps
1. Clone this repository with ``git clone --recurse-submodules``.
2. ``cd metareskill``
3. ``pip install -e robosuite``
4. ``pip install -e rl-utils``
5. ``pip install -e .``

# Training

## Learning Preconditions
Before learning the recovery skills, you need to learn task sub-goals. This
requires a set of nominal controllers. You might need to do steps 1-4 a
few times to learn good subgoal classifiers.

1. **Generate subgoals from nominal skill chain**

`python scripts/learn_skill_chain_preconditions.py intiailize=True`

- **Input** - Nominal skill chain

- **Output** - `all_subgoals.pkl` + `all_gt_subgoals.pkl` + `all_labels.pkl`

Move the outputs to `data/door_opening/final/nominal_skills/subgoals/<dir>`

2. **Train subgoal classifiers**

`python scripts/learn_subgoals.py`

- **Input** - `all_subgoals.pkl` + `all_gt_subgoals.pkl` + `all_labels.pkl`

- **Output** - `subgoals.pkl`

Move the generated `subgoals.pkl` to `nominal_skills/subgoals`

3. **Learn skill preconditions iteratively by chaining**

`python scripts/learn_skill_chain_preconditions.py finetune=True skill_ids=[2]`

- **Input** - `subgoals.pkl`

- **Output** - `learnt_preconds.pkl` + `learnt_gt_preconds.pkl`

Move the generated files to `nominal_skills/preconds/` and iteratively train
the preconditions back from the goal to the start.

4. **Verify the preconditions**

- `python scripts/visualize_sim_states.py view_learnt_preconds_probs=True`

- `python scripts/visualize_sim_states.py view_backchaining_states=True`

## Failure Discovery

5. **Collect failures**

`python scripts/evaluate_nominal_skills.py`

- **Input** - Nominal skill chain + init sets

- **Output** - `failures.pkl`

6. **Cluster failures**

`python scripts/cluster_failures.py -i <path_to_failures.pkl>`

- **Input** - list of failures

- **Output** - list of GMM models

## Recovery Learning

7. **Learn recovery skills**

`python scripts/learn_recovery_skills.py mode=train`

- **Input** - failures.pkl + Nominal skill chain + failure clusters

- **Output** - `recovery_skills.pkl`
