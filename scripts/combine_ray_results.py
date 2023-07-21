import argparse
import glob
import logging
from os.path import join
import sys

from recovery_skills.utils import *

logger = logging.getLogger(__name__)
logger.setLevel("INFO")


def parse_arguments(input_args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-d",
        "--root-dir",
        type=str,
        required=True,
        help="Root directory with files"
    )
    args = parser.parse_args(input_args)
    return args


def main(input_args):
    args = parse_arguments(input_args)
    root_dir = args.root_dir

    ray_subdirs = glob.glob(join(root_dir, "*/"), recursive=True)
    logger.info(f"Combining subdirs: {ray_subdirs}")

    combined_gt_init_sets, combined_gt_rl_init_sets = None, None
    for subdir in ray_subdirs:
        init_sets  = pkl_load(join(subdir, 'gt_init_sets.pkl'))
        rl_init_sets = pkl_load(join(subdir, 'gt_rl_init_sets.pkl'))

        if combined_gt_init_sets is None:
            combined_gt_init_sets = [[] for _ in range(len(init_sets))]
            combined_gt_rl_init_sets = [[] for _ in range(len(init_sets))]

        for i, (init_set, rl_init_set) in enumerate(zip(init_sets, rl_init_sets)):
            combined_gt_init_sets[i].extend(init_set)
            combined_gt_rl_init_sets[i].extend(rl_init_set)

    logger.info(f"Total data-points: {len(combined_gt_init_sets[0])}")
    pkl_dump(combined_gt_init_sets, join(root_dir, "combined_gt_init_sets.pkl"))
    pkl_dump(combined_gt_rl_init_sets, join(root_dir, "combined_gt_rl_init_sets.pkl"))


if __name__ == "__main__":
    main(sys.argv[1:])
