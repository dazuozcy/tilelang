"""A/B arm: baseline single kernel on 3d-multidim-reduce (128,16384) fp16."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench_common as bc

W = [
    w
    for w in bc.tune_workloads("_logsumexp_kernel_single")
    if w["workload_id"] == "3d-multidim-reduce-float16"
][0]


def make_fn():
    run_fn, _ = bc.make_runner(bc.BASELINE_IMPLS["_logsumexp_kernel_single"], W)
    return run_fn
