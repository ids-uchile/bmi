import bmi.estimators as estimators
import bmi.estimators.external.r_estimators as r_estimators
import bmi.estimators.external.julia_estimators as julia_estimators
from bmi.benchmark import BENCHMARK_TASKS

from bmi.plot_utils.subplots_from_axsize import subplots_from_axsize
from _common_figure_utils import (
    read_results,
    plot_benchmark_mi_estimate,
)


# === CONFIG ===

ESTIMATORS = {
    'KSG-5': estimators.KSGEnsembleFirstEstimator(neighborhoods=(5,)),
    'KSG-10': estimators.KSGEnsembleFirstEstimator(neighborhoods=(10,)),

    'R-KSG-I-5': r_estimators.RKSGEstimator(variant=1, neighbors=5),
    'R-KSG-I-10': r_estimators.RKSGEstimator(variant=1, neighbors=10),
    'R-KSG-II-5': r_estimators.RKSGEstimator(variant=2, neighbors=5),
    'R-KSG-II-10': r_estimators.RKSGEstimator(variant=2, neighbors=10),

    'Julia-KSG-I-5': julia_estimators.JuliaKSGEstimator(variant=1, neighbors=5),
    'Julia-KSG-II-5': julia_estimators.JuliaKSGEstimator(variant=2, neighbors=5),
}

ESTIMATOR_NAMES = {
    'KSG-5': 'KSG I (n=5, Python)',
    'KSG-10': 'KSG I (n=10, Python)',

    'R-KSG-I-5': 'KSG I (n=5, R)',
    'R-KSG-I-10': 'KSG I (n=10, R)',
    'R-KSG-II-5': 'KSG II (n=5, R)',
    'R-KSG-II-10': 'KSG II (n=10, R)',

    'Julia-KSG-I-5': 'KSG I (n=5, Julia)',
    'Julia-KSG-II-5': 'KSG II (n=5, Julia)',
}

#TASKS = BENCHMARK_TASKS
TASKS = {
    task_id: BENCHMARK_TASKS[task_id]
    for task_id in {
        '1v1-bimodal-0.75',
        'student-dense-1-1-5-0.75',
        'swissroll_x-normal_cdf-1v1-normal-0.75',
        'multinormal-sparse-3-3-2-0.8-0.1',
        'multinormal-sparse-5-5-2-0.8-0.1',
    }
}

N_SAMPLES = [10000]

SEEDS = [0, 1]


# === WORKDIR ===
workdir: "generated/benchmark_ksg/"


# === RULES ===
rule all:
    input: 'figures/mi_estimate.pdf'

rule figure_mi_estimate:
    input: 'results.csv'
    output: 'figures/mi_estimate.pdf'
    run:
        results = read_results(str(input))
        fig, ax = subplots_from_axsize(
            axsize=(len(TASKS) * 0.3, len(ESTIMATORS) * 0.3),
            left=1.8, bottom=2.4
        )
        plot_benchmark_mi_estimate(ax, results, ESTIMATORS, TASKS, ESTIMATOR_NAMES)
        fig.savefig(str(output))

include: "_core_rules.smk"
