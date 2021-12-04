import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from benchmark import resolve_benchmark_result_root


def plot(x_axis, y_axis):

    fig, ax = plt.subplots()

    ax.plot(x_axis, y_axis, 'o', color='black')

    return fig, ax


def benchmark_plot(iter_number: int):

    res_root = resolve_benchmark_result_root()

    npz = np.load(os.path.join(res_root, "%05d_bound_log_count_until_inc.npz" % iter_number))

    fig, ax = plot(npz["x"], npz["y"])
    ax.set_xlabel("Bit length of B")
    ax.set_ylabel("Log count")

    fig.tight_layout()
    fig.savefig(os.path.join(res_root, "%05d_bound_log_count_until_inc.svg" % iter_number))

    # npz = np.load(os.path.join(res_root, "%05d_monadic_decomposable_wb.npz" % iter_number))
    # npz = np.load(os.path.join(res_root, "%05d_monadic_decomposable.npz" % iter_number))


if __name__ == "__main__":

    if len(sys.argv) <= 1:
        print("Usage: python benchmark_plot.py [ITERATION_NUMBER]")
    else:
        iter_number = int(sys.argv[1])
        assert iter_number >= 1
        benchmark_plot(iter_number)
