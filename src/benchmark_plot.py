import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from benchmark import resolve_benchmark_result_root


def load_npz(suffix: str, iter_number: int):
    return np.load(os.path.join(
        resolve_benchmark_result_root(),
        ("%05d_" % iter_number) + suffix + ".npz"
    ))


def save_as_img(fig, suffix: str, iter_number: int):
    fig.tight_layout()
    fig.savefig(os.path.join(
        resolve_benchmark_result_root(),
        ("%05d_" % iter_number) + suffix + ".svg"
    ))


def simple_plot(x_axis, y_axis):

    fig, ax = plt.subplots()

    ax.plot(x_axis, y_axis, 'o', color='black')

    return fig, ax


def benchmark_plot(iter_number: int):

    with load_npz("bound_log_count_until_inc", iter_number) as npz:

        fig, ax = simple_plot(npz["x"], npz["y"])
        ax.set_xlabel("Bit length of B")
        ax.set_ylabel("Average k such that the result with bound log^k(B) is consistent")

        save_as_img(fig, "bound_log_count_until_inc", iter_number)

    with load_npz("bound_log_count_until_inc_r", iter_number) as npz:

        fig, ax = simple_plot(npz["x"], npz["y"])
        ax.set_xlabel("k such that the result with bound log^k(B) is consistent")
        ax.set_ylabel("Average bit length of B")

        save_as_img(fig, "bound_log_count_until_inc_r", iter_number)

    with load_npz("md_file_size", iter_number) as npz:

        fig, ax = simple_plot(npz["x"], npz["y"])
        ax.set_xlabel("Monadic decomposition performance (ms)")
        ax.set_ylabel("Average .smt2 file size containing formula (bytes)")

        save_as_img(fig, "md_file_size", iter_number)

    with load_npz("md_file_size_r", iter_number) as npz:

        fig, ax = simple_plot(npz["x"], npz["y"])
        ax.set_xlabel(".smt2 file size containing formula (bytes)")
        ax.set_ylabel("Average monadic decomposition performance (ms)")

        save_as_img(fig, "md_file_size_r", iter_number)


if __name__ == "__main__":

    if len(sys.argv) <= 1:
        print("Usage: python benchmark_plot.py [ITERATION_NUMBER]")
    else:
        iter_number = int(sys.argv[1])
        assert iter_number >= 1
        benchmark_plot(iter_number)
