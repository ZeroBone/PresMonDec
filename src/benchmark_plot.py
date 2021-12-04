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

    ax.grid(True)

    ax.plot(x_axis, y_axis, "o", color="blue", alpha=.7)

    return fig, ax


def benchmark_plot(iter_number: int):

    with load_npz("bound_log_count_until_inc", iter_number) as npz:

        fig, ax = simple_plot(npz["x"], npz["y"])
        ax.set_xlabel("Bit length of B")
        ax.set_ylabel("Average k s.t. decomposition with bound log^k(B) is consistent")

        save_as_img(fig, "bound_log_count_until_inc", iter_number)

    with load_npz("bound_log_count_until_inc_r", iter_number) as npz:

        fig, ax = simple_plot(npz["x"], npz["y"])
        ax.set_xlabel("smallest k s.t. decomposition with bound log^k(B) is consistent")
        ax.set_ylabel("Average bit length of B")

        save_as_img(fig, "bound_log_count_until_inc_r", iter_number)

    with load_npz("var_count_bound", iter_number) as npz:

        fig, ax = simple_plot(npz["x"], npz["y"])
        ax.set_xlabel("Variable count")
        ax.set_ylabel("Average bit length of B")

        save_as_img(fig, "var_count_bound", iter_number)

    with load_npz("md_file_size", iter_number) as md_file_size,\
            load_npz("md_wb_file_size", iter_number) as md_wb_file_size,\
            load_npz("md_file_size_r", iter_number) as file_size_md,\
            load_npz("md_wb_file_size_r", iter_number) as file_size_md_wb:

        fig_fs, ax_fs = plt.subplots()
        fig_md, ax_md = plt.subplots()

        for ax, x, y, x_wb, y_wb in [
            (
                ax_fs,
                file_size_md["x"],
                file_size_md["y"],
                file_size_md_wb["x"],
                file_size_md_wb["y"]
            ),
            (
                ax_md,
                md_file_size["x"],
                md_file_size["y"],
                md_wb_file_size["x"],
                md_wb_file_size["y"]
            )
        ]:
            ax.grid(True)

            with_bound = ax.scatter(x, y, alpha=.7, color="blue")
            without_bound = ax.scatter(x_wb, y_wb, alpha=.7, color="orange")

            ax.legend(
                (with_bound, without_bound),
                ("With bound", "Without bound"),
                loc="upper right"
            )

        ax_fs.set_xlabel(".smt2 file size (bytes)")
        ax_fs.set_ylabel("Average monadic decomposition performance (ms)")

        ax_md.set_xlabel("monadic decomposition performance (ms)")
        ax_md.set_ylabel("Average .smt2 file size (bytes)")

        save_as_img(fig_fs, "md_file_size_r", iter_number)
        save_as_img(fig_md, "md_file_size", iter_number)


if __name__ == "__main__":

    if len(sys.argv) <= 1:
        print("Usage: python benchmark_plot.py [ITERATION_NUMBER]")
    else:
        iter_number = int(sys.argv[1])
        assert iter_number >= 1
        benchmark_plot(iter_number)
