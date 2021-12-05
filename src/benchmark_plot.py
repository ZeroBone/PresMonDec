import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from benchmark import resolve_benchmark_result_root


def load_npz(suffix: str, iter_number: int):
    return np.load(os.path.join(
        resolve_benchmark_result_root(),
        ("%05d_" % iter_number) + suffix + ".npz"
    ))


def save_as_img(fig, suffix: str):
    fig.tight_layout()
    fig.savefig(os.path.join(
        resolve_benchmark_result_root(),
        suffix + ".svg"
    ))


def simple_plot(x_axis, y_axis):

    fig, ax = plt.subplots()

    ax.grid()

    ax.plot(x_axis, y_axis, "o", color="blue", alpha=.7)

    return fig, ax


def benchmark_plot(iter_number: int):

    with load_npz("bound_log_count_until_inc", iter_number) as npz:

        fig, ax = simple_plot(npz["x"], npz["y"])
        ax.set_xlabel("Bit length of B")
        ax.set_ylabel("avg(max{k:decomp. with bound log^k(B) is consistent})")

        save_as_img(fig, "bound_log_count_until_inc")

    with load_npz("bound_log_count_until_inc_r", iter_number) as npz:

        fig, ax = simple_plot(npz["x"], npz["y"])
        ax.set_xlabel("max{k:decomposition with bound log^k(B) is consistent}")
        ax.set_ylabel("Average bit length of B")

        save_as_img(fig, "bound_log_count_until_inc_r")

    with load_npz("var_count_bound", iter_number) as npz:

        fig, ax = simple_plot(npz["x"], npz["y"])
        ax.set_xlabel("Variable count")
        ax.set_ylabel("Average bit length of B")

        save_as_img(fig, "var_count_bound")

    div_by_1000_and_round = ticker.FuncFormatter(lambda x, pos: "{:.2f}".format(x / 1000.))

    for subject in ["file_size", "var_count"]:
        with load_npz("md_%s" % subject, iter_number) as md_subject,\
                load_npz("md_wb_%s" % subject, iter_number) as md_wb_subject,\
                load_npz("md_%s_r" % subject, iter_number) as subject_md,\
                load_npz("md_wb_%s_r" % subject, iter_number) as subject_md_wb:

            fig_s_md, ax_s_md = plt.subplots()
            fig_md_s, ax_md_s = plt.subplots()

            for ax, x, y, x_wb, y_wb in [
                (
                    ax_s_md,
                    subject_md["x"],
                    subject_md["y"],
                    subject_md_wb["x"],
                    subject_md_wb["y"]
                ),
                (
                    ax_md_s,
                    md_subject["x"],
                    md_subject["y"],
                    md_wb_subject["x"],
                    md_wb_subject["y"]
                )
            ]:
                ax.grid(zorder=0)

                with_bound = ax.scatter(x, y, alpha=.7, color="blue", zorder=3)
                without_bound = ax.scatter(x_wb, y_wb, alpha=.7, color="orange", zorder=3)

                ax.legend(
                    (with_bound, without_bound),
                    ("With bound", "Without bound"),
                    loc="upper right"
                )

            if subject == "file_size":
                ax_s_md.set_xlabel(".smt2 file size (KB)")
                ax_s_md.set_ylabel("Average monadic decomposition performance (seconds)")

                ax_s_md.xaxis.set_major_formatter(div_by_1000_and_round)
                ax_s_md.xaxis.set_minor_formatter(div_by_1000_and_round)

                ax_s_md.yaxis.set_major_formatter(div_by_1000_and_round)
                ax_s_md.yaxis.set_minor_formatter(div_by_1000_and_round)

                ax_md_s.set_xlabel("monadic decomposition performance (seconds)")
                ax_md_s.set_ylabel("Average .smt2 file size (KB)")

                ax_md_s.xaxis.set_major_formatter(div_by_1000_and_round)
                ax_md_s.xaxis.set_minor_formatter(div_by_1000_and_round)

                ax_md_s.yaxis.set_major_formatter(div_by_1000_and_round)
                ax_md_s.yaxis.set_minor_formatter(div_by_1000_and_round)

            else:
                assert subject == "var_count"

                ax_s_md.set_xlabel("Variable count")
                ax_s_md.set_ylabel("Average monadic decomposition performance (seconds)")

                ax_s_md.yaxis.set_major_formatter(div_by_1000_and_round)
                ax_s_md.yaxis.set_minor_formatter(div_by_1000_and_round)

                ax_md_s.set_xlabel("monadic decomposition performance (seconds)")
                ax_md_s.set_ylabel("Average variable count")

                ax_md_s.xaxis.set_major_formatter(div_by_1000_and_round)
                ax_md_s.xaxis.set_minor_formatter(div_by_1000_and_round)

            save_as_img(fig_s_md, "md_%s_r" % subject)
            save_as_img(fig_md_s, "md_%s" % subject)


if __name__ == "__main__":

    if len(sys.argv) <= 1:
        print("Usage: python benchmark_plot.py [ITERATION_NUMBER]")
    else:
        iter_number = int(sys.argv[1])
        assert iter_number >= 1
        benchmark_plot(iter_number)
