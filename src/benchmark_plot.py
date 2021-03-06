import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from benchmark import resolve_benchmark_result_root, \
    PERFORMANCE_GROUPS, \
    PERFORMANCE_GROUP_GENERAL, \
    PERFORMANCE_GROUP_DECOMPOSABLE, \
    PERFORMANCE_GROUP_NON_DECOMPOSABLE


def load_npz(suffix: str, iter_number: int):
    return np.load(os.path.join(
        resolve_benchmark_result_root(),
        ("%05d_" % iter_number) + suffix + ".npz"
    ))


def save_as_img(fig, suffix: str):
    fig.tight_layout()
    fig.savefig(os.path.join(
        resolve_benchmark_result_root(),
        suffix + ".png",
    ), dpi=300)


def simple_plot(x_axis, y_axis):

    fig, ax = plt.subplots()

    ax.grid()

    ax.plot(x_axis, y_axis, "o", color="blue", alpha=.7)

    return fig, ax


def benchmark_plot(iter_number: int):

    print("Loading & plotting data...")

    with load_npz("bound_log_count_until_inc", iter_number) as npz:

        fig, ax = simple_plot(npz["x"], npz["y"][0])
        ax.set_xlabel("Bit length of B")
        ax.set_ylabel("avg(max{k:decomp. with bound log^k(B) is consistent})")

        save_as_img(fig, "bound_log_count_until_inc")

    with load_npz("bound_log_count_until_inc_r", iter_number) as npz:

        x = npz["x"]
        y = npz["y"][0]

        fig, ax = plt.subplots()

        ax.grid(zorder=0)

        ax.plot(x, y, color="blue", alpha=.7, zorder=3)

        ax.set_xlabel("max{k:decomposition with bound log^k(B) is consistent}")
        ax.set_ylabel("Average bit length of B")

        ax.set_yscale("log")

        save_as_img(fig, "bound_log_count_until_inc_r")

    div_by_1000_and_round = ticker.FuncFormatter(lambda x, pos: "{:.2f}".format(x / 1000.))

    with load_npz("var_count_bound", iter_number) as npz:

        x = npz["x"]
        y = npz["y"][0]

        # simple version

        fig, ax = simple_plot(x, y)

        ax.set_xlabel("Variable count")
        ax.set_ylabel("Average bit length of B (Kbits)")

        ax.yaxis.set_major_formatter(div_by_1000_and_round)
        ax.yaxis.set_minor_formatter(div_by_1000_and_round)

        ax.set_xscale("log")

        save_as_img(fig, "var_count_bound_simple")

        # version with cumulative sum

        fig, ax = plt.subplots()

        ax.grid(zorder=0)

        avg_values_legend = ax.scatter(x, y, alpha=.7, color="blue", zorder=3, label="Value")
        cumsum, = ax.plot(x, y.cumsum(), color="orange", zorder=4, label="Cumulative sum")

        ax.legend(
            loc="upper left",
            handles=[avg_values_legend, cumsum]
        )

        ax.set_xlabel("Variable count")
        ax.set_ylabel("Average bit length of B (Kbits)")

        ax.yaxis.set_major_formatter(div_by_1000_and_round)
        ax.yaxis.set_minor_formatter(div_by_1000_and_round)

        ax.set_xscale("log")

        save_as_img(fig, "var_count_bound")

    for subject in ["file_size", "var_count"]:
        with load_npz("md_%s" % subject, iter_number) as md_subject,\
                load_npz("md_wb_%s" % subject, iter_number) as md_wb_subject,\
                load_npz("md_%s_r" % subject, iter_number) as subject_md,\
                load_npz("md_wb_%s_r" % subject, iter_number) as subject_md_wb:

            for group in range(PERFORMANCE_GROUPS):

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

                    x = np.array(x, dtype=float)
                    y = np.array(y[group], dtype=float)

                    x_wb = np.array(x_wb, dtype=float)
                    y_wb = np.array(y_wb[group], dtype=float)

                    with_bound = ax.scatter(x, y, alpha=.7, color="blue", zorder=3)
                    without_bound = ax.scatter(x_wb, y_wb, alpha=.5, color="red", zorder=4)

                    ax.legend(
                        (with_bound, without_bound),
                        ("With bound", "Without bound"),
                        loc="upper left"
                    )

                if group == PERFORMANCE_GROUP_GENERAL:
                    suffix = ""
                    perf = "decomposition performance"
                elif group == PERFORMANCE_GROUP_DECOMPOSABLE:
                    suffix = "_dec"
                    perf = "decomposable proving performance"
                else:
                    assert group == PERFORMANCE_GROUP_NON_DECOMPOSABLE
                    suffix = "_nondec"
                    perf = "non-decomposable proving performance"

                if subject == "file_size":

                    # subject to performance axis

                    ax_s_md.set_xlabel(".smt2 file size (bytes)")
                    ax_s_md.set_ylabel("Average %s (seconds)" % perf)

                    ax_s_md.yaxis.set_major_formatter(div_by_1000_and_round)
                    ax_s_md.yaxis.set_minor_formatter(div_by_1000_and_round)

                    ax_s_md.set_xscale("log")

                    # performance to subject axis

                    ax_md_s.set_xlabel("%s (seconds)" % perf)
                    ax_md_s.set_ylabel("Average .smt2 file size (bytes)")

                    ax_md_s.xaxis.set_major_formatter(div_by_1000_and_round)
                    ax_md_s.xaxis.set_minor_formatter(div_by_1000_and_round)

                    ax_md_s.set_yscale("log")

                else:
                    assert subject == "var_count"

                    # subject to performance axis

                    ax_s_md.set_xlabel("Variable count")
                    ax_s_md.set_ylabel("Average %s (seconds)" % perf)

                    ax_s_md.yaxis.set_major_formatter(div_by_1000_and_round)
                    ax_s_md.yaxis.set_minor_formatter(div_by_1000_and_round)

                    ax_s_md.set_xscale("log")

                    # performance to subject axis

                    ax_md_s.set_xlabel("%s (seconds)" % perf)
                    ax_md_s.set_ylabel("Average variable count")

                    ax_md_s.xaxis.set_major_formatter(div_by_1000_and_round)
                    ax_md_s.xaxis.set_minor_formatter(div_by_1000_and_round)

                    ax_md_s.set_yscale("log")

                save_as_img(fig_s_md, "md_%s_r%s" % (subject, suffix))
                save_as_img(fig_md_s, "md_%s%s" % (subject, suffix))


def benchmark_plot_logk_hist():

    fig, ax = plt.subplots()

    labels = np.array([0, 1, 2, 3, 4])
    counts = np.array([int(input("Enter number of occurrences for k = %d: " % i)) for i in range(5)])

    bars = ax.bar(labels, counts, color="c", alpha=.7, edgecolor="k")
    ax.bar_label(bars)

    ax.set_xlabel("max{k:decomposition with bound log^k(B) is consistent}")
    ax.set_ylabel("Occurrences")

    save_as_img(fig, "log_count_until_inc_dist")


if __name__ == "__main__":

    if len(sys.argv) <= 1:
        print("Usage: python benchmark_plot.py [ITERATION_NUMBER]")
    elif "--logk_dist" in sys.argv[1:]:
        benchmark_plot_logk_hist()
    else:
        iter_number = int(sys.argv[1])
        assert iter_number >= 1
        benchmark_plot(iter_number)
        print("Done.")
