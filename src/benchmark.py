import bisect
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from z3 import *

from presmondec import monadic_decomposable, monadic_decomposable_without_bound, compute_bound, MonDecTestFailed
from utils import get_formula_variables


def _resolve_benchmark_root():
    base_path = Path(__file__).parent
    return (base_path / "../benchmark").resolve()


def remove_unparsable_benchmarks():

    removed_count = 0
    remaining_count = 0

    for root, dirs, files in os.walk(_resolve_benchmark_root()):
        for file in files:
            full_file_path = os.path.join(root, file)

            assert os.path.isfile(full_file_path)

            try:
                parse_smt2_file(full_file_path)
            except Z3Exception:
                print("[LOG]: Removing file '%s'" % full_file_path)
                os.remove(full_file_path)
                removed_count += 1
                continue

            remaining_count += 1

    print("[LOG]: Done - removed: %d remaining: %d" % (removed_count, remaining_count))


def benchmark_smts():

    for root, dirs, files in os.walk(_resolve_benchmark_root()):
        for file in files:
            full_file_path = os.path.join(root, file)

            assert os.path.isfile(full_file_path)

            try:
                yield parse_smt2_file(full_file_path), full_file_path
            except Z3Exception:
                print("[WARN]: Could not parse benchmark file '%s'" % full_file_path)


class NonLinearAverageValuePlotter:

    def __init__(self):
        self._x_value_to_x_axis_index = {}
        self._x_axis = []
        self._y_running_average_values = []

    def add_point(self, x, y):
        # using binary search, find the appropriate place
        # for the value of x on the x axis
        x_axis_index = bisect.bisect_left(self._x_axis, x)

        if x_axis_index >= len(self._x_axis) or self._x_axis[x_axis_index] != x:
            # then x value is not yet known

            self._x_axis[x_axis_index:x_axis_index] = [x]
            self._x_value_to_x_axis_index[x] = x_axis_index

            # -1 is needed to prevent division by zero in case there
            # exists an x-value without a y-value
            self._y_running_average_values[x_axis_index:x_axis_index] = [(0, -1)]

        v, n = self._y_running_average_values[x_axis_index]
        self._y_running_average_values[x_axis_index] = v + y, 1 if n == -1 else n + 1

    def plot(self):
        fig, ax = plt.subplots()

        x_axis = np.array(self._x_axis)
        y_axis = np.array([v / n for v, n in self._y_running_average_values])

        ax.plot(x_axis, y_axis, 'o', color='black')

        return fig, ax


class BenchmarkContext:

    def __init__(self, rounds_limit=0):
        self._rounds_limit = rounds_limit

        self._round_no = 0
        self._cur_phi = None
        self._cur_phi_var_count = None
        self._cur_smt_path = None
        self._cur_phi_var = None
        self._inconsistencies = []

        self._stat_var_count_bound = NonLinearAverageValuePlotter()
        self._stat_var_count_md_without_bound = NonLinearAverageValuePlotter()
        self._stat_var_count_md = NonLinearAverageValuePlotter()

    def update_state(self, cur_phi=None, cur_phi_var_count=None, smt_path=None, cur_phi_var=None):

        if cur_phi is not None:
            self._cur_phi = cur_phi

        if cur_phi_var_count is not None:
            self._cur_phi_var_count = cur_phi_var_count

        if smt_path is not None:
            self._cur_smt_path = smt_path

        if cur_phi_var is not None:
            self._cur_phi_var = cur_phi_var

    def _assert_formula_state_defined(self):
        assert self._cur_phi is not None
        assert self._cur_phi_var_count is not None
        assert self._cur_smt_path is not None

    def _assert_formula_variable_state_defined(self):
        self._assert_formula_state_defined()
        assert self._cur_phi_var is not None

    def report_bound(self, bound):
        self._assert_formula_state_defined()

        self._stat_var_count_bound.add_point(bound, self._cur_phi_var_count)

    def report_monadic_decomposable_without_bound_perf(self, nanos):

        self._assert_formula_variable_state_defined()

        if nanos is None:
            return

        ms = nanos / 1e6

        self._stat_var_count_md_without_bound.add_point(ms, self._cur_phi_var_count)

    def report_monadic_decomposable_perf(self, nanos):

        self._assert_formula_variable_state_defined()

        if nanos is None:
            return

        ms = nanos / 1e6

        self._stat_var_count_md.add_point(ms, self._cur_phi_var_count)

    def report_mondec_results(self, monadic_decomposable, monadic_decomposable_without_bound):
        self._assert_formula_variable_state_defined()

        if monadic_decomposable != monadic_decomposable_without_bound:
            self._inconsistencies.append((self._cur_smt_path, self._cur_phi_var, self._cur_phi))

    def next_round(self) -> bool:

        self._round_no += 1

        if self._rounds_limit != 0 and self._round_no == self._rounds_limit:
            return True

        assert self._rounds_limit == 0 or self._round_no < self._rounds_limit

        if self._round_no % 5 == 0:
            print("[LOG]: Inconsistencies so far: %5d" % len(self._inconsistencies))
            print("[LOG]: Starting round: %5d" % self._round_no)

        return False

    def export_graphs(self):

        fig, ax = self._stat_var_count_bound.plot()

        ax.set_xlabel("Bound B")
        ax.set_ylabel("Average variable count")

        fig.tight_layout()
        fig.savefig("../benchmark_results/var_count_bound.svg")

        fig, ax = self._stat_var_count_md_without_bound.plot()

        ax.set_xlabel("monadic_decomposable_without_bound performance (ms)")
        ax.set_ylabel("Average variable count")

        fig.tight_layout()
        fig.savefig("../benchmark_results/monadic_decomposable_without_bound.svg")

        fig, ax = self._stat_var_count_md.plot()

        ax.set_xlabel("monadic_decomposable performance (ms)")
        ax.set_ylabel("Average variable count")

        fig.tight_layout()
        fig.savefig("../benchmark_results/monadic_decomposable.svg")

    def print_inconsistencies(self):
        print("[LOG]: Inconsistencies: %d" % len(self._inconsistencies))
        for inc in self._inconsistencies:
            print(inc)


def run_benchmark(rounds_limit=0, vars_per_formula_limit=5):

    ctx = BenchmarkContext(rounds_limit)

    print("[LOG]: Benchmark started.")

    for smt, smt_path in benchmark_smts():

        print("[LOG]: Considering formula in '%s'" % smt_path)

        phi = And([f for f in smt])
        phi_vars = [var.unwrap() for var in get_formula_variables(phi)][:vars_per_formula_limit]
        var_count = len(phi_vars)

        phi = And([phi] + [v >= 0 for v in phi_vars])

        b = compute_bound(phi)

        ctx.update_state(phi, var_count, smt_path)

        ctx.report_bound(b)

        for phi_var in phi_vars:

            ctx.update_state(cur_phi_var=phi_var)

            try:
                start_nanos = time.perf_counter_ns()
                dec = monadic_decomposable(phi, phi_var, b)
                end_nanos = time.perf_counter_ns()

                ctx.report_monadic_decomposable_perf(end_nanos - start_nanos)
            except MonDecTestFailed:
                ctx.report_monadic_decomposable_perf(None)
                continue

            try:
                start_nanos = time.perf_counter_ns()
                dec_without_bound = monadic_decomposable_without_bound(phi, phi_var)
                end_nanos = time.perf_counter_ns()

                ctx.report_monadic_decomposable_without_bound_perf(end_nanos - start_nanos)
            except MonDecTestFailed:
                ctx.report_monadic_decomposable_without_bound_perf(None)
                continue

            ctx.report_mondec_results(dec, dec_without_bound)

        if ctx.next_round():
            break

    print("[LOG]: Benchmarking complete!")

    return ctx


if __name__ == "__main__":

    if "--clean" in sys.argv[1:]:
        remove_unparsable_benchmarks()
    else:
        set_option(timeout=5 * 1000)

        ctx = run_benchmark(rounds_limit=10, vars_per_formula_limit=1)

        ctx.export_graphs()
        ctx.print_inconsistencies()
